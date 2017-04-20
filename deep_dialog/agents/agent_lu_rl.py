'''
'''

import lasagne
import theano
import lasagne.layers as L
import theano.tensor as T
import numpy as np
import sys
import time

from deep_dialog import dialog_config

from collections import Counter, defaultdict, deque

import random
import cPickle as pkl

EPS = 1e-10

def categorical_sample(probs, mode='sample'):
    if mode=='max':
        return np.argmax(probs)
    else:
        x = np.random.uniform()
        s = probs[0]
        i = 0
        while s<x:
            i += 1
            try:
                s += probs[i]
            except IndexError:
                sys.stderr.write('Sample out of Bounds!! Probs = {} Sample = {}\n'.format(probs, x))
                return i-1
        return i

def ordered_sample(probs, N, mode='sample'):
    if mode=='max':
        return np.argsort(probs)[::-1][:N]
    else:
        p = np.copy(probs)
        pop = range(len(probs))
        sample = []
        for i in range(N):
            s = categorical_sample(p)
            sample.append(pop[s])
            del pop[s]
            p = np.delete(p,s)
            p = p/p.sum()
        return sample

def aggregate_rewards(rewards,discount):
    running_add = 0.
    for t in xrange(1,len(rewards)):
        running_add += rewards[t]*discount**(t-1)
    return running_add

class E2ERLAgent:
    def _init_model(self, in_size, out_size, slot_sizes, db, \
            n_hid=10, learning_rate_sl=0.005, learning_rate_rl=0.005, batch_size=32, ment=0.1, \
            inputtype='full', sl='e2e', rl='e2e'):
        self.in_size = in_size
        self.out_size = out_size
        self.slot_sizes = slot_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate_rl
        self.n_hid = n_hid
        self.r_hid = self.n_hid
        self.sl = sl
        self.rl = rl

        table = db.table
        counts = db.counts
        m_unk = [db.inv_counts[s][-1] for s in dialog_config.inform_slots]
        prior = [db.priors[s] for s in dialog_config.inform_slots]
        unknown = [db.unks[s] for s in dialog_config.inform_slots]
        ids = [db.ids[s] for s in dialog_config.inform_slots]

        input_var, turn_mask, act_mask, reward_var = T.ftensor3('in'), T.bmatrix('tm'), \
                T.btensor3('am'), T.fvector('r')
        T_var, N_var = T.as_tensor_variable(table), T.as_tensor_variable(counts)
        db_index_var = T.imatrix('db')
        db_index_switch = T.bvector('s')

        l_mask_in = L.InputLayer(shape=(None,None), input_var=turn_mask)
        flat_mask = T.reshape(turn_mask, (turn_mask.shape[0]*turn_mask.shape[1],1))

        def _smooth(p):
            p_n = p+EPS
            return p_n/(p_n.sum(axis=1)[:,np.newaxis])

        def _add_unk(p,m,N):
            # p: B x V, m- num missing, N- total, p0: 1 x V
            t_unk = T.as_tensor_variable(float(m)/N)
            ps = p*(1.-t_unk)
            return T.concatenate([ps, T.tile(t_unk, (ps.shape[0],1))], axis=1)

        def kl_divergence(p,q):
            p_n = _smooth(p)
            return -T.sum(q*T.log(p_n), axis=1)

        # belief tracking
        l_in = L.InputLayer(shape=(None,None,self.in_size), input_var=input_var)
        p_vars = []
        pu_vars = []
        phi_vars = []
        p_targets = []
        phi_targets = []
        hid_in_vars = []
        hid_out_vars = []
        bt_loss = T.as_tensor_variable(0.)
        kl_loss = []
        x_loss = []
        self.trackers = []
        for i,s in enumerate(dialog_config.inform_slots):
            hid_in = T.fmatrix('h')
            l_rnn = L.GRULayer(l_in, self.r_hid, hid_init=hid_in,  \
                    mask_input=l_mask_in,
                    grad_clipping=10.) # B x H x D
            l_b_in = L.ReshapeLayer(l_rnn, 
                    (input_var.shape[0]*input_var.shape[1], self.r_hid)) # BH x D
            hid_out = L.get_output(l_rnn)[:,-1,:]

            p_targ = T.ftensor3('p_target_'+s)
            p_t = T.reshape(p_targ, 
                    (p_targ.shape[0]*p_targ.shape[1],self.slot_sizes[i]))
            phi_targ = T.fmatrix('phi_target'+s)
            phi_t = T.reshape(phi_targ, (phi_targ.shape[0]*phi_targ.shape[1], 1))

            l_b = L.DenseLayer(l_b_in, self.slot_sizes[i], 
                    nonlinearity=lasagne.nonlinearities.softmax)
            l_phi = L.DenseLayer(l_b_in, 1, 
                    nonlinearity=lasagne.nonlinearities.sigmoid)

            phi = T.clip(L.get_output(l_phi), 0.01, 0.99)
            p = L.get_output(l_b)
            p_u = _add_unk(p, m_unk[i], db.N)
            kl_loss.append(T.sum(flat_mask.flatten()*kl_divergence(p, p_t))/T.sum(flat_mask))
            x_loss.append(T.sum(flat_mask*lasagne.objectives.binary_crossentropy(phi,phi_t))/
                    T.sum(flat_mask))
            bt_loss += kl_loss[-1] + x_loss[-1]

            p_vars.append(p)
            pu_vars.append(p_u)
            phi_vars.append(phi)
            p_targets.append(p_targ)
            phi_targets.append(phi_targ)
            hid_in_vars.append(hid_in)
            hid_out_vars.append(hid_out)
            self.trackers.append(l_b)
            self.trackers.append(l_phi)
        self.bt_params = L.get_all_params(self.trackers)

        def check_db(pv, phi, Tb, N):
            O = T.alloc(0.,pv[0].shape[0],Tb.shape[0]) # BH x T.shape[0]
            for i,p in enumerate(pv):
                p_dc = T.tile(phi[i], (1, Tb.shape[0]))
                O += T.log(p_dc*(1./db.table.shape[0]) + \
                        (1.-p_dc)*(p[:,Tb[:,i]]/N[np.newaxis,:,i]))
            Op = T.exp(O)#+EPS # BH x T.shape[0]
            Os = T.sum(Op, axis=1)[:,np.newaxis] # BH x 1
            return Op/Os

        def entropy(p):
            p = _smooth(p)
            return -T.sum(p*T.log(p), axis=-1)

        def weighted_entropy(p,q,p0,unks,idd):
            w = T.dot(idd,q.transpose()) # Pi x BH
            u = p0[np.newaxis,:]*(q[:,unks].sum(axis=1)[:,np.newaxis]) # BH x Pi
            p_tilde = w.transpose()+u
            return entropy(p_tilde)

        p_db = check_db(pu_vars, phi_vars, T_var, N_var) # BH x T.shape[0]
        
        if inputtype=='entropy':
            H_vars = [weighted_entropy(pv,p_db,prior[i],unknown[i],ids[i]) \
                    for i,pv in enumerate(p_vars)]
            H_db = entropy(p_db)
            phv = [ph[:,0] for ph in phi_vars]
            t_in = T.stacklists(H_vars+phv+[H_db]).transpose() # BH x 2M+1
            t_in_resh = T.reshape(t_in, (turn_mask.shape[0], turn_mask.shape[1], \
                    t_in.shape[1])) # B x H x 2M+1
            l_in_pol = L.InputLayer(
                    shape=(None,None,2*len(dialog_config.inform_slots)+1), \
                    input_var=t_in_resh)
        else:
            in_reshaped = T.reshape(input_var, 
                    (input_var.shape[0]*input_var.shape[1], \
                    input_var.shape[2]))
            prev_act = in_reshaped[:,-len(dialog_config.inform_slots):]
            t_in = T.concatenate(pu_vars+phi_vars+[p_db,prev_act], 
                    axis=1) # BH x D-sum+A
            t_in_resh = T.reshape(t_in, (turn_mask.shape[0], turn_mask.shape[1], \
                    t_in.shape[1])) # B x H x D-sum
            l_in_pol = L.InputLayer(shape=(None,None,sum(self.slot_sizes)+ \
                    3*len(dialog_config.inform_slots)+ \
                    table.shape[0]), input_var=t_in_resh)

        pol_in = T.fmatrix('pol-h')
        l_pol_rnn = L.GRULayer(l_in_pol, n_hid, hid_init=pol_in, 
                mask_input=l_mask_in,
                grad_clipping=10.) # B x H x D
        pol_out = L.get_output(l_pol_rnn)[:,-1,:]
        l_den_in = L.ReshapeLayer(l_pol_rnn, 
                (turn_mask.shape[0]*turn_mask.shape[1], n_hid)) # BH x D
        l_out = L.DenseLayer(l_den_in, self.out_size, \
                nonlinearity=lasagne.nonlinearities.softmax) # BH x A

        self.network = l_out
        self.pol_params = L.get_all_params(self.network)
        self.params = self.bt_params + self.pol_params

        # db loss
        p_db_reshaped = T.reshape(p_db, (turn_mask.shape[0],turn_mask.shape[1],table.shape[0]))
        p_db_final = p_db_reshaped[:,-1,:] # B x T.shape[0]
        p_db_final = _smooth(p_db_final)
        ix = T.tile(T.arange(p_db_final.shape[0]),(db_index_var.shape[1],1)).transpose()
        sample_probs = p_db_final[ix,db_index_var] # B x K
        if dialog_config.SUCCESS_MAX_RANK==1:
            log_db_probs = T.log(sample_probs).sum(axis=1)
        else:
            cum_probs,_ = theano.scan(fn=lambda x, prev: x+prev, \
                    outputs_info=T.zeros_like(sample_probs[:,0]), \
                    sequences=sample_probs[:,:-1].transpose())
            cum_probs = T.clip(cum_probs.transpose(), 0., 1.-1e-5) # B x K-1
            log_db_probs = T.log(sample_probs).sum(axis=1) - T.log(1.-cum_probs).sum(axis=1) # B
        log_db_probs = log_db_probs * db_index_switch

        # rl
        probs = L.get_output(self.network) # BH x A
        probs = _smooth(probs)
        out_probs = T.reshape(probs, (turn_mask.shape[0],turn_mask.shape[1],self.out_size)) # B x H x A
        log_probs = T.log(out_probs)
        act_probs = (log_probs*act_mask).sum(axis=2) # B x H
        ep_probs = (act_probs*turn_mask).sum(axis=1) # B
        H_probs = -T.sum(T.sum(out_probs*log_probs,axis=2),axis=1) # B
        self.act_loss = -T.mean(ep_probs*reward_var)
        self.db_loss = -T.mean(log_db_probs*reward_var)
        self.reg_loss = -T.mean(ment*H_probs)
        self.loss = self.act_loss + self.db_loss + self.reg_loss

        self.inps = [input_var, turn_mask, act_mask, reward_var, db_index_var, db_index_switch, \
                pol_in] + hid_in_vars
        self.obj_fn = theano.function(self.inps, self.loss, on_unused_input='warn')
        self.act_fn = theano.function([input_var,turn_mask,pol_in]+hid_in_vars, \
                [out_probs,p_db,pol_out]+pu_vars+phi_vars+hid_out_vars, on_unused_input='warn')
        self.debug_fn = theano.function(self.inps, [probs, p_db, self.loss], on_unused_input='warn')
        self._rl_train_fn(self.learning_rate)

        ## sl
        sl_loss = 0. + bt_loss - T.mean(ep_probs) 

        if self.sl=='e2e':
            sl_updates = lasagne.updates.rmsprop(sl_loss, self.params, \
                    learning_rate=learning_rate_sl, epsilon=1e-4)
            sl_updates_with_mom = lasagne.updates.apply_momentum(sl_updates)
        elif self.sl=='bel':
            sl_updates = lasagne.updates.rmsprop(sl_loss, self.bt_params, \
                    learning_rate=learning_rate_sl, epsilon=1e-4)
            sl_updates_with_mom = lasagne.updates.apply_momentum(sl_updates)
        else:
            sl_updates = lasagne.updates.rmsprop(sl_loss, self.pol_params, \
                    learning_rate=learning_rate_sl, epsilon=1e-4)
            sl_updates_with_mom = lasagne.updates.apply_momentum(sl_updates)

        sl_inps = [input_var, turn_mask, act_mask, pol_in] + p_targets + phi_targets + hid_in_vars
        self.sl_train_fn = theano.function(sl_inps, [sl_loss]+kl_loss+x_loss, updates=sl_updates, \
                on_unused_input='warn')
        self.sl_obj_fn = theano.function(sl_inps, sl_loss, on_unused_input='warn')

    def _rl_train_fn(self, lr):
        if self.rl=='e2e':
            updates = lasagne.updates.rmsprop(self.loss, self.params, learning_rate=lr, epsilon=1e-4)
            updates_with_mom = lasagne.updates.apply_momentum(updates)
        elif self.rl=='bel':
            updates = lasagne.updates.rmsprop(self.loss, self.bt_params, learning_rate=lr, \
                    epsilon=1e-4)
            updates_with_mom = lasagne.updates.apply_momentum(updates)
        else:
            updates = lasagne.updates.rmsprop(self.loss, self.pol_params, learning_rate=lr, \
                    epsilon=1e-4)
            updates_with_mom = lasagne.updates.apply_momentum(updates)
        self.train_fn = theano.function(self.inps, [self.act_loss,self.db_loss,self.reg_loss], \
                updates=updates)

    def train(self, inp, tur, act, rew, db, dbs, pin, hin):
        return self.train_fn(inp, tur, act, rew, db, dbs, pin, *hin)

    def evaluate(self, inp, tur, act, rew, db, dbs, pin, hin):
        return self.obj_fn(inp, tur, act, rew, db, dbs, pin, *hin)

    def act(self, inp, pin, hin, mode='sample'):
        tur = np.ones((inp.shape[0],inp.shape[1])).astype('int8')
        outs = self.act_fn(inp, tur, pin, *hin)
        act_p, db_p, p_out = outs[0], outs[1], outs[2]
        n_slots = len(dialog_config.inform_slots)
        pv = outs[3:3+n_slots]
        phiv = outs[3+n_slots:3+2*n_slots]
        h_out = outs[3+2*n_slots:]
        action = categorical_sample(act_p.flatten(), mode=mode)
        if action==self.out_size-1:
            db_sample = ordered_sample(db_p.flatten(), dialog_config.SUCCESS_MAX_RANK, mode=mode)
        else:
            db_sample = []
        return action, db_sample, db_p.flatten(), p_out, h_out, pv, phiv

    def sl_train(self, inp, tur, act, pin, ptargets, phitargets, hin):
        return self.sl_train_fn(inp, tur, act, pin, *ptargets+phitargets+hin)

    def sl_evaluate(self, inp, tur, act, pin, ptargets, phitargets, hin):
        return self.sl_obj_fn(inp, tur, act, pin, *ptargets+phitargets+hin)

    def anneal_lr(self):
        self.learning_rate /= 2.
        self._rl_train_fn(self.learning_rate)

    def _debug(self, inp, tur, act, rew, beliefs):
        print 'Input = {}, Action = {}, Reward = {}'.format(inp, act, rew)
        out = self.debug_fn(inp, tur, act, rew, *beliefs)
        for item in out:
            print item

    def _init_experience_pool(self, pool):
        self.input_pool = deque([], pool)
        self.actmask_pool = deque([], pool)
        self.reward_pool = deque([], pool)
        self.db_pool = deque([], pool)
        self.dbswitch_pool = deque([], pool)
        self.turnmask_pool = deque([], pool)
        self.ptarget_pool = deque([], pool)
        self.phitarget_pool = deque([], pool)

    def add_to_pool(self, inp, turn, act, rew, db, dbs, ptargets, phitargets):
        self.input_pool.append(inp)
        self.actmask_pool.append(act)
        self.reward_pool.append(rew)
        self.db_pool.append(db)
        self.dbswitch_pool.append(dbs)
        self.turnmask_pool.append(turn)
        self.ptarget_pool.append(ptargets)
        self.phitarget_pool.append(phitargets)

    def _get_minibatch(self, N):
        n = min(N, len(self.input_pool))
        index = random.sample(range(len(self.input_pool)), n)
        i = [self.input_pool[ii] for ii in index]
        a = [self.actmask_pool[ii] for ii in index]
        r = [self.reward_pool[ii] for ii in index]
        d = [self.db_pool[ii] for ii in index]
        ds = [self.dbswitch_pool[ii] for ii in index]
        t = [self.turnmask_pool[ii] for ii in index]
        p = [self.ptarget_pool[ii] for ii in index]
        pp = [np.asarray([row[ii] for row in p], dtype='float32') for ii in range(len(p[0]))]
        ph = [self.phitarget_pool[ii] for ii in index]
        pph = [np.asarray([row[ii] for row in ph], dtype='float32') for ii in range(len(ph[0]))]
        return np.asarray(i, dtype='float32'), \
                np.asarray(t, dtype='int8'), \
                np.asarray(a, dtype='int8'), \
                np.asarray(r, dtype='float32'), \
                np.asarray(d, dtype='int32'), \
                np.asarray(ds, dtype='int8'), \
                pp, pph

    def update(self, verbose=False, regime='RL'):
        i, t, a, r, d, ds, p, ph = self._get_minibatch(self.batch_size)
        hi = [np.zeros((1,self.r_hid)).astype('float32') \
                for s in dialog_config.inform_slots]
        pi = np.zeros((1,self.n_hid)).astype('float32')
        if verbose: print i, t, a, r, d, ds, p, ph, hi
        if regime=='RL':
            r -= np.mean(r)
            al,dl,rl = self.train(i,t,a,r,d,ds,pi,hi)
            g = al+dl+rl
        else:
            g = self.sl_train(i,t,a,pi,p,ph,hi)
        return g

    def eval_objective(self, N):
        try:
            obj = self.evaluate(self.eval_i, self.eval_t, self.eval_a, self.eval_r, self.eval_b)
        except AttributeError:
            self.eval_i, self.eval_t, self.eval_a, self.eval_r, self.eval_b = self._get_minibatch(N)
            obj = self.evaluate(self.eval_i, self.eval_t, self.eval_a, self.eval_r, self.eval_b)
        return obj

    def load_model(self, load_path):
        with open(load_path, 'r') as f:
            data = pkl.load(f)
            L.set_all_param_values(self.network, data)
            for item in self.trackers:
                data = pkl.load(f)
                L.set_all_param_values(item, data)

    def save_model(self, save_path):
        with open(save_path, 'w') as f:
            data = L.get_all_param_values(self.network)
            pkl.dump(data, f)
            for item in self.trackers:
                data = L.get_all_param_values(item)
                pkl.dump(data, f)
