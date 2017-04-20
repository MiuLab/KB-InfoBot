'''
'''

import lasagne
import theano
import lasagne.layers as L
import theano.tensor as T
import numpy as np
import sys

from collections import Counter, defaultdict, deque

import random
import cPickle as pkl

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
                sys.stderr.write('Sample out of Bounds!! Probs = {} Sample = {}'.format(probs, x))
                return i-1
        return i

def aggregate_rewards(rewards,discount):
    running_add = 0.
    for t in xrange(1,len(rewards)):
        running_add += rewards[t]*discount**(t-1)
    return running_add

class RLAgent:
    def _init_model(self, in_size, out_size, n_hid=10, learning_rate_sl=0.005, \
            learning_rate_rl=0.005, batch_size=32, ment=0.1):
        # 2-layer MLP
        self.in_size = in_size # x and y coordinate
        self.out_size = out_size # up, down, right, left
        self.batch_size = batch_size
        self.learning_rate = learning_rate_rl
        self.n_hid = n_hid

        input_var, turn_mask, act_mask, reward_var = T.ftensor3('in'), T.imatrix('tm'), \
                T.itensor3('am'), T.fvector('r')

        in_var = T.reshape(input_var, (input_var.shape[0]*input_var.shape[1],self.in_size))

        l_mask_in = L.InputLayer(shape=(None,None), input_var=turn_mask)

        pol_in = T.fmatrix('pol-h')
        l_in = L.InputLayer(shape=(None,None,self.in_size), input_var=input_var)
        l_pol_rnn = L.GRULayer(l_in, n_hid, hid_init=pol_in, mask_input=l_mask_in) # B x H x D
        pol_out = L.get_output(l_pol_rnn)[:,-1,:]
        l_den_in = L.ReshapeLayer(l_pol_rnn, (turn_mask.shape[0]*turn_mask.shape[1], n_hid)) # BH x D
        l_out = L.DenseLayer(l_den_in, self.out_size, nonlinearity=lasagne.nonlinearities.softmax)

        self.network = l_out
        self.params = L.get_all_params(self.network)

        # rl
        probs = L.get_output(self.network) # BH x A
        out_probs = T.reshape(probs, (input_var.shape[0],input_var.shape[1],self.out_size)) # B x H x A
        log_probs = T.log(out_probs)
        act_probs = (log_probs*act_mask).sum(axis=2) # B x H
        ep_probs = (act_probs*turn_mask).sum(axis=1) # B
        H_probs = -T.sum(T.sum(out_probs*log_probs,axis=2),axis=1) # B
        self.loss = 0.-T.mean(ep_probs*reward_var + ment*H_probs)

        updates = lasagne.updates.rmsprop(self.loss, self.params, learning_rate=learning_rate_rl, \
                epsilon=1e-4)

        self.inps = [input_var, turn_mask, act_mask, reward_var, pol_in]
        self.train_fn = theano.function(self.inps, self.loss, updates=updates)
        self.obj_fn = theano.function(self.inps, self.loss)
        self.act_fn = theano.function([input_var, turn_mask, pol_in], [out_probs, pol_out])

        # sl
        sl_loss = 0.-T.mean(ep_probs)
        sl_updates = lasagne.updates.rmsprop(sl_loss, self.params, learning_rate=learning_rate_sl, \
                epsilon=1e-4)

        self.sl_train_fn = theano.function([input_var, turn_mask, act_mask, pol_in], sl_loss, \
                updates=sl_updates)
        self.sl_obj_fn = theano.function([input_var, turn_mask, act_mask, pol_in], sl_loss)

    def train(self, inp, tur, act, rew, pin):
        return self.train_fn(inp, tur, act, rew, pin)

    def evaluate(self, inp, tur, act, rew, pin):
        return self.obj_fn(inp, tur, act, rew, pin)

    def act(self, inp, pin, mode='sample'):
        tur = np.ones((inp.shape[0],inp.shape[1])).astype('int8')
        act_p, p_out = self.act_fn(inp, tur, pin)
        return categorical_sample(act_p.flatten(), mode=mode), act_p.flatten(), p_out

    def sl_train(self, inp, tur, act, pin):
        return self.sl_train_fn(inp, tur, act, pin)

    def sl_evaluate(self, inp, tur, act, pin):
        return self.sl_obj_fn(inp, tur, act, pin)

    def anneal_lr(self):
        self.learning_rate /= 2.
        updates = lasagne.updates.rmsprop(self.loss, self.params, learning_rate=self.learning_rate, \
                epsilon=1e-4)
        self.train_fn = theano.function(self.inps, self.loss, updates=updates)

    def _debug(self, inp, tur, act, rew):
        print 'Input = {}, Action = {}, Reward = {}'.format(inp, act, rew)
        out = self.debug_fn(inp, tur, act, rew)
        for item in out:
            print item

    def _init_experience_pool(self, pool):
        self.input_pool = deque([], pool)
        self.actmask_pool = deque([], pool)
        self.reward_pool = deque([], pool)
        self.turnmask_pool = deque([], pool)

    def add_to_pool(self, inp, turn, act, rew):
        self.input_pool.append(inp)
        self.actmask_pool.append(act)
        self.reward_pool.append(rew)
        self.turnmask_pool.append(turn)

    def _get_minibatch(self, N):
        n = min(N, len(self.input_pool))
        index = random.sample(range(len(self.input_pool)), n)
        i = [self.input_pool[ii] for ii in index]
        a = [self.actmask_pool[ii] for ii in index]
        r = [self.reward_pool[ii] for ii in index]
        t = [self.turnmask_pool[ii] for ii in index]
        return np.asarray(i, dtype='float32'), \
                np.asarray(t, dtype='int32'), \
                np.asarray(a, dtype='int32'), \
                np.asarray(r, dtype='float32')

    def update(self, verbose=False, regime='RL'):
        i, t, a, r = self._get_minibatch(self.batch_size)
        pi = np.zeros((1,self.n_hid)).astype('float32')
        if verbose: print i, t, a, r
        if regime=='RL':
            r -= np.mean(r)
            #r /= np.std(r)
            g = self.train(i,t,a,r,pi)
        else:
            g = self.sl_train(i,t,a,pi)
        return g

    def eval_objective(self, N):
        try:
            obj = self.evaluate(self.eval_i, self.eval_t, self.eval_a, self.eval_r)
        except AttributeError:
            self.eval_i, self.eval_t, self.eval_a, self.eval_r = self._get_minibatch(N)
            obj = self.evaluate(self.eval_i, self.eval_t, self.eval_a, self.eval_r)
        return obj

    def load_model(self, load_path):
        with open(load_path, 'r') as f:
            data = pkl.load(f)
        L.set_all_param_values(self.network, data)

    def save_model(self, save_path):
        data = L.get_all_param_values(self.network)
        with open(save_path, 'w') as f:
            pkl.dump(data, f)
