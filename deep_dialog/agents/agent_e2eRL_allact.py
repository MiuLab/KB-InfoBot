'''
'''

import numpy as np
import cPickle as pkl

from deep_dialog import dialog_config, tools
from collections import Counter, defaultdict, deque
from agent_lu_rl import E2ERLAgent, aggregate_rewards
from belief_tracker import BeliefTracker
from softDB import SoftDB
from feature_extractor import FeatureExtractor
from utils import *

import operator
import random
import math
import copy
import re
import nltk
import time

# params
DISPF = 1
SAVEF = 100
ANNEAL = 800

class AgentE2ERLAllAct(E2ERLAgent,SoftDB,BeliefTracker):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, db=None, corpus=None,
            train=True, _reload=False, n_hid=100, batch=128, ment=0., inputtype='full', upd=10, 
            sl='e2e', rl='e2e', pol_start=600, lr=0.005, N=1, tr=2.0, ts=0.5, max_req=2, frac=0.5, 
            name=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.database = db
        self.max_turn = dialog_config.MAX_TURN
        self.training = train
        self.feat_extractor = FeatureExtractor(corpus,self.database.path,N=N)
        out_size = len(dialog_config.inform_slots)+1
        in_size = len(self.feat_extractor.grams) + len(dialog_config.inform_slots)
        slot_sizes = [self.movie_dict.lengths[s] for s in dialog_config.inform_slots]
        self._init_model(in_size, out_size, slot_sizes, self.database, \
                n_hid=n_hid, learning_rate_sl=lr, batch_size=batch, ment=ment, inputtype=inputtype, \
                sl=sl, rl=rl)
        self._name = name
        if _reload: self.load_model(dialog_config.MODEL_PATH+self._name)
        if train: self.save_model(dialog_config.MODEL_PATH+self._name)
        self._init_experience_pool(batch)
        self.episode_count = 0
        self.recent_rewards = deque([], 1000)
        self.recent_successes = deque([], 1000)
        self.recent_turns = deque([], 1000)
        self.recent_loss = deque([], 10)
        self.discount = 0.99
        self.num_updates = 0
        self.pol_start = pol_start
        self.tr = tr
        self.ts = ts
        self.max_req = max_req
        self.frac = frac
        self.upd = upd

    def _print_progress(self,loss,te,*args):
        self.recent_loss.append(loss)
        avg_ret = float(sum(self.recent_rewards))/len(self.recent_rewards)
        avg_turn = float(sum(self.recent_turns))/len(self.recent_turns)
        avg_loss = float(sum(self.recent_loss))/len(self.recent_loss)
        n_suc, n_fail, n_inc, tot = 0, 0, 0, 0
        for s in self.recent_successes:
            if s==-1: n_fail += 1
            elif s==0: n_inc += 1
            else: n_suc += 1
            tot += 1
        if len(args)>0:
            print 'Update %d. Avg turns = %.2f . Avg Reward = %.2f . Success Rate = %.2f . Fail Rate = %.2f . Incomplete Rate = %.2f . Loss = %.3f . Time = %.2f' % \
                    (self.num_updates, avg_turn, avg_ret, \
                    float(n_suc)/tot, float(n_fail)/tot, float(n_inc)/tot, avg_loss, te)
            #print 'kl loss = {}'.format(args[0])
            #print 'x_loss = {}'.format(args[1])
        else:
            print 'Update %d. Avg turns = %.2f . Avg Reward = %.2f . Success Rate = %.2f . Fail Rate = %.2f . Incomplete Rate = %.2f . Loss = %.3f . Time = %.2f' % \
                    (self.num_updates, avg_turn, avg_ret, \
                    float(n_suc)/tot, float(n_fail)/tot, float(n_inc)/tot, avg_loss, te)

    def initialize_episode(self):
        self.episode_count += 1
        if self.training and self.episode_count%self.batch_size==0:
            self.num_updates += 1
            if self.num_updates>self.pol_start and self.num_updates%ANNEAL==0: self.anneal_lr()
            tst = time.time()
            if self.num_updates < self.pol_start: 
                all_loss = self.update(regime='SL')
                loss = all_loss[0]
                kl_loss = all_loss[1:len(dialog_config.inform_slots)+1]
                x_loss = all_loss[len(dialog_config.inform_slots)+1:]
                t_elap = time.time() - tst
                if self.num_updates%DISPF==0: self._print_progress(loss, t_elap, kl_loss, x_loss)
            else: 
                loss = self.update(regime='RL')
                t_elap = time.time() - tst
                if self.num_updates%DISPF==0: self._print_progress(loss, t_elap)
            if self.num_updates%SAVEF==0: self.save_model(dialog_config.MODEL_PATH+self._name)

        self.state = {}
        self.state['database'] = pkl.loads(pkl.dumps(self.database,-1))
        self.state['prevact'] = 'begin@begin'
        self.state['inform_slots'] = self._init_beliefs()
        self.state['turn'] = 0
        self.state['num_requests'] = {s:0 for s in self.state['database'].slots}
        self.state['slot_tracker'] = set()
        self.state['dont_care'] = set()
        p_db_i = (1./self.state['database'].N)*np.ones((self.state['database'].N,))
        self.state['init_entropy'] = calc_entropies(self.state['inform_slots'], p_db_i, 
                self.state['database'])
        self.state['inputs'] = []
        self.state['actions'] = []
        self.state['rewards'] = []
        self.state['indices'] = []
        self.state['ptargets'] = []
        self.state['phitargets'] = []
        self.state['hid_state'] = [np.zeros((1,self.r_hid)).astype('float32') \
                for s in dialog_config.inform_slots]
        self.state['pol_state'] = np.zeros((1,self.n_hid)).astype('float32')

    ''' get next action based on rules '''
    def next(self, user_action, verbose=False):
        self.state['turn'] += 1

        p_vector = np.zeros((self.in_size,)).astype('float32')
        p_vector[:self.feat_extractor.n] = self.feat_extractor.featurize( \
                user_action['nl_sentence'])
        if self.state['turn']>1:
            pr_act = self.state['prevact'].split('@')
            assert pr_act[0]!='inform', 'Agent called after informing!'
            act_id = dialog_config.inform_slots.index(pr_act[1])
            p_vector[self.feat_extractor.n+act_id] = 1
        p_vector = np.expand_dims(np.expand_dims(p_vector, axis=0), axis=0)
        p_vector = standardize(p_vector)

        p_targets = []
        phi_targets = []
        if self.training and self.num_updates<self.pol_start:
            self._update_state(user_action['nl_sentence'], upd=self.upd, verbose=verbose)
            db_probs = self._check_db()
            H_db = tools.entropy_p(db_probs)
            H_slots = calc_entropies(self.state['inform_slots'], db_probs, self.state['database'])

            # act on policy but train on expert
            pp = np.zeros((len(dialog_config.inform_slots)+1,))
            for i,s in enumerate(dialog_config.inform_slots):
                pp[i] = H_slots[s]
            pp[-1] = H_db
            pp = np.expand_dims(np.expand_dims(pp, axis=0), axis=0)
            _, action = self._rule_act(pp, db_probs)
            act, _, p_out, hid_out, p_db = self._prob_act(p_vector, mode='sample')
            for s in dialog_config.inform_slots:
                p_s = self.state['inform_slots'][s]/self.state['inform_slots'][s].sum()
                p_targets.append(p_s)
                if s in self.state['dont_care']:
                    phi_targets.append(np.ones((1,)).astype('float32'))
                else:
                    phi_targets.append(np.zeros((1,)).astype('float32'))
        else:
            if self.training: act, action, p_out, hid_out, db_probs = self._prob_act(p_vector, mode='sample')
            else: act, action, p_out, hid_out, db_probs = self._prob_act(p_vector, mode='max')

        self._state_update(act, p_vector, action, user_action['reward'], p_out, hid_out, p_targets, \
                phi_targets)

        act['posterior'] = db_probs

        return act

    def _state_update(self, act, p, action, rew, p_out, h_out, p_t, phi_t):
        if act['diaact']=='inform':
            self.state['prevact'] = 'inform@inform'
            self.state['indices'] = np.asarray(act['target'][:dialog_config.SUCCESS_MAX_RANK], \
                    dtype='int32')
        else:
            req = act['request_slots'].keys()[0]
            self.state['prevact'] = 'request@%s' %req
            self.state['num_requests'][req] += 1
        self.state['inputs'].append(p[0,0,:])
        self.state['actions'].append(action)
        self.state['rewards'].append(rew)
        self.state['hid_state'] = h_out
        self.state['pol_state'] = p_out
        self.state['ptargets'].append(p_t)
        self.state['phitargets'].append(phi_t)

    def _prob_act(self, p, mode='sample'):
        act = {}
        act['diaact'] = 'UNK'
        act['request_slots'] = {}
        act['target'] = []

        action, db_sample, db_probs, p_out, h_out, pv, phiv = self.act(p, self.state['pol_state'], \
                self.state['hid_state'], mode=mode)
        if action==self.out_size-1:
            act['diaact'] = 'inform'
            act['target'] = [0]*self.state['database'].N
            act['target'][:dialog_config.SUCCESS_MAX_RANK] = db_sample
            act['target'][dialog_config.SUCCESS_MAX_RANK:] = list(set(range(self.state['database'].N))\
                    - set(db_sample))
        else:
            act['diaact'] = 'request'
            s = dialog_config.inform_slots[action]
            act['request_slots'][s] = 'UNK'
        act['probs'] = pv
        act['phis'] = [phv.flatten() for phv in phiv]
        return act, action, p_out, h_out, db_probs

    def _rule_act(self, p, db_probs):
        act = {}
        act['diaact'] = 'UNK'
        act['request_slots'] = {}
        act['target'] = []

        if p[0,0,-1] < self.tr:
            # agent reasonable confident, inform
            act['diaact'] = 'inform'
            act['target'] = self._inform(db_probs)
            action = len(dialog_config.inform_slots)
        else:
            H_slots = {s:p[0,0,i] for i,s in enumerate(dialog_config.inform_slots)}
            sorted_entropies = sorted(H_slots.items(), key=operator.itemgetter(1), reverse=True)
            req = False
            for (s,h) in sorted_entropies:
                if H_slots[s]<self.frac*self.state['init_entropy'][s] or H_slots[s]<self.ts or \
                        self.state['num_requests'][s] >= self.max_req:
                    continue
                act['diaact'] = 'request'
                act['request_slots'][s] = 'UNK'
                action = dialog_config.inform_slots.index(s)
                req = True
                break
            if not req:
                # agent confident about all slots, inform
                act['diaact'] = 'inform'
                act['target'] = self._inform(db_probs)
                action = len(dialog_config.inform_slots)
        return act, action

    def terminate_episode(self, user_action):
        assert self.state['turn'] <= self.max_turn, "More turn than MAX_TURN!!"
        total_reward = aggregate_rewards(self.state['rewards']+[user_action['reward']],self.discount)
        
        if self.state['turn']==self.max_turn:
            db_index = np.arange(dialog_config.SUCCESS_MAX_RANK).astype('int32')
            db_switch = 0
        else:
            db_index = self.state['indices']
            db_switch = 1

        inp = np.zeros((self.max_turn,self.in_size)).astype('float32')
        actmask = np.zeros((self.max_turn,self.out_size)).astype('int8')
        turnmask = np.zeros((self.max_turn,)).astype('int8')
        p_targets = [np.zeros((self.max_turn,self.slot_sizes[i])).astype('float32') \
                for i in range(len(dialog_config.inform_slots))]
        phi_targets = [np.zeros((self.max_turn,)).astype('float32') \
                for i in range(len(dialog_config.inform_slots))]
        for t in xrange(0,self.state['turn']):
            actmask[t,self.state['actions'][t]] = 1
            inp[t,:] = self.state['inputs'][t]
            turnmask[t] = 1
            if self.training and self.num_updates<self.pol_start:
                for i in range(len(dialog_config.inform_slots)):
                    p_targets[i][t,:] = self.state['ptargets'][t][i]
                    phi_targets[i][t] = self.state['phitargets'][t][i]

        self.add_to_pool(inp, turnmask, actmask, total_reward, db_index, db_switch, p_targets, \
                phi_targets)
        self.recent_rewards.append(total_reward)
        self.recent_turns.append(self.state['turn'])
        if self.state['turn'] == self.max_turn: self.recent_successes.append(0)
        elif user_action['reward']>0: self.recent_successes.append(1)
        else: self.recent_successes.append(-1)

