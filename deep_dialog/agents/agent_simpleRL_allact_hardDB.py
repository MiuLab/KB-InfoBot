'''
'''

import numpy as np
import cPickle as pkl

from deep_dialog import dialog_config, tools
from collections import Counter, defaultdict, deque
from agent_rl import RLAgent, aggregate_rewards
from belief_tracker import BeliefTracker
from hardDB import HardDB
from utils import *

import operator
import random
import math
import copy
import re
import nltk

# params
DISPF = 1
SAVEF = 100
ANNEAL = 800

class AgentSimpleRLAllActHardDB(RLAgent,HardDB,BeliefTracker):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, db=None, 
            train=True, _reload=False, n_hid=100, batch=128, ment=0.,
            inputtype='full', pol_start=0, upd=10, tr=2.0, ts=0.5, 
            max_req=2, frac=0.5, lr=0.005, name=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.database = db
        self.max_turn = dialog_config.MAX_TURN
        self.training = train
        self.inputtype = inputtype
        self.pol_start = pol_start
        self.upd = upd
        if inputtype=='entropy':
            #in_size = 3*len(dialog_config.inform_slots)+1
            in_size = 3*len(dialog_config.inform_slots)+6 # 6 bins for number of retrieved results
        else:
            in_size = sum([len(self.movie_dict.dict[s])+2 for s in dialog_config.inform_slots]) + \
                    self.database.N
        out_size = len(dialog_config.inform_slots)+1
        self._init_model(in_size, out_size, n_hid=n_hid, learning_rate_sl=lr, batch_size=batch, \
                ment=ment)
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
        self.tr = tr
        self.ts = ts
        self.frac = frac
        self.max_req = max_req

    def _dict2vec(self, p_dict):
        p_vec = []
        for s in dialog_config.inform_slots:
            s_np = p_dict[s]/p_dict[s].sum()
            if s in self.state['dont_care']:
                np.append(s_np,1.)
            else: 
                np.append(s_np,0.)
            p_vec.append(s_np)
        return np.concatenate(p_vec).astype('float32')

    def _print_progress(self,loss):
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
        print 'Update %d. Avg turns = %.2f . Avg Reward = %.2f . Success Rate = %.2f . Fail Rate = %.2f . Incomplete Rate = %.2f . Loss = %.3f' % \
                (self.num_updates, avg_turn, avg_ret, \
                float(n_suc)/tot, float(n_fail)/tot, float(n_inc)/tot, avg_loss)

    def initialize_episode(self):
        self.episode_count += 1
        if self.training and self.episode_count%self.batch_size==0:
            self.num_updates += 1
            if self.num_updates>self.pol_start and self.num_updates%ANNEAL==0: self.anneal_lr()
            if self.num_updates < self.pol_start: loss = self.update(regime='SL')
            else: loss = self.update(regime='RL')
            if self.num_updates%DISPF==0: self._print_progress(loss)
            if self.num_updates%SAVEF==0: self.save_model(dialog_config.MODEL_PATH+self._name)

        self.state = {}
        self.state['database'] = pkl.loads(pkl.dumps(self.database,-1))
        self.state['prevact'] = 'begin@begin'
        self.state['inform_slots'] = self._init_beliefs()
        self.state['turn'] = 0
        self.state['num_requests'] = {s:0 for s in self.state['inform_slots'].keys()}
        self.state['slot_tracker'] = set()
        self.state['dont_care'] = set()
        p_db_i = (1./self.state['database'].N)*np.ones((self.state['database'].N,))
        self.state['init_entropy'] = calc_entropies(self.state['inform_slots'], p_db_i, 
                self.state['database'])
        self.state['inputs'] = []
        self.state['actions'] = []
        self.state['rewards'] = []
        self.state['pol_state'] = np.zeros((1,self.n_hid)).astype('float32')

    ''' get next action based on rules '''
    def next(self, user_action, verbose=False):
        self._update_state(user_action['nl_sentence'], upd=self.upd, verbose=verbose)
        self.state['turn'] += 1

        db_status, db_index = self._check_db()
        N_db = len(db_index)
        H_slots = {}
        for s in dialog_config.inform_slots:
            s_p = self.state['inform_slots'][s]/self.state['inform_slots'][s].sum()
            H_slots[s] = tools.entropy_p(s_p)
        p_vector = np.zeros((self.in_size,)).astype('float32')
        if self.inputtype=='entropy':
            for i,s in enumerate(dialog_config.inform_slots):
                if s in H_slots: p_vector[i] = H_slots[s]
                p_vector[i+len(dialog_config.inform_slots)] = 1. if s in self.state['dont_care'] \
                        else 0.
            if self.state['turn']>1:
                pr_act = self.state['prevact'].split('@')
                act_id = dialog_config.inform_slots.index(pr_act[1])
                p_vector[2*len(dialog_config.inform_slots)+act_id] = 1.
            #p_vector[-1] = N_db/self.state['database'].N
            if N_db<=5: p_vector[N_db-6] = 1.
            else: p_vector[-1] = 1.
        else:
            p_slots = self._dict2vec(self.state['inform_slots'])
            p_vector[:p_slots.shape[0]] = p_slots
            if self.state['turn']>1:
                pr_act = self.state['prevact'].split('@')
                act_id = dialog_config.inform_slots.index(pr_act[1])
                p_vector[p_slots.shape[0]+act_id] = 1.
            db_i_vector = np.zeros((self.database.N,)).astype('float32')
            db_i_vector[db_index] = 1.
            p_vector[-self.database.N:] = db_i_vector
        p_vector = np.expand_dims(np.expand_dims(p_vector, axis=0), axis=0)
        p_vector = standardize(p_vector)

        if self.training and self.num_updates<self.pol_start:
            # act on policy but train on expert
            pp = np.zeros((len(dialog_config.inform_slots)+1,))
            for i,s in enumerate(dialog_config.inform_slots):
                pp[i] = H_slots[s]
            pp[-1] = N_db
            _, action = self._rule_act(pp, db_index)
            act, _, p_out = self._prob_act(p_vector, db_index, mode='sample')
        else:
            if self.training: act, action, p_out = self._prob_act(p_vector, db_index, mode='sample')
            else: act, action, p_out = self._prob_act(p_vector, db_index, mode='max')

        self.state['inputs'].append(p_vector[0,0,:])
        self.state['actions'].append(action)
        self.state['rewards'].append(user_action['reward'])
        self.state['pol_state'] = p_out

        act['posterior'] = np.zeros((len(self.database.labels),))
        if len(db_index)>0:
            act['posterior'][db_index] = 1./len(db_index)
        else:
            act['posterior'] = 1./len(self.database.labels)

        return act

    def _prob_act(self, p, db_index, mode='sample'):
        act = {}
        act['diaact'] = 'UNK'
        act['request_slots'] = {}
        act['target'] = []

        action, probs, p_out = self.act(p, self.state['pol_state'], mode=mode)
        if action==self.out_size-1:
            act['diaact'] = 'inform'
            act['target'] = self._inform(db_index)
            self.state['prevact'] = 'inform@inform'
        else:
            act['diaact'] = 'request'
            s = dialog_config.inform_slots[action]
            act['request_slots'][s] = 'UNK'
            self.state['prevact'] = 'request@%s' %s
            self.state['num_requests'][s] += 1
        return act, action, p_out

    def _rule_act(self, p, db_index):
        act = {}
        act['diaact'] = 'UNK'
        act['request_slots'] = {}
        act['target'] = []

        if p[-1] == 0:
            # no match, some error, re-ask some slot
            act['diaact'] = 'request'
            request_slot = random.choice(self.state['inform_slots'].keys())
            act['request_slots'][request_slot] = 'UNK'
            self.state['prevact'] = 'request@%s' %request_slot
            self.state['num_requests'][request_slot] += 1
            action = dialog_config.inform_slots.index(request_slot)
        elif p[-1] == 1:
            # agent reasonable confident, inform
            act['diaact'] = 'inform'
            act['target'] = self._inform(db_index)
            action = len(dialog_config.inform_slots)
            self.state['prevact'] = 'inform@inform'
        else:
            H_slots = {s:p[i] for i,s in enumerate(dialog_config.inform_slots)}
            sorted_entropies = sorted(H_slots.items(), key=operator.itemgetter(1), reverse=True)
            req = False
            for (s,h) in sorted_entropies:
                if H_slots[s]<self.frac*self.state['init_entropy'][s] or H_slots[s]<self.ts or \
                        self.state['num_requests'][s] >= self.max_req:
                    continue
                act['diaact'] = 'request'
                act['request_slots'][s] = 'UNK'
                self.state['prevact'] = 'request@%s' %s
                self.state['num_requests'][s] += 1
                action = dialog_config.inform_slots.index(s)
                req = True
                break
            if not req:
                # agent confident about all slots, inform
                act['diaact'] = 'inform'
                act['target'] = self._inform(db_index)
                self.state['prevact'] = 'inform@inform'
                action = len(dialog_config.inform_slots)
        return act, action

    def terminate_episode(self, user_action):
        assert self.state['turn'] <= self.max_turn, "More turn than MAX_TURN!!"
        total_reward = aggregate_rewards(self.state['rewards']+[user_action['reward']],self.discount)
        inp = np.zeros((self.max_turn,self.in_size)).astype('float32')
        actmask = np.zeros((self.max_turn,self.out_size)).astype('int32')
        turnmask = np.zeros((self.max_turn,)).astype('int32')
        for t in xrange(0,self.state['turn']):
            actmask[t,self.state['actions'][t]] = 1
            inp[t,:] = self.state['inputs'][t]
            turnmask[t] = 1
        self.add_to_pool(inp, turnmask, actmask, total_reward)
        self.recent_rewards.append(total_reward)
        self.recent_turns.append(self.state['turn'])
        if self.state['turn'] == self.max_turn: self.recent_successes.append(0)
        elif user_action['reward']>0: self.recent_successes.append(1)
        else: self.recent_successes.append(-1)

