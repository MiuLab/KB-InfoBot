'''
'''

from deep_dialog import dialog_config, tools
from agent import Agent
from hardDB import HardDB
from belief_tracker import BeliefTracker
from utils import *

from collections import Counter, defaultdict

import operator
import random
import math
import numpy as np
import cPickle as pkl
import copy
import re
import nltk

class AgentNLRuleHard(Agent,HardDB,BeliefTracker):

    def initialize_episode(self):
        self.state = {}
        self.state['database'] = pkl.loads(pkl.dumps(self.database,-1))
        for slot in self.state['database'].slots:
            if slot not in dialog_config.inform_slots: self.state['database'].delete_slot(slot)
        self.state['prevact'] = 'begin@begin'
        self.state['inform_slots'] = self._init_beliefs()
        self.state['turn'] = 0
        p_db_i = np.ones((self.state['database'].N,))/self.state['database'].N
        self.state['init_entropy'] = calc_entropies(self.state['inform_slots'], p_db_i,
                self.state['database'])
        self.state['num_requests'] = {s:0 for s in self.state['inform_slots'].keys()}
        self.state['slot_tracker'] = set()
        self.state['dont_care'] = set()

    ''' get next action based on rules '''
    def next(self, user_action, verbose=False):
        self._update_state(user_action['nl_sentence'], upd=self.upd, verbose=verbose)
        self.state['turn'] += 1

        act = {}
        act['diaact'] = 'UNK'
        act['request_slots'] = {}
        act['target'] = []

        db_status, db_index = self._check_db()
        H_slots = {}
        for s in dialog_config.inform_slots:
            s_p = self.state['inform_slots'][s]/self.state['inform_slots'][s].sum()
            H_slots[s] = tools.entropy_p(s_p)
        sorted_entropies = sorted(H_slots.items(), key=operator.itemgetter(1), reverse=True)
        if verbose:
            print 'Agent slot belief entropies - '
            print ' '.join(['%s:%.2f' %(k,v) for k,v in H_slots.iteritems()])

        if not db_status:
            # no match, some error, re-ask some slot
            act['diaact'] = 'request'
            request_slot = random.choice(self.state['inform_slots'].keys())
            act['request_slots'][request_slot] = 'UNK'
            self.state['prevact'] = 'request@%s' %request_slot
            self.state['num_requests'][request_slot] += 1
        elif len(db_status)==1:
            act['diaact'] = 'inform'
            act['target'] = self._inform(db_index)
            self.state['prevact'] = 'inform@inform'
        else:
            req = False
            for (s,h) in sorted_entropies:
                if H_slots[s]<self.frac*self.state['init_entropy'][s] or H_slots[s]<self.ts or \
                        self.state['num_requests'][s] >= self.max_req:
                    continue
                act['diaact'] = 'request'
                act['request_slots'][s] = 'UNK'
                self.state['prevact'] = 'request@%s' %s
                self.state['num_requests'][s] += 1
                req = True
                break
            if not req:
                # agent confident about all slots, inform
                act['diaact'] = 'inform'
                act['target'] = self._inform(db_index)
                self.state['prevact'] = 'inform@inform'

        act['posterior'] = np.zeros((len(self.database.labels),))
        act['posterior'][db_index] = 1./len(db_index)

        return act

    def terminate_episode(self, user_action):
        return
