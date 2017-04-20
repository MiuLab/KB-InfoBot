'''
'''

from deep_dialog import dialog_config, tools
from agent import Agent
from softDB import SoftDB
from belief_tracker import BeliefTracker
from utils import *

from collections import Counter, defaultdict

import operator
import random
import math
import numpy as np
import copy
import re
import nltk
import cPickle as pkl

class AgentNLRuleNoDB(Agent,SoftDB,BeliefTracker):

    def initialize_episode(self):
        self.state = {}
        self.state['database'] = pkl.loads(pkl.dumps(self.database,-1))
        self.state['prevact'] = 'begin@begin'
        self.state['inform_slots'] = self._init_beliefs()
        self.state['turn'] = 0
        self.state['init_entropy'] = {}
        for s in dialog_config.inform_slots:
            s_p = self.state['inform_slots'][s]/self.state['inform_slots'][s].sum()
            self.state['init_entropy'][s] = tools.entropy_p(s_p)
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

        db_probs = self._check_db()
        H_slots = {}
        for s in dialog_config.inform_slots:
            s_p = self.state['inform_slots'][s]/self.state['inform_slots'][s].sum()
            H_slots[s] = tools.entropy_p(s_p)
        if verbose:
            print 'Agent slot belief entropies - '
            print ' '.join(['%s:%.2f' %(k,v) for k,v in H_slots.iteritems()])

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
            req = True
            break
        if not req:
            # agent confident about all slots, inform
            act['diaact'] = 'inform'
            act['target'] = self._inform(db_probs)
            self.state['prevact'] = 'inform@inform'

        act['probs'] = [np.concatenate([self.state['inform_slots'][s]/ \
                self.state['inform_slots'][s].sum(), \
                np.asarray([float(self.state['database'].inv_counts[s][-1])/ \
                self.state['database'].N])]) \
                for s in dialog_config.inform_slots]
        act['phis'] = [1. if s in self.state['dont_care'] else 0. for s in dialog_config.inform_slots]
        act['posterior'] = db_probs
        return act

    def terminate_episode(self, user_action):
        return
