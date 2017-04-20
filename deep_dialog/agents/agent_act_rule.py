'''
'''

from deep_dialog import dialog_config, tools
from agent import Agent

import operator
import random

class AgentActRule(Agent):

    def initialize_episode(self):
        self.state = {}
        self.state['diaact'] = 'UNK'
        self.state['inform_slots'] = {}
        self.state['turn'] = 0

    ''' update agent state '''
    def _update_state(self, user_action):
        for s in user_action['inform_slots'].keys():
            self.state['inform_slots'][s] = user_action['inform_slots'][s]
    
    ''' get next action based on rules '''
    def next(self, user_action, verbose=False):
        self._update_state(user_action)

        act = {}
        act['diaact'] = 'UNK'
        act['request_slots'] = {}
        act['target'] = []

        db_status, db_index = self._check_db()

        if not db_status:
            # no match, some error, re-ask some slot
            act['diaact'] = 'request'
            request_slot = random.choice(self.state['inform_slots'].keys())
            act['request_slots'][request_slot] = 'UNK'

        elif len(self.state['inform_slots']) == len(dialog_config.sys_request_slots) \
                or len(db_status)==1:
            act['diaact'] = 'inform'
            act['target'] = self._inform(db_index)

        else:
            # request a slot not known with max entropy
            known_slots = self.state['inform_slots'].keys()
            unknown_slots = [s for s in dialog_config.sys_request_slots if s not in known_slots]
            slot_entropy = {}
            for s in unknown_slots:
                db_idx = self.database.slots.index(s)
                db_matches = [m[db_idx] for m in db_status]
                slot_entropy[s] = tools.entropy(db_matches)
            request_slot, max_ent = max(slot_entropy.iteritems(), key=operator.itemgetter(1))
            if max_ent > 0.:
                act['diaact'] = 'request'
                act['request_slots'][request_slot] = 'UNK'
            else:
                act['diaact'] = 'inform'
                act['target'] = self._inform(db_index)

        act['posterior'] = np.zeros((len(self.database.labels),))
        act['posterior'][db_index] = 1./len(db_index)

        return act

    def terminate_episode(self, user_action):
        return

    def _inform(self, db_index):
        target = db_index
        if len(target) > 1: random.shuffle(target)
        full_range = range(self.database.N)
        random.shuffle(full_range)
        for i in full_range:
            if i not in db_index: target.append(i)
        return target

    ''' query DB based on current known slots '''
    def _check_db(self):
        # from query to db form current inform_slots
        db_query = []
        for s in self.database.slots:
            if s in self.state['inform_slots']:
                db_query.append(self.state['inform_slots'][s])
            else:
                db_query.append(None)
        matches, index = self.database.lookup(db_query)
        return matches, index

    ''' sample value from current state of database '''
    def _sample_slot(self, slot, matches):
        if not matches:
            return None
        index = self.database.slots.index(slot)
        return random.choice([m[index] for m in matches])
