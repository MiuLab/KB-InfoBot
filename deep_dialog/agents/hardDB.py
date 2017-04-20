'''
'''

import random
import operator
import numpy as np

class HardDB:
    ''' get dist over DB based on current beliefs '''
    def _check_db_soft(self):
        # induce disttribution over DB based on current beliefs over slots
        probs = {}
        p_s = np.zeros((self.state['database'].N, \
                len(self.state['database'].slots))).astype('float32')
        for i,s in enumerate(self.state['database'].slots):
            p = self.state['inform_slots'][s]/self.state['inform_slots'][s].sum()
            n = self.state['database'].inv_counts[s]
            p_unk = float(n[-1])/self.state['database'].N
            p_tilde = p*(1.-p_unk)
            p_tilde = np.concatenate([p_tilde,np.asarray([p_unk])])
            p_s[:,i] = p_tilde[self.state['database'].table[:,i]]/ \
                    n[self.state['database'].table[:,i]]
        p_db = np.sum(np.log(p_s), axis=1)
        p_db = np.exp(p_db - np.min(p_db))
        p_db = p_db/p_db.sum()
        return p_db

    def _inform(self, db_index):
        probs = self._check_db_soft()
        return np.argsort(probs)[::-1].tolist()

    ''' query DB based on current known slots '''
    def _check_db(self):
        # from query to db form current inform_slots
        db_query = []
        for s in self.state['database'].slots:
            if s in self.state['slot_tracker'] and s in self.state['inform_slots']:
                max_i = np.argmax(self.state['inform_slots'][s])
                max_key = self.movie_dict.dict[s][max_i]
                db_query.append(max_key)
            else:
                db_query.append(None)
        matches, index = self.state['database'].lookup(db_query)
        return matches, index

