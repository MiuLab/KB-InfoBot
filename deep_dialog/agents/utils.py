from collections import Counter
from deep_dialog import tools
import numpy as np
import time

def standardize(arr):
    return arr

def calc_entropies(state, q, db):
    entropies = {}
    for s,c in state.iteritems():
        if s not in db.slots:
            entropies[s] = 0.
        else:
            p = (db.ids[s]*q).sum(axis=1)
            u = db.priors[s]*q[db.unks[s]].sum()
            c_tilde = p+u
            c_tilde = c_tilde/c_tilde.sum()
            entropies[s] = tools.entropy_p(c_tilde)
    return entropies
