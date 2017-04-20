'''
'''

from collections import Counter
import math
import numpy as np
import sys
import string
import nltk
from nltk.corpus import stopwords

EXC = set(string.punctuation)

def to_tokens(text):
    utt = ''.join(ch for ch in text if ch not in EXC)
    tokens = nltk.word_tokenize(utt.lower())
    return [w for w in tokens if w not in stopwords.words('english')]

def entropy(items):
    if type(items) is Counter or type(items) is dict:
        P = items
    elif type(items) is list:
        P = Counter(items)
    if not P:
        # empty distribution
        return -1
    H = 0.
    N = 0.
    for v in P.values():
        if v==0:
            continue
        H -= v*math.log(v,2)
        N += v
    if N==0:
        return -1
    H = (H/N) + math.log(N,2)
    if math.isnan(H):
        print '\n'.join(['%s:%.7f' %(k,v) for k,v in items.iteritems()])
        sys.exit()
    return H

def entropy_p(p):
    return np.sum(-p*np.nan_to_num(np.log2(p)))

def categorical_sample(probs):
    x = np.random.uniform()
    s = probs[0]
    i = 0
    while s<x:
        i += 1
        s += probs[i]
    return i
