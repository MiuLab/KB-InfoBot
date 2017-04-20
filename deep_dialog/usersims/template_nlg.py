'''

Takes user action and templates from file and produces NL utterance. 
'''

import cPickle as pkl
import random

class TemplateNLG:
    def __init__(self, template_file):
        self.templates = pkl.load(open(template_file, 'rb'))

    def generate(self, act, request_slots, inform_slots):
        n_r = len(request_slots.keys())
        i_slots = {k:v for k,v in inform_slots.iteritems() if v is not None}
        n_i = len(i_slots.keys())
        key = '%s_%d_%d' % (act, n_r, n_i)

        temp = random.choice(self.templates[key])
        sent = self._fill_slots(temp, request_slots, i_slots)

        return unicode(sent)

    def _fill_slots(self, temp, request_slots, i_slots):
        reqs = request_slots.keys()
        infs = i_slots.keys()
        random.shuffle(reqs)
        random.shuffle(infs)

        for i,k in enumerate(reqs):
            temp = temp.replace('@rslot%d'%i, k)

        for i,k in enumerate(infs):
            temp = temp.replace('@islot%d'%i, k)
            temp = temp.replace('@ival%d'%i, i_slots[k])

        return temp
