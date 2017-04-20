'''

Takes user action and produces NL utterance using Xiujun's NLG
Backs off to template nlg
'''

import cPickle as pkl
import random
import copy
import sys

from NLG import predict
from NLG.decoders.lstm_decoder_tanh import lstm_decoder_tanh
from NLG.decoders.decoder import decoder

BEAM_SIZE = 3
SAMPLING = 1

class S2SNLG:
    def __init__(self, template_file, slot_file, model_file, temp):
        self.templates = pkl.load(open(template_file, 'rb'))
        self._read_slots(slot_file)
        self._load_model(model_file, temp)

    def _load_model(self, model_path, temp):
        model_params = pkl.load(open(model_path, 'rb'))
        hidden_size = model_params['model']['Wd'].shape[0]
        output_size = model_params['model']['Wd'].shape[1]

        model_params['params']['beam_size'] =  BEAM_SIZE
        model_params['params']['decoder_sampling'] = SAMPLING
        model_params['params']['temp'] = temp
        
        if model_params['params']['model'] == 'lstm_tanh': # lstm_tanh
            diaact_input_size = model_params['model']['Wah'].shape[0]
            input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
            self.rnnmodel = lstm_decoder_tanh(diaact_input_size, input_size, hidden_size, output_size)
        self.rnnmodel.model = copy.deepcopy(model_params['model'])
        self.model_params = model_params

    def _read_slots(self, fil):
        f = open(fil,'r')
        self.slots = []
        for line in f:
            self.slots.append(line.rstrip())

    def generate(self, act, request_slots, inform_slots):
        if all([r in self.slots for r in request_slots.keys()]) and \
                all([i in self.slots for i in inform_slots.keys()]):
            return self.generate_from_nlg(act, request_slots, inform_slots)
        else:
            return self.generate_from_template(act, request_slots, inform_slots)

    def generate_from_nlg(self, act, request_slots, inform_slots):
        act_string = act + '('
        for s,v in request_slots.iteritems():
            act_string += '%s=%s;' % (s,v) if v!='UNK' else '%s;' %s
        i_slots = {k:v for k,v in inform_slots.iteritems() if v is not None}
        for s,v in i_slots.iteritems():
            act_string += '%s=%s;' % (s,v)
        act_string = act_string.rstrip(';')
        act_string += ')'
        sent = predict.generate(self.model_params, self.rnnmodel, act_string)
        try:
            out = unicode(sent)
        except UnicodeDecodeError:
            out = unicode(sent.decode('utf8'))
        return out

    def generate_from_template(self, act, request_slots, inform_slots):
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

if __name__=='__main__':
    temp_file = '../data/templates.p'
    slot_file = 'NLG/data/slot_set.txt'
    model_file = 'NLG/checkpoints/nlg_infobot/lstm_tanh_[1470015675.73]_115_120_0.657.p'

    acts = ['inform', 'request']
    slots = ['actor', 'director', 'release_year', 'genre', 'mpaa_rating', 'critic_rating']

    nlg = S2SNLG(temp_file, slot_file, model_file, 2.0)

    for i in range(10000):
        a = random.choice(acts)
        if a=='inform':
            i_slots = [random.choice(slots)]
            inform_slots = {}
            for s in i_slots:
                inform_slots[s] = u'blah'
            request_slots = {}
            print a, inform_slots, request_slots
            print nlg.generate(a, inform_slots, request_slots)
        else:
            request_slots = {}
            request_slots['moviename'] = 'UNK'
            inform_slots = {}
            i_slots = random.sample(slots, 2)
            for s in i_slots:
                inform_slots[s] = u'blah'
            print a, inform_slots, request_slots
            print nlg.generate(a, inform_slots, request_slots)
