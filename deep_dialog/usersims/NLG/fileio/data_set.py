"""
"""


import argparse, json, random, copy
import cPickle as pickle
import numpy as np


class DataSet:
    def __init__(self, params, act_dict=None, slot_dict=None, word_dict=None, template_word_dict=None):
        self.dia_slot_val = params.get('dia_slot_val', 0) # 1, 2 for bag-of-values
        self.data = read_data(params, act_dict, slot_dict, word_dict, template_word_dict)
        self.slot_rep_bit = 1 if params['slot_rep']==0 else 2
        self.diaact_sentence_pairs = self.get_diaact_sentence_pairs()
        self.split_data(params)
        

    def get_dia_input_size(self):
        total_dim = len(self.data['act_dict'])
        total_dim += len(self.data['slot_dict']) * self.slot_rep_bit
        
        if self.dia_slot_val == 1:
            total_dim += len(self.data['word_dict'])
        elif self.dia_slot_val == 2:
            total_dim += len(self.data['template_word_dict'])
        return total_dim
    
    def get_input_size(self):
        if self.dia_slot_val == 2 or self.dia_slot_val == 3:
            total_dim = len(self.data['template_word_dict'])
        else:
            total_dim = len(self.data['word_dict'])
        return total_dim
    
    def get_output_size(self):
        if self.dia_slot_val == 2 or self.dia_slot_val == 3:
            total_dim = len(self.data['template_word_dict'])
        else:
            total_dim = len(self.data['word_dict'])
        return total_dim

    def get_data_size(self):
        dsize = {'train':0, 'test':0, 'valid':0}
        for s in self.split.keys():
            dsize[s] = len(self.split[s])
        return dsize

    def get_stat_info(self):
        stat_info = {}
        stat_info['conversations'] = len(self.data['dialog_set'])
        return stat_info

    def split_data(self, params):
        """ Split the data into {train, valid, test}: randomly and fixed """
        
        folder = params.get('cv_fold', 6)
        split_method = params.get('split_method', 1)

        one_folder = len(self.diaact_sentence_pairs)/folder
        all_data = self.diaact_sentence_pairs

        self.split = {}
        if split_method == 0:
            self.split['valid'] = random.sample(all_data, one_folder)
            for con in self.split['valid']: all_data.remove(con)

            self.split['test'] = random.sample(all_data, one_folder)
            for con in self.split['test']: all_data.remove(con)

            self.split['train'] = all_data
        elif split_method == 1:
            self.split['train'] = []
            self.split['valid'] = []
            self.split['test'] = []
            for index, pair in enumerate(self.diaact_sentence_pairs):
                if index % folder == 1: self.split['valid'].append(pair)
                elif index % folder == 2: self.split['test'].append(pair)
                else: self.split['train'].append(pair)

    def get_struct_info(self):
        return {'input_size': self.get_input_size(), 'output_size':self.get_output_size(), 'data_size':self.get_data_size(), 'diaact_input_size':self.get_dia_input_size()}
    
    
    def get_diaact_sentence_pairs(self):
        diaact_sentence_pairs = []
        
        for conv_id in self.data['dialog_set'].keys():
            for turn_id in self.data['dialog_set'][conv_id].keys():
                for sen_id in self.data['dialog_set'][conv_id][turn_id].keys():
                    instance = {}
                    instance['sentence'] = self.data['dialog_set'][conv_id][turn_id][sen_id]['sentence']
                    instance['sentence_template'] = self.data['dialog_set'][conv_id][turn_id][sen_id]['sentence_template']
                    instance['diaact_act'] = self.data['dialog_set'][conv_id][turn_id][sen_id]['diaact']
                    instance['diaact_rep'] = self.data['dialog_set'][conv_id][turn_id][sen_id]['diaact_rep']
                    instance['slotval'] = self.data['dialog_set'][conv_id][turn_id][sen_id]['slotval']
                    instance['slotrep'] = self.data['dialog_set'][conv_id][turn_id][sen_id]['slotrep']
                    
                    if self.dia_slot_val == 2 or self.dia_slot_val == 3:
                        instance.update(self.prepare_input_rep_with_slot_placeholder(instance))
                    else: # dia_slot_val == 0 or 1
                        instance.update(self.prepare_input_rep(instance))
                    diaact_sentence_pairs.append(instance)
        return diaact_sentence_pairs
    
    def prepare_input_rep(self, instance):
        sentence_rep = {}
            
        vec = np.zeros((1, self.get_dia_input_size()))
        vec[0][instance['diaact_rep']] = 1
        for v in instance['slotrep']: vec[0][v] = 1
        
        if self.dia_slot_val == 1: # add slot-val into dia_act rep
            start_index = self.get_dia_input_size() - len(self.data['word_dict'])
            for slot in instance['slotval'].keys():
                slot_val = instance['slotval'][slot]
                if slot_val in self.data['word_dict'].keys() and slot_val != 'UNK':
                    slot_val_index = start_index + self.data['word_dict'][slot_val]
                    vec[0][slot_val_index] = 1
                
        word_arr = instance['sentence'].split(' ')
        word_vecs = np.zeros((len(word_arr), len(self.data['word_dict'])))
        labels = [0] * (len(word_arr)-1)
        for w_index, w in enumerate(word_arr[:-1]):
            if w in self.data['word_dict'].keys():
                w_dict_index = self.data['word_dict'][w]
                word_vecs[w_index][w_dict_index] = 1
                
            if word_arr[w_index+1] in self.data['word_dict'].keys():
                labels[w_index] = self.data['word_dict'][word_arr[w_index+1]] 
            
        sentence_rep['diaact'] = vec
        sentence_rep['words'] = word_vecs
        sentence_rep['labels'] = labels
        return sentence_rep
    
    """ replace the slot value with slot placeholder """
    def prepare_input_rep_with_slot_placeholder(self, instance):
        sentence_rep = {}
        suffix = "_PLACEHOLDER"
            
        vec = np.zeros((1, self.get_dia_input_size()))
        vec[0][instance['diaact_rep']] = 1
        for v in instance['slotrep']: vec[0][v] = 1
        
        if self.dia_slot_val == 2: # add slot-val into dia_act rep
            start_index = self.get_dia_input_size() - len(self.data['template_word_dict'])
            for slot in instance['slotval'].keys():
                slot_val = slot+suffix
                if slot_val in self.data['template_word_dict'].keys() and slot_val != 'UNK':
                    slot_val_index = start_index + self.data['template_word_dict'][slot_val]
                    vec[0][slot_val_index] = 1
                
        word_arr = instance['sentence_template'].split(' ')
        word_vecs = np.zeros((len(word_arr), len(self.data['template_word_dict'])))
        labels = [0] * (len(word_arr)-1)
        for w_index, w in enumerate(word_arr[:-1]):
            if w in self.data['template_word_dict'].keys():
                w_dict_index = self.data['template_word_dict'][w]
                word_vecs[w_index][w_dict_index] = 1
                
            if word_arr[w_index+1] in self.data['template_word_dict'].keys():
                labels[w_index] = self.data['template_word_dict'][word_arr[w_index+1]] 
            
        sentence_rep['diaact'] = vec
        sentence_rep['words'] = word_vecs
        sentence_rep['labels'] = labels
        return sentence_rep
    

def read_data(params, actdict=None, slotdict=None, worddict=None, template_worddict=None):
    """ Extract movies, theaters, starttime from .csv files """
    path = params['data_path']
    act_set = params['act_set']
    slot_set = params['slot_set']
    slot_rep_method = params.get('slot_rep', 0)

    file = open(path, 'r')
    lines = [line for line in file]

    act_dict = {}
    slot_dict = {}
    slot_val_dict = {}
    
    word_dict = {}
    template_word_dict = {}
    
    max_len = 0
    
    act_stat = {}
    slot_stat = {}

    pre_convID = ''
    pre_turnID = 0
    pre_sentID = 0

    dialog_set = {}
    suffix = "_PLACEHOLDER"
    
    # load act Dict and slot Dict from files
    if actdict == None:
        if act_set != 'none': act_dict = load_act_set(act_set)
    else:
        act_dict = copy.deepcopy(actdict)
    
    if slotdict == None:   
        if slot_set != 'none': slot_dict = load_slot_set(slot_set)
    else:
        slot_dict = copy.deepcopy(slotdict)
        
    slot_rep_bit = 1 if slot_rep_method==0 else 2

    for lineindex, line in enumerate(lines[1:]):
        arr = line.split(',')

        convID = arr[0]
        turnID = int(arr[1]) - 1
        turn_msg = arr[3]
        sentID = 0
        if pre_convID == convID and pre_turnID == turnID: sentID = pre_sentID + 1

        pre_convID = convID
        pre_turnID = turnID
        pre_sentID = sentID

        annot = arr[4]
        sentence = arr[7]
        act = annot.strip(' ')
        #if act == '' or len(act) == 0: #continue # no action here

        sent_slot_rep = [] # slot-rep

        if annot.find('(') > 0 and annot.find(')') > 0:
            act = annot[0: annot.find('(')].strip(' ').lower() #Dia act
            annot = annot[annot.find('(')+1:-1].strip(' ') #slot-value pairs

            if act in act_stat: act_stat[act] += 1
            else: act_stat[act] = 1
        
        if len(annot) > 0: # slot-pair values: slot[val] = id
            annot_segs = annot.split(';') #slot-value pairs
            sent_slot_vals = {} # slot-pair real value
            sent_rep_vals = {} # slot-pair id value

            for annot_seg in annot_segs:
                annot_seg = annot_seg.strip(' ')
                annot_slot = annot_seg
                if annot_seg.find('=') > 0:
                    annot_slot = annot_seg[:annot_seg.find('=')] #annot_seg.split('=')[0].strip(' ')
                    annot_val = annot_seg[annot_seg.find('=')+1:] #annot_seg.split('=')[1].strip(' ')
                else: #requested
                    annot_val = 'UNK' # for request
                    if annot_slot == 'taskcomplete': annot_val = 'FINISH'

                if annot_slot in slot_stat: slot_stat[annot_slot] += 1
                else: slot_stat[annot_slot] = 1

                if annot_slot == 'mc_list': continue
                    
                # slot may have multiple values
                sent_slot_vals[annot_slot] = []
                sent_rep_vals[annot_slot] = []

                if annot_val.startswith('{') and annot_val.endswith('}'): # multiple-choice or result={}
                    annot_val = annot_val[1:-1].strip(' ')

                    if annot_slot == 'result': # result={slot=value}
                        result_annot_seg_arr = annot_val.strip(' ').split('&')
                        if len(annot_val.strip(' '))> 0:
                            for result_annot_seg_item in result_annot_seg_arr:
                                result_annot_seg_arr = result_annot_seg_item.strip(' ').split('=')

                                result_annot_seg_slot = result_annot_seg_arr[0]
                                result_annot_seg_slot_val = result_annot_seg_arr[1]
                                if result_annot_seg_slot in slot_val_dict.keys():
                                    if result_annot_seg_slot_val not in slot_val_dict[result_annot_seg_slot].keys():
                                        slot_val_dict[result_annot_seg_slot][result_annot_seg_slot_val] = len(slot_val_dict[result_annot_seg_slot])
                                else:
                                    slot_val_dict[result_annot_seg_slot] = {}
                                    slot_val_dict[result_annot_seg_slot][result_annot_seg_slot_val] = len(slot_val_dict[result_annot_seg_slot])

                                sent_slot_vals[annot_slot].append({result_annot_seg_slot:result_annot_seg_slot_val})
                                sent_rep_vals[annot_slot].append({result_annot_seg_slot:slot_val_dict[result_annot_seg_slot][result_annot_seg_slot_val]})
                        else: # result={}
                            pass
                    else: # multi-choice or mc_list
                        annot_val_arr = annot_val.split('#')
                        for annot_val_item in annot_val_arr:
                            if annot_slot in slot_val_dict.keys():
                                if annot_val_item not in slot_val_dict[annot_slot].keys():
                                    slot_val_dict[annot_slot][annot_val_item] = len(slot_val_dict[annot_slot])
                            else:
                                slot_val_dict[annot_slot] = {}
                                slot_val_dict[annot_slot][annot_val_item] = len(slot_val_dict[annot_slot])

                            sent_slot_vals[annot_slot].append(annot_val_item)
                            sent_rep_vals[annot_slot].append(slot_val_dict[annot_slot][annot_val_item])
                else: # single choice
                    if annot_slot in slot_val_dict.keys():
                        if annot_val not in slot_val_dict[annot_slot].keys():
                            slot_val_dict[annot_slot][annot_val] = len(slot_val_dict[annot_slot])
                    else:
                        slot_val_dict[annot_slot] = {}
                        slot_val_dict[annot_slot][annot_val] = len(slot_val_dict[annot_slot])

                    sent_slot_vals[annot_slot].append(annot_val)
                    sent_rep_vals[annot_slot].append(slot_val_dict[annot_slot][annot_val])

                if slot_dict.has_key(annot_slot): # 'slotrep'
                    annot_slot_id = slot_dict[annot_slot]
                    slot_rep_id = len(act_dict) + slot_rep_bit*annot_slot_id # 2*annot_slot_id
                    if annot_val == 'UNK': slot_rep_id += (0 if slot_rep_method==0 else 1)
                    sent_slot_rep.append(slot_rep_id)

                if slot_set == 'none': # load slot set from labeled data: slot_dict[slot] = id
                    if annot_slot not in slot_dict.keys(): slot_dict[annot_slot] = len(slot_dict)

            if convID in dialog_set.keys():
                if turnID not in dialog_set[convID].keys():
                    dialog_set[convID][turnID] = {}
            else:
                dialog_set[convID] = {}
                dialog_set[convID][turnID] = {}

            dialog_set[convID][turnID][sentID] = {}
            dialog_set[convID][turnID][sentID]['slotval'] = sent_slot_vals
            dialog_set[convID][turnID][sentID]['slotval_rep'] = sent_rep_vals

        else: # no slot-value pairs
            if convID in dialog_set.keys():
                if turnID not in dialog_set[convID].keys():
                    dialog_set[convID][turnID] = {}
            else:
                dialog_set[convID] = {}
                dialog_set[convID][turnID] = {}

            dialog_set[convID][turnID][sentID] = {}
            dialog_set[convID][turnID][sentID]['slotval'] = {} # no slot
            dialog_set[convID][turnID][sentID]['slotval_rep'] = {}

        dialog_set[convID][turnID][sentID]['slotrep'] = sent_slot_rep

        if act_set == 'none': # load act set from labeled data: act_dict[act] = id
            if act not in act_dict.keys(): act_dict[act] = len(act_dict)

        dialog_set[convID][turnID][sentID]['diaact'] = act
        dialog_set[convID][turnID][sentID]['diaact_rep'] = -1 # no action
        if act in act_dict.keys(): dialog_set[convID][turnID][sentID]['diaact_rep'] = act_dict[act]
        dialog_set[convID][turnID][sentID]['turn_msg'] = turn_msg
        
        new_sentence = 's_o_s ' + sentence + ' e_o_s'
        new_sentence = new_sentence.lower()
        dialog_set[convID][turnID][sentID]['sentence'] = new_sentence
        
        # 07/08/2016 replace the slot value with slot placeholder; may have some error here
        #template_sentence = new_sentence
        #for slot in dialog_set[convID][turnID][sentID]['slotval'].keys():
        #    slot_vals = dialog_set[convID][turnID][sentID]['slotval'][slot]
        #    slot_placeholder = slot + suffix
        #    if slot == 'result' or slot == 'numberofpeople': continue
        #    for slot_val in slot_vals:
        #        tmp_sentence = template_sentence.replace(slot_val.lower(), slot_placeholder)
        #        template_sentence = tmp_sentence
        #if 'numberofpeople' in dialog_set[convID][turnID][sentID]['slotval'].keys():
        #    slot_vals = dialog_set[convID][turnID][sentID]['slotval']['numberofpeople']
        #    slot_placeholder = 'numberofpeople' + suffix
        #    for slot_val in slot_vals:
        #        tmp_sentence = template_sentence.replace(slot_val.lower(), slot_placeholder)
        #        template_sentence = tmp_sentence
        
        template_sentence = process_slot_template(new_sentence, dialog_set[convID][turnID][sentID]['slotval'])
        dialog_set[convID][turnID][sentID]['sentence_template'] = template_sentence
        
        if template_worddict == None:
            template_word_arr = template_sentence.split(' ')
            for t_w in template_word_arr:
                if t_w not in template_word_dict.keys():
                    template_word_dict[t_w] = len(template_word_dict)
        # end of replacement for slot_vals
        
        if worddict == None:
            word_arr = new_sentence.split(' ')
            for w in word_arr:
                if w not in word_dict.keys():
                    word_dict[w] = len(word_dict)
    
    if worddict != None:
        word_dict = copy.deepcopy(worddict)
    
    if template_worddict == None:    
        for slot in slot_dict.keys():
            slot_placeholder = slot + suffix
            if slot_placeholder not in template_word_dict.keys():
                template_word_dict[slot_placeholder] = len(template_word_dict)
    else:
        template_word_dict = copy.deepcopy(template_worddict)
    
    print 'Some Statistics Info:'
    print 'Dialog Act Dict Size:', len(act_dict)
    print 'Dialog Act Set', json.dumps(act_dict, indent=2)
    print 'Dialog Act Set Stat', json.dumps(act_stat, indent=2)
    print 'Slot Dict Size:', len(slot_dict)
    print 'Slot Set', json.dumps(slot_dict, indent=2)

    print 'Slot Value Size:', len(slot_val_dict)
    print 'Real Word Dict Size:', len(word_dict)
    print 'Template Word Dict Size:', len(template_word_dict)
    for slot in slot_val_dict.keys():
        if 'UNK' not in slot_val_dict[slot].keys(): slot_val_dict[slot]['UNK'] = len(slot_val_dict[slot])
        #if 'FINISH' not in slot_val_dict[slot].keys(): slot_val_dict[slot]['FINISH'] = len(slot_val_dict[slot])
        print slot, len(slot_val_dict[slot]), slot_stat[slot]
    
    data = {'act_dict': act_dict, 'slot_dict': slot_dict, 'slot_val_dict':slot_val_dict, 'dialog_set':dialog_set, 'word_dict': word_dict, 'template_word_dict':template_word_dict}
    return data


""" pre-process all the slot_val with slot_placeholder """
def process_slot_template(real_sentence, slot_val_dict):
    template_sentence = real_sentence
    suffix = "_PLACEHOLDER"
    
    for slot in slot_val_dict.keys():
        slot_vals = slot_val_dict[slot]
        slot_placeholder = slot + suffix
        if slot == 'result' or slot == 'numberofpeople': continue
        for slot_val in slot_vals:
            tmp_sentence = template_sentence.replace(slot_val.lower(), slot_placeholder)
            template_sentence = tmp_sentence
    
    if 'numberofpeople' in slot_val_dict.keys():
        slot_vals = slot_val_dict['numberofpeople']
        slot_placeholder = 'numberofpeople' + suffix
        for slot_val in slot_vals:
            tmp_sentence = template_sentence.replace(slot_val.lower(), slot_placeholder)
            template_sentence = tmp_sentence
    
    return template_sentence
        

def load_act_set(path):
    """ Load the dia act set """
    act_set = {}

    file = open(path, 'r')
    index = 0
    for line in file:
        act_set[line.strip('\n').strip('\r')] = index
        index += 1
    return act_set

def load_slot_set(path):
    """ Load the slot tag set """
    slot_set = {}

    file = open(path, 'r')
    index = 0
    for line in file:
        slot_set[line.strip('\n').strip('\r')] = index
        index += 1
    return slot_set


def load_json(path):
    """ test function for load .json file  """
    
    #kb = json.load(open(path, 'rb'))
    kb = pickle.load(open(path, 'rb'))
    
    for k in kb: print k, kb[k]
    
    print 'size', len(kb)
    


def main(params):
    dset = DataSet(params)
    print dset.get_struct_info()
    

#lstm_tanh_relu_[1468149176.91]_2_0.597.p: 45.4/34.6/28.5/24.5
#lstm_tanh_[1468195823.36]_120_0.603.p: 47.5/35.7/29.2/24.6
# 49.0/39.2/33.6/29.7


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-dpath', '--data_path', dest='data_path', type=str, default='../data/movieMultiLine.Annot.Corrected.Final.v3.csv', help='path to data file')
    parser.add_argument('--act_set', dest='act_set', type=str, default='../data/dia_acts.txt', help='path to dia act set; none for loading from labelled file')
    parser.add_argument('--slot_set', dest='slot_set', type=str, default='../data/slot_set.txt', help='path to slot set; none for loading from labelled file')
    parser.add_argument('--slot_rep', dest='slot_rep', type=int, default=1, help='0 for 1 bit; 1 for 2 bits')
    parser.add_argument('--dia_slot_val', dest='dia_slot_val', type=int, default=2, help='0 for no slot value; 1 for with slot value')
    
    args = parser.parse_args()
    params = vars(args)
    print json.dumps(params, indent=2)

    main(params)
