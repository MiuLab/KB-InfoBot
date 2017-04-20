'''

A script
1. load the trained model
2. generated result files for bleu score evaluation

Command: python predict.py --model_path .\checkpoints\rnn\lstm_tanh_relu_[1466371634.05]_2_0.504.p --predict_mode 0
Evaluate: perl .\eval\multi-bleu.perl .\results\real.txt < .\results\pred.txt

'''


import argparse, json, copy
import cPickle as pickle
import numpy as np

from fileio import DataSet
from decoders.decoder import decoder

from decoders.lstm_decoder_tanh import lstm_decoder_tanh

""" post_process to fill the slot value in the generated template sentence """
def post_process(pred_template, slot_val_dict, slot_dict):
    sentence = pred_template
    suffix = "_PLACEHOLDER"
        
    for slot in slot_val_dict.keys():
        slot_vals = slot_val_dict[slot]
        slot_placeholder = slot + suffix
        if slot == 'result' or slot == 'numberofpeople': continue
        for slot_val in slot_vals:
            try:
                tmp_sentence = sentence.replace(slot_placeholder, slot_val, 1) # add 1 is better
            except UnicodeDecodeError:
                tmp_sentence = sentence.decode('utf8').replace(slot_placeholder, \
                        slot_val, 1) # add 1 is better
            sentence = tmp_sentence
                
    if 'numberofpeople' in slot_val_dict.keys():
        slot_vals = slot_val_dict['numberofpeople']
        slot_placeholder = 'numberofpeople' + suffix
        for slot_val in slot_vals:
            tmp_sentence = sentence.replace(slot_placeholder, slot_val, 1)
            sentence = tmp_sentence
    
    for slot in slot_dict.keys():
        slot_placeholder = slot + suffix
        tmp_sentence = sentence.replace(slot_placeholder, '')
        sentence = tmp_sentence  
    return sentence

    

""" Function 1: Evaluate on the .csv file and write the results to txt files, then compute the Bleu scores """
""" Predict on the dset[split] """
def evaluate(word_dict, template_word_dict, slot_dict, dset, split, decoder, params):
    results = []
    if params['dia_slot_val'] == 2 or params['dia_slot_val'] == 3:
        inverse_word_dict = {template_word_dict[k]:k for k in template_word_dict.keys()}
    else:
        inverse_word_dict = {word_dict[k]:k for k in word_dict.keys()}
    
    for i, ele in enumerate(dset.split[split]):
        #pred_ys, pred_words = decoder.forward(inverse_word_dict, ele, params, predict_model=True)
        pred_ys, pred_words = decoder.beam_forward(inverse_word_dict, ele, params, predict_model=True)
        
        pred_template_sentence = ' '.join(pred_words[:-1])
        real_sentence = ' '.join(ele['sentence'].split(' ')[1:-1])
        real_template_sentence = ' '.join(ele['sentence_template'].split(' ')[1:-1])
           
        # replace the slot_val place holder with real value
        sentence = post_process(pred_template_sentence, ele['slotval'], slot_dict)
                
        #results.append({'real':real_template_sentence, 'pred':pred_template_sentence}) 
        results.append({'real':real_sentence, 'pred':sentence}) 
        
        print 'test case', i, "/", len(dset.split[split])
        print 'teamplate:', real_template_sentence
        print 'pred_template:', pred_template_sentence
        print 'real:', real_sentence
        print 'pred_sentence:', sentence
        print ''
    
    return results

""" Predict on the dset """
def evaluate_all(word_dict, template_word_dict, slot_dict, dset, decoder, params):
    results = []
    if params['dia_slot_val'] == 2 or params['dia_slot_val'] == 3:
        inverse_word_dict = {template_word_dict[k]:k for k in template_word_dict.keys()}
    else:
        inverse_word_dict = {word_dict[k]:k for k in word_dict.keys()}
    
    for i, ele in enumerate(dset.diaact_sentence_pairs):
        #pred_ys, pred_words = decoder.forward(inverse_word_dict, ele, params, predict_model=True)
        pred_ys, pred_words = decoder.beam_forward(inverse_word_dict, ele, params, predict_model=True)
        
        pred_template_sentence = ' '.join(pred_words[:-1])
        real_sentence = ' '.join(ele['sentence'].split(' ')[1:-1])
        real_template_sentence = ' '.join(ele['sentence_template'].split(' ')[1:-1])
           
        # replace the slot_val place holder with real value
        sentence = post_process(pred_template_sentence, ele['slotval'], slot_dict)
                
        #results.append({'real':real_template_sentence, 'pred':pred_template_sentence}) 
        results.append({'real':real_sentence, 'pred':sentence}) 
        
        print 'test case', i, "/", len(dset.diaact_sentence_pairs)
        print 'teamplate:', real_template_sentence
        print 'pred_template:', pred_template_sentence
        print 'real:', real_sentence
        print 'pred_sentence:', sentence
        print ''
    
    return results

""" Save the results to txt files (real and predict) """
def save_results(results):
    with open("./results/real.txt", "w") as text_file:
        for res in results:
            text_file.write(res['real']+'\n')
    
    with open("./results/pred.txt", "w") as text_file:
        for res in results:
            text_file.write(res['pred']+'\n')



""" Function 2: Read the Dia-act file, and output the NL. """
""" read the Dia-act from file """
def read_diaact_from_file(path, word_dict, template_word_dict, act_dict, slot_dict, decoder, params):
    file = open(path, 'r')
    lines = [line for line in file]
    
    if params['dia_slot_val'] == 2 or params['dia_slot_val'] == 3:
        inverse_word_dict = {template_word_dict[k]:k for k in template_word_dict.keys()}
    else:
        inverse_word_dict = {word_dict[k]:k for k in word_dict.keys()}
     
    for lineindex, l in enumerate(lines):
        l = l.strip('\n')
        dia_act = parse_str_to_diaact(l, act_dict, slot_dict)
        dia_act_rep = prepare_diaact_representation(dia_act, word_dict, template_word_dict, act_dict, slot_dict, params)
        
        #pred_ys, pred_words = decoder.forward(inverse_word_dict, dia_act_rep, params, predict_model=True)
        pred_ys, pred_words = decoder.beam_forward(inverse_word_dict, dia_act_rep, params, predict_model=True)
        pred_sentence = ' '.join(pred_words[:-1])
            
        # replace the slot_val place holder with real value
        sentence = post_process(pred_sentence, dia_act['inform_slots'], slot_dict)
            
        print 'Test case', lineindex
        print 'Dia-Act:', l
        print 'Pred_Template:', pred_sentence 
        print 'NLG:', sentence
              
""" parse the string into dia-act """
def parse_str_to_diaact(string, act_dict, slot_dict):
    annot = string.strip(' ').strip('\n').strip('\r')
    act = annot

    if annot.find('(') > 0 and annot.find(')') > 0:
        act = annot[0: annot.find('(')].strip(' ').lower() #Dia act
        annot = annot[annot.find('(')+1:-1].strip(' ') #slot-value pairs
    else: annot = ''

    act_slot_value = {}
    act_slot_value['diaact'] = 'UNK'
    act_slot_value['inform_slots'] = {}
    act_slot_value['request_slots'] = {}

    if act in act_dict.keys(): # dialog_config.all_acts
        act_slot_value['diaact'] = act

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

            if annot_slot == 'mc_list': continue

            # slot may have multiple values
            sent_slot_vals[annot_slot] = []
            sent_rep_vals[annot_slot] = []

            if annot_val.startswith('{') and annot_val.endswith('}'):
                annot_val = annot_val[1:-1]

                if annot_slot == 'result':
                    result_annot_seg_arr = annot_val.strip(' ').split('&')
                    if len(annot_val.strip(' '))> 0:
                        for result_annot_seg_item in result_annot_seg_arr:
                            result_annot_seg_arr = result_annot_seg_item.strip(' ').split('=')
                            result_annot_seg_slot = result_annot_seg_arr[0]
                            result_annot_seg_slot_val = result_annot_seg_arr[1]

                            if result_annot_seg_slot_val == 'UNK': act_slot_value['request_slots'][result_annot_seg_slot] = 'UNK'
                            else: act_slot_value['inform_slots'][result_annot_seg_slot] = [result_annot_seg_slot_val]
                    else: # result={}
                        pass
                else: # multi-choice or mc_list
                    annot_val_arr = annot_val.split('#')
                    act_slot_value['inform_slots'][annot_slot] = []
                    for annot_val_ele in annot_val_arr:
                        act_slot_value['inform_slots'][annot_slot].append(annot_val_ele)
            else: # single choice
                if annot_slot in slot_dict.keys():
                    if annot_val == 'UNK':
                        act_slot_value['request_slots'][annot_slot] = 'UNK'
                    else:
                        act_slot_value['inform_slots'][annot_slot] = [annot_val]
        
    return act_slot_value

""" prepare the diaact into vector representation """
def prepare_diaact_representation(dia_act, word_dict, template_word_dict, act_dict, slot_dict, params):
    act_rep = np.zeros((1, len(act_dict)))
    act_rep[0, act_dict[dia_act['diaact']]] = 1.0
    
    slot_rep_bit = 1 if params['slot_rep']==0 else 2
    slot_rep = np.zeros((1, len(slot_dict)*slot_rep_bit)) 
    
    if params['dia_slot_val'] == 2 or params['dia_slot_val'] == 3:
        word_rep = np.zeros((1, len(template_word_dict)))
        words = np.zeros((1, len(template_word_dict)))
        words[0, template_word_dict['s_o_s']] = 1.0
    else: 
        word_rep = np.zeros((1, len(word_dict)))
        words = np.zeros((1, len(word_dict)))
        words[0, word_dict['s_o_s']] = 1.0
        
    for slot in dia_act['inform_slots'].keys():
        slot_index = slot_dict[slot]
        slot_rep[0, slot_index*slot_rep_bit] = 1.0
        
        for slot_val in dia_act['inform_slots'][slot]:
            if params['dia_slot_val'] == 2:
                slot_placeholder = slot + "_PLACEHOLDER"
                if slot_placeholder in template_word_dict.keys():
                    word_rep[0, template_word_dict[slot_placeholder]] = 1.0
            elif params['dia_slot_val'] == 1:
                if slot_val in word_dict.keys():
                    word_rep[0, word_dict[slot_val]] = 1.0
    
    for slot in dia_act['request_slots'].keys():
        slot_index = slot_dict[slot]
        slot_rep[0, slot_index*slot_rep_bit + 1] = 1.0
    
    if params['dia_slot_val'] == 0 or params['dia_slot_val'] == 3:
        final_representation = np.hstack([act_rep, slot_rep])
    else: # dia_slot_val == 1 or 2
        final_representation = np.hstack([act_rep, slot_rep, word_rep])
    
    dia_act_rep = {}
    dia_act_rep['diaact'] = final_representation
    dia_act_rep['words'] = words
    return dia_act_rep



""" Function 3: Command Line input """
def read_diaact_from_cmd(word_dict, template_word_dict, act_dict, slot_dict, decoder, params):
    if params['dia_slot_val'] == 2 or params['dia_slot_val'] == 3:
        inverse_word_dict = {template_word_dict[k]:k for k in template_word_dict.keys()}
    else:
        inverse_word_dict = {word_dict[k]:k for k in word_dict.keys()}
     
    while True:
        print("Your input: ")
        command = raw_input()
        annot = command.strip(' ').strip('\n').strip('\r')
        if len(annot) > 0: 
            dia_act = parse_str_to_diaact(annot, act_dict, slot_dict)
            dia_act_rep = prepare_diaact_representation(dia_act, word_dict, template_word_dict, act_dict, slot_dict, params)
            
            #pred_ys, pred_words = decoder.forward(inverse_word_dict, dia_act_rep, params, predict_model=True)
            pred_ys, pred_words = decoder.beam_forward(inverse_word_dict, dia_act_rep, params, predict_model=True)
            pred_sentence = ' '.join(pred_words[:-1])
            
            # replace the slot_val place holder with real value
            sentence = post_process(pred_sentence, dia_act['inform_slots'], slot_dict)
                    
            print 'Dia-Act:', annot
            print 'Dia-Act Parsing:', dia_act
            print 'Pred_Template:', pred_sentence 
            print 'NLG:', sentence
        else:
            break

""" Function 4: String input """
def read_diaact_from_string(text, word_dict, template_word_dict, act_dict, slot_dict, decoder, params):
    if params['dia_slot_val'] == 2 or params['dia_slot_val'] == 3:
        inverse_word_dict = {template_word_dict[k]:k for k in template_word_dict.keys()}
    else:
        inverse_word_dict = {word_dict[k]:k for k in word_dict.keys()}
     
    annot = text.strip(' ').strip('\n').strip('\r')
    dia_act = parse_str_to_diaact(annot, act_dict, slot_dict)
    dia_act_rep = prepare_diaact_representation(dia_act, word_dict, template_word_dict, act_dict, slot_dict, params)
    
    pred_ys, pred_words = decoder.beam_forward(inverse_word_dict, dia_act_rep, params, predict_model=True)
    pred_sentence = ' '.join(pred_words[:-1])
    
    # replace the slot_val place holder with real value
    sentence = post_process(pred_sentence, dia_act['inform_slots'], slot_dict)

    return sentence
                

""" Save the Dia-Act to a file """
def save_diaact_to_file(dset, split, act_dict):
    dia_acts = []
    inverse_act_dict = {act_dict[k]:k for k in act_dict.keys()}
    for i, ele in enumerate(dset.split[split]):
        if ele['diaact_rep'] not in inverse_act_dict.keys(): continue
        act = inverse_act_dict[ele['diaact_rep']]
        slots = ""
        for slot in ele['slotval'].keys():
            if len(ele['slotval'][slot]) == 1:
                if ele['slotval'][slot][0] == "UNK" or slot == 'taskcomplete':
                    slots = slot + slots
                else:
                    slots += ";" + slot + "=" + ele['slotval'][slot][0]
        
        if slots.startswith(";"):
            slots = slots[1:]
        
        diaact = act + "(" + slots + ")"
        dia_acts.append(diaact)
        
    with open("./data/test_diaacts.txt", "w") as text_file:
        for diaact in dia_acts:
            text_file.write(diaact+'\n')

def generate(model_params, rnnmodel, act_string):
    text = read_diaact_from_string(act_string, model_params['word_dict'], model_params['template_word_dict'], model_params['act_dict'], model_params['slot_dict'], rnnmodel, model_params['params'])

    return text

def main(params):
    model_path = params['model_path']
    model_params = pickle.load(open(model_path, 'rb'))
    
    model_params['params']['data_path'] = params['data_path']
    dset = DataSet(model_params['params'], model_params['act_dict'], model_params['slot_dict'], model_params['word_dict'], model_params['template_word_dict'])
    
    hidden_size = model_params['model']['Wd'].shape[0]
    output_size = model_params['model']['Wd'].shape[1]
    
    if model_params['params']['model'] == 'lstm_tanh': # lstm_tanh
        diaact_input_size = model_params['model']['Wah'].shape[0]
        input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
        rnnmodel = lstm_decoder_tanh(diaact_input_size, input_size, hidden_size, output_size)
     
    rnnmodel.model = copy.deepcopy(model_params['model'])
    predict_mode = params['predict_mode']
    
    model_params['params']['beam_size'] =  params['beam_size']
    model_params['params']['decoder_sampling'] = params['decoder_sampling']
    
    if predict_mode == 0:
        #results = evaluate(model_params['word_dict'], model_params['template_word_dict'], model_params['slot_dict'], dset, 'test', rnnmodel, model_params['params'])
        results = evaluate_all(model_params['word_dict'], model_params['template_word_dict'],  model_params['slot_dict'], dset, rnnmodel, model_params['params'])
        save_results(results)
    elif predict_mode == 1:
        read_diaact_from_file(params['diaact_file_path'], model_params['word_dict'], model_params['template_word_dict'], model_params['act_dict'], model_params['slot_dict'], rnnmodel, model_params['params'])
    elif predict_mode == 2:
        read_diaact_from_cmd(model_params['word_dict'], model_params['template_word_dict'], model_params['act_dict'], model_params['slot_dict'], rnnmodel, model_params['params'])
    
    #save_diaact_to_file(dset, 'test', model_params['act_dict'])
    
    # read the .csv file and save the results to txt files, to compute Bleu scores
    #results = evaluate(model_params['word_dict'], dset, 'test', rnnmodel, model_params['params'])
    #results = evaluate_all(model_params['word_dict'], dset, rnnmodel, model_params['params'])
    #save_results(results)
    
    # read the dia-act from file
    #read_diaact_from_file(params['diaact_file_path'], model_params['word_dict'], model_params['act_dict'], model_params['slot_dict'], rnnmodel, model_params['params'])

    # command line input
    #read_diaact_from_cmd(model_params['word_dict'], model_params['act_dict'], model_params['slot_dict'], rnnmodel, model_params['params'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--diaact_file_path', dest='diaact_file_path', type=str, default='data/test_diaacts_2.txt', help='path to dia-act data file')
    
    parser.add_argument('-dpath', '--data_path', dest='data_path', type=str, default='data/movieMultiLine.Annot.Corrected.Final.v3.csv', help='path to data file')
    parser.add_argument('--act_set', dest='act_set', type=str, default='data/dia_acts.txt', help='path to dia act set; none for loading from labelled file')
    parser.add_argument('--slot_set', dest='slot_set', type=str, default='data/slot_set.txt', help='path to slot set; none for loading from labelled file')
    parser.add_argument('--slot_rep', dest='slot_rep', type=int, default=1, help='0 for 1 bit; 1 for 2 bits')
    
    parser.add_argument('--decoder_sampling', dest='decoder_sampling', type=int, default=0, help='0 for argmax; 1 for sampling')
    parser.add_argument('--beam_size', dest='beam_size', type=int, default=10, help='beam search size')
    
    parser.add_argument('--model', dest='model', type=str, default='rnn', help='rnn, lstm')
    parser.add_argument('--model_path', dest='model_path', type=str, default='checkpoints/rnn/lstm_relu_[1465946901.91]_0.266.p', help='path to model file')
    
    parser.add_argument('--predict_mode', dest='predict_mode', type=int, default=0, help='0 for test bleu score; 1 for test a file; 2 for command line input')
    
    args = parser.parse_args()
    params = vars(args)
    print json.dumps(params, indent=2)

    main(params)
