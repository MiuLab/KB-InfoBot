import argparse, json, shutil, io, time
import numpy as np

from collections import Counter

agent_map = {'rule-no'  :   'nl-rule-no',
        'rl-no'         :   'simple-rl-no',
        'rule-hard'     :   'nl-rule-hard',
        'rl-hard'       :   'simple-rl-hard',
        'rule-soft'     :   'nl-rule-soft',
        'rl-soft'       :   'simple-rl-soft',
        'e2e-soft'      :   'e2e-rl-soft',
        }

parser = argparse.ArgumentParser()

parser.add_argument('--agent', dest='agent_type', type=str, default='rule-soft', 
        help='agent to use (rule-no / rl-no / rule-hard / rl-hard / rule-soft / rl-soft / e2e-soft)')
parser.add_argument('--N', dest='N', type=int, default=5000, help='Number of simulations')
parser.add_argument('--db', dest='db', type=str, default='imdb-M', 
        help='imdb-(S/M/L/XL) -- This is the KB split to use, e.g. imdb-M')
parser.add_argument('--max_turn', dest='max_turn', default=20, type=int, 
        help='maximum length of each dialog (default=20, 0=no maximum length)')
parser.add_argument('--err_prob', dest='err_prob', default=0.5, type=float, 
        help='the probability of the user simulator corrupting a slot value')
parser.add_argument('--dontknow_prob', dest='dontknow_prob', default=0.5, type=float, 
        help='the probability that user simulator does not know a slot value')
parser.add_argument('--sub_prob', dest='sub_prob', default=0.05, type=float, 
        help='the probability that user simulator substitutes a slot value')
parser.add_argument('--nlg_temp', dest='nlg_temp', type=float, default=1., 
        help='Natural Language Generator softmax temperature (to control noise)')
parser.add_argument('--max_first_turn', dest='max_first_turn', type=int, default=5, 
        help='Maximum number of slots informed by user in first turn')
parser.add_argument('--model_name', dest='model_name', type=str, default='pretrained', 
        help='model name to evaluate (This should be the same as what you gave for training). Pass "pretrained" to use pretrained models.')

args = parser.parse_args()
params = vars(args)

params['act_set'] = './data/dia_acts.txt'
params['template_path'] = './data/templates.p'
params['nlg_slots_path'] = './data/nlg_slot_set.txt'
params['nlg_model_path'] = './data/pretrained/lstm_tanh_[1470015675.73]_115_120_0.657.p'

shutil.copyfile('settings/config_'+params['db']+'.py', 'config.py')
from config import *
for k,v in dataset_params[params['db']].iteritems():
    params[k] = v
for k,v in agent_params[agent_map[params['agent_type']]].iteritems():
    params[k] = v

print 'Dialog Parameters: '
print json.dumps(params, indent=2)


max_turn = params['max_turn']
err_prob = params['err_prob']
dk_prob = params['dontknow_prob']
template_path = params['template_path']
agent_type = agent_map[params['agent_type']]
N = params['N']
save_path = None

datadir = './data/' + params['dataset']
db_full_path = datadir + '/db.txt'
db_inc_path = datadir + '/incomplete_db_%.2f.txt' %params['unk']
dict_path = datadir + '/dicts.json'
slot_path = datadir + '/slot_set.txt'
corpus_path = './data/corpora/' + params['dataset'] + '_corpus.txt'

from deep_dialog.dialog_system import DialogManager, MovieDict, DictReader, Database
from deep_dialog.agents import AgentNLRuleSoft, AgentNLRuleHard, AgentNLRuleNoDB
from deep_dialog.agents import AgentSimpleRLAllAct, AgentSimpleRLAllActHardDB
from deep_dialog.agents import AgentSimpleRLAllActNoDB, AgentE2ERLAllAct
from deep_dialog.usersims import RuleSimulator, TemplateNLG, S2SNLG
from deep_dialog.objects import SlotReader

act_set = DictReader()
act_set.load_dict_from_file(params['act_set'])

slot_set = SlotReader(slot_path)

movie_kb = MovieDict(dict_path)

db_full = Database(db_full_path, movie_kb, name=params['dataset'])
db_inc = Database(db_inc_path, movie_kb, name='incomplete%.2f_'%params['unk']+params['dataset'])

nlg = S2SNLG(template_path, params['nlg_slots_path'], params['nlg_model_path'], params['nlg_temp'])
user_sim = RuleSimulator(movie_kb, act_set, slot_set, None, max_turn, nlg, err_prob, db_full, \
        1.-dk_prob, sub_prob=params['sub_prob'], max_first_turn=params['max_first_turn'])

if params['model_name']=='pretrained':
    params['model_name'] = 'best_'+agent_type+'_imdb.m'
if agent_type == 'act-rule':
    agent = AgentActRule(movie_kb, act_set, slot_set, db_inc,
            upd=params['upd'], tr=params['tr'], ts=params['ts'], 
            frac=params['frac'], max_req=params['max_req'])
elif agent_type == 'simple-rl-soft':
    agent = AgentSimpleRLAllAct(movie_kb, act_set, slot_set, db_inc, train=False, _reload=True,
            n_hid=params['nhid'], batch=params['batch'], ment=params['ment'],
            inputtype=params['input'],
            pol_start=params['pol_start'], lr=params['lr'], upd=params['upd'], tr=params['tr'],
            ts=params['ts'], frac=params['frac'], max_req=params['max_req'], 
            name=params['model_name'])
elif agent_type == 'simple-rl-hard':
    agent = AgentSimpleRLAllActHardDB(movie_kb, act_set, slot_set, db_inc, train=False, 
            _reload=True,
            n_hid=params['nhid'], batch=params['batch'], ment=params['ment'],
            inputtype=params['input'],
            pol_start=params['pol_start'], lr=params['lr'], upd=params['upd'],
            ts=params['ts'], frac=params['frac'], max_req=params['max_req'], 
            name=params['model_name'])
elif agent_type == 'simple-rl-no':
    agent = AgentSimpleRLAllActNoDB(movie_kb, act_set, slot_set, db_inc, train=False, 
            _reload=True,
            n_hid=params['nhid'], batch=params['batch'], ment=params['ment'],
            inputtype=params['input'],
            pol_start=params['pol_start'], lr=params['lr'], upd=params['upd'],
            ts=params['ts'], frac=params['frac'], max_req=params['max_req'], 
            name=params['model_name'])
elif agent_type == 'e2e-rl-soft':
    agent = AgentE2ERLAllAct(movie_kb, act_set, slot_set, db_inc, corpus_path, train=False,
            _reload=True, pol_start=params['pol_start'], sl=params['sl'], rl=params['rl'],
            n_hid=params['nhid'], batch=params['batch'], ment=params['ment'], lr=params['lr'],
            N=params['featN'],
            inputtype=params['input'], tr=params['tr'], ts=params['ts'], frac=params['frac'],
            max_req=params['max_req'], upd=params['upd'], name=params['model_name'])
elif agent_type=='nl-rule-hard':
    agent = AgentNLRuleHard(movie_kb, act_set, slot_set, db_inc, corpus_path,
            ts=params['ts'], frac=params['frac'],
            max_req=params['max_req'], upd=params['upd'])
elif agent_type=='nl-rule-soft':
    agent = AgentNLRuleSoft(movie_kb, act_set, slot_set, db_inc, corpus_path,
            tr=params['tr'], ts=params['ts'], frac=params['frac'],
            max_req=params['max_req'], upd=params['upd'])
else:
    agent = AgentNLRuleNoDB(movie_kb, act_set, slot_set, db_inc, corpus_path,
            ts=params['ts'], frac=params['frac'],
            max_req=params['max_req'], upd=params['upd'])

dialog_manager = DialogManager(agent, user_sim, db_full, db_inc, movie_kb, verbose=False)

all_rewards = np.zeros((N,))
all_success = np.zeros((N,))
all_turns = np.zeros((N,))
if save_path is not None: fs = io.open(save_path, 'w')
tst = time.time()

for i in range(N):
    current_reward = 0
    current_success = False
    ua = dialog_manager.initialize_episode()
    utt = ua['nl_sentence']
    if save_path is not None: fs.write(utt+'\n')
    t = 0
    while(True):
        t += 1
        episode_over, reward, ua, sa = dialog_manager.next_turn()
        utt = ua['nl_sentence']
        if save_path is not None: fs.write(utt+'\n')
        current_reward += reward
        if episode_over:
            if reward > 0:
                print ("Successful Dialog! Total reward = {}".format(current_reward))
                current_success = True
            else:
                print ("Failed Dialog! Total reward = {}".format(current_reward))
            break
    all_rewards[i] = current_reward
    all_success[i] = 1 if current_success else 0
    all_turns[i] = t
if save_path is not None: fs.close()
time_elapsed = time.time()-tst
nn = np.sqrt(N)
print("Overall: {} times, (mean/std) {} / {} reward, {} / {} success rate, {} / {} turns, {} time elapsed".format(N,
        np.mean(all_rewards), np.std(all_rewards)/nn, np.mean(all_success), 
        np.std(all_success)/nn,
        np.mean(all_turns), np.std(all_turns)/nn, time_elapsed))
