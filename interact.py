'''
'''

WELCOME="""\nHello and welcome to the InfoBot! Please take a moment to read the instructions below on how to interact with the system.

BACKGROUND: The InfoBot helps users search a database for entities (in our case movies) based on its attributes (in our case any of actor, director, release-year, critic-rating, mpaa-rating). 

INSTRUCTIONS: In each interaction a movie will be selected at random from the database and presented to you, along with some of its attributes. To simulate a real-world scenario where the user may not know all attribute values perfectly, multiple noisy values may be presented (separated by ','). For example, the dialog may start as follows:

    turn  0
    agent action:  Hi! I am Info-Bot. I can help you search for movies if you tell me their attributes!
    target  movie :  louis c.k.: shameless
    known slots:  release_year={ 2007 } critic_rating={ 6.5 , 8.7 } actor={} director={} mpaa_rating={ tv-ma } genre={}
    your input: 

Only some of the slot values will be provided. In the above example critic-rating may be either 6.5 or 8.7, please select one value when informing the agent. At this stage you must initiate the dialog by asking the InfoBot for a movie which matches some of the provided attributes. You may specify all attributes in one go or only a subset of them. Please frame your inputs in natural language and try to provide a diverse vareity of inputs. For example:

    your input: which movie has critic rating 6.5?

In each subsequent turn, the agent will either request for an attribute or inform results from the database. In case of the latter the dialog will end. A typical turn may look like:

    turn  1
    agent action:  request actor
    target  movie :  louis c.k.: shameless
    known slots:  release_year={ 2007 } critic_rating={ 6.5 , 8.7 } actor={} director={} mpaa_rating={ tv-ma } genre={}
    your input:

Here the agent is requesting for the actor of the movie. Since the actor is not in one of the known slots, you may respond by saying:

    your input: i dont know

This is just an example, you may respond anyway you like. Be creative! 

At the end of the dialog, the agent will inform the top 5 matches from the database, which will be checked if they contain the correct movie:

    agent results:  night catches us, spider man 3, precious, she's out of my league, pineapple
    target movie rank =  169
    failed dialog
    number of turns =  4

This is it. After this a new dialog will be initiated. 

Type 'quit' to end the current dialog (it will be considered a failure). Press Ctrl-C at any time to exit the application."""

import argparse, json, shutil, sys, os, random, copy
import numpy as np
import cPickle as pkl
import datetime
import importlib

agent_map = {'rule-no'  :   'nl-rule-no',
        'rl-no'         :   'simple-rl-no',
        'rule-hard'     :   'nl-rule-hard',
        'rl-hard'       :   'simple-rl-hard',
        'rule-soft'     :   'nl-rule-soft',
        'rl-soft'       :   'simple-rl-soft',
        'e2e-soft'      :   'e2e-rl-soft',
        }

sys.setrecursionlimit(10000)

""" Conduct dialogs between InfoBot agents and real users
"""

parser = argparse.ArgumentParser()

parser.add_argument('--agent', dest='agent', type=str, default='rl-soft', 
        help='Agent to run -- (rule-no / rl-no / rule-hard / rl-hard / rule-soft / rl-soft / e2e-soft')

args = parser.parse_args()
params = vars(args)

params['N'] = 1000
params['db'] = 'imdb-M'
params['act_set'] = './data/dia_acts.txt'
params['max_turn'] = 20
params['err_prob'] = 0.5
params['dontknow_prob'] = 0.5
params['sub_prob'] = 0.05
params['max_first_turn'] = 5
config = importlib.import_module('settings.config_'+params['db'])
agent_params = config.agent_params
dataset_params = config.dataset_params
for k,v in dataset_params[params['db']].iteritems():
    params[k] = v

max_turn = params['max_turn']
err_prob = params['err_prob']
dk_prob = params['dontknow_prob']
N = params['N']

datadir = './data/' + params['dataset']
db_full_path = datadir + '/db.txt'
db_inc_path = datadir + '/incomplete_db_%.2f.txt' %params['unk']
dict_path = datadir + '/dicts.json'
slot_path = datadir + '/slot_set.txt'
corpus_path = './data/corpora/' + params['dataset'] + '_corpus.txt'

from deep_dialog.dialog_system import DialogManager, MovieDict, DictReader, Database
from deep_dialog.agents import AgentActRule, AgentNLRuleSoft, AgentNLRuleHard, AgentNLRuleNoDB, AgentSimpleRLAllAct, AgentSimpleRLAllActHardDB, AgentSimpleRLAllActNoDB, AgentE2ERLAllAct
from deep_dialog.usersims import CmdUser
from deep_dialog.objects import SlotReader

act_set = DictReader()
act_set.load_dict_from_file(params['act_set'])

slot_set = SlotReader(slot_path)

movie_kb = MovieDict(dict_path)

db_full = Database(db_full_path, movie_kb, name=params['dataset'])
db_inc = Database(db_inc_path, movie_kb, name='incomplete%.2f_'%params['unk']+params['dataset'])

user_sim = CmdUser(movie_kb, act_set, slot_set, None, max_turn, err_prob, db_full, \
        dk_prob, sub_prob=params['sub_prob'], max_first_turn=params['max_first_turn'],
        fdict_path = 'data/'+params['db']+'/fdict_2.p')

# load all agents
print WELCOME
print "Loading agents... This may take a few minutes"
agent_type = agent_map[params['agent']]
for k,v in agent_params[agent_type].iteritems():
    params[k] = v
params['model_name'] = 'best_'+agent_type+'_imdb.m'

if agent_type == 'simple-rl-soft':
    agent = AgentSimpleRLAllAct(movie_kb, act_set, slot_set, db_inc, train=False, _reload=True,
            n_hid=params['nhid'], batch=params['batch'], ment=params['ment'],
            inputtype=params['input'], 
            pol_start=params['pol_start'], lr=params['lr'], upd=params['upd'],
            tr=params['tr'], ts=params['ts'], frac=params['frac'], max_req=params['max_req'],
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
            _reload=True, n_hid=params['nhid'], batch=params['batch'], ment=params['ment'],
            lr=params['lr'], N=params['featN'],
            inputtype=params['input'], sl=params['sl'], rl=params['rl'],
            pol_start=params['pol_start'], tr=params['tr'], ts=params['ts'], frac=params['frac'],
            max_req=params['max_req'], upd=params['upd'], name=params['model_name'])
else:
    print "Invalid Agent"
    sys.exit()

uname = raw_input("Please Enter User Name: ").lower()
uid = hash(uname)

cdir = "sessions/"+str(uid)+'_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+"/"
if not os.path.exists(cdir): os.makedirs(cdir)
f = open(os.path.join(cdir,'credentials'), 'w')
f.write(uname)
f.close()
try:
    for i in range(N):
        print "--------------------------------------------------------------------------------"
        print "Dialog %d" %i
        dia = []
        curr_agent = agent
        dia.append(curr_agent)
        dialog_manager = DialogManager(curr_agent, user_sim, db_full, db_inc, movie_kb, verbose=False)
        utt = dialog_manager.initialize_episode()
        dia.append(copy.deepcopy(utt))
        total_reward = 0
        while(True):
            episode_over, reward, utt, agact = dialog_manager.next_turn()
            dia.append(agact)
            dia.append(copy.deepcopy(utt))
            total_reward += reward
            if episode_over:
                break
        pkl.dump(dia, open(cdir+str(i)+".p",'w'))
except KeyboardInterrupt:
    sys.exit()
