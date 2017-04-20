'''
Author: @t-bhdhi
Created on August 5, 2016

This file specifies the default parameter settings for all agents, user simulator and the database.
'''

agent_params = {}

agent_params['act-rule'] = {}
agent_params['act-rule']['tr'] = 1.0
agent_params['act-rule']['ts'] = 1.0
agent_params['act-rule']['max_req'] = 1
agent_params['act-rule']['frac'] = 0.5
agent_params['act-rule']['upd'] = 10

agent_params['nl-rule-no'] = {}
agent_params['nl-rule-no']['ts'] = 0.5
agent_params['nl-rule-no']['max_req'] = 2
agent_params['nl-rule-no']['frac'] = 0.5
agent_params['nl-rule-no']['upd'] = 05

agent_params['nl-rule-hard'] = {}
agent_params['nl-rule-hard']['ts'] = 0.5
agent_params['nl-rule-hard']['max_req'] = 2
agent_params['nl-rule-hard']['frac'] = 0.5
agent_params['nl-rule-hard']['upd'] = 05

agent_params['nl-rule-soft'] = {}
agent_params['nl-rule-soft']['tr'] = 5.0
agent_params['nl-rule-soft']['ts'] = 0.5
agent_params['nl-rule-soft']['max_req'] = 1
agent_params['nl-rule-soft']['frac'] = 0.5
agent_params['nl-rule-soft']['upd'] = 10

agent_params['simple-rl-soft'] = {}
agent_params['simple-rl-soft']['tr'] = 5.0
agent_params['simple-rl-soft']['ts'] = 0.5
agent_params['simple-rl-soft']['max_req'] = 1
agent_params['simple-rl-soft']['frac'] = 0.5
agent_params['simple-rl-soft']['upd'] = 10
agent_params['simple-rl-soft']['input'] = 'entropy'
agent_params['simple-rl-soft']['pol_start'] = 0
agent_params['simple-rl-soft']['nhid'] = 50
agent_params['simple-rl-soft']['lr'] = 0.05
agent_params['simple-rl-soft']['batch'] = 128
agent_params['simple-rl-soft']['ment'] = 0.

agent_params['simple-rl-hard'] = {}
agent_params['simple-rl-hard']['ts'] = 0.5
agent_params['simple-rl-hard']['max_req'] = 2
agent_params['simple-rl-hard']['frac'] = 0.5
agent_params['simple-rl-hard']['upd'] = 05
agent_params['simple-rl-hard']['input'] = 'entropy'
agent_params['simple-rl-hard']['pol_start'] = 0
agent_params['simple-rl-hard']['nhid'] = 50
agent_params['simple-rl-hard']['lr'] = 0.05
agent_params['simple-rl-hard']['batch'] = 128
agent_params['simple-rl-hard']['ment'] = 0.

agent_params['simple-rl-no'] = {}
agent_params['simple-rl-no']['ts'] = 0.5
agent_params['simple-rl-no']['max_req'] = 2
agent_params['simple-rl-no']['frac'] = 0.5
agent_params['simple-rl-no']['upd'] = 05
agent_params['simple-rl-no']['input'] = 'entropy'
agent_params['simple-rl-no']['pol_start'] = 0
agent_params['simple-rl-no']['nhid'] = 50
agent_params['simple-rl-no']['lr'] = 0.05
agent_params['simple-rl-no']['batch'] = 128
agent_params['simple-rl-no']['ment'] = 0.

agent_params['e2e-rl-soft'] = {}
agent_params['e2e-rl-soft']['tr'] = 5.0
agent_params['e2e-rl-soft']['ts'] = 0.5
agent_params['e2e-rl-soft']['max_req'] = 1
agent_params['e2e-rl-soft']['frac'] = 0.5
agent_params['e2e-rl-soft']['upd'] = 10
agent_params['e2e-rl-soft']['input'] = 'entropy'
agent_params['e2e-rl-soft']['pol_start'] = 500
agent_params['e2e-rl-soft']['nhid'] = 100
agent_params['e2e-rl-soft']['lr'] = 0.05
agent_params['e2e-rl-soft']['featN'] = 2
agent_params['e2e-rl-soft']['batch'] = 128
agent_params['e2e-rl-soft']['ment'] = 0.
agent_params['e2e-rl-soft']['sl'] = 'e2e'
agent_params['e2e-rl-soft']['rl'] = 'e2e'

dataset_params = {}

dataset_params['imdb-S'] = {}
dataset_params['imdb-S']['dataset'] = 'imdb-S'
dataset_params['imdb-S']['unk'] = 0.20

dataset_params['imdb-M'] = {}
dataset_params['imdb-M']['dataset'] = 'imdb-M'
dataset_params['imdb-M']['unk'] = 0.20

dataset_params['imdb-L'] = {}
dataset_params['imdb-L']['dataset'] = 'imdb-L'
dataset_params['imdb-L']['unk'] = 0.20

dataset_params['imdb-XL'] = {}
dataset_params['imdb-XL']['dataset'] = 'imdb-XL'
dataset_params['imdb-XL']['unk'] = 0.20
