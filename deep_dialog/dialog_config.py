'''
'''

all_acts = ['request', 'inform']
inform_slots = ['actor','critic_rating','genre','mpaa_rating','director','release_year']

sys_request_slots = ['actor', 'critic_rating', 'genre', 'mpaa_rating', 'director', 'release_year']

start_dia_acts = {
    #'greeting':[],
    'request':['moviename', 'starttime', 'theater', 'city', 'state', 'date', 'genre', 'ticket', 'numberofpeople', 'numberofkids']
}   

#reward information
FAILED_DIALOG_REWARD = -1
SUCCESS_DIALOG_REWARD = 2
PER_TURN_REWARD = -0.1
SUCCESS_MAX_RANK = 5
MAX_TURN = 10

MODEL_PATH = './data/pretrained/'
