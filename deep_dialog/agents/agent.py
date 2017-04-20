'''
'''

from deep_dialog import tools
from feature_extractor import FeatureExtractor

class Agent:
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, db=None, corpus=None, \
            tr=None, ts=None, frac=None, max_req=None, upd=None):
        self.movie_dict = movie_dict
        self.act_det = act_set
        self.slot_set = slot_set
        self.database = db
        self.tr = tr
        self.ts = ts
        self.frac = frac
        self.max_req = max_req
        self.upd = upd
        
    def init(self):
        self.state = {}
        
    def next(self, usr_action):
        pass
