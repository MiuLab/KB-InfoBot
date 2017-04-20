'''
'''


class DictReader:
    def __init__(self):
        pass
        
    def load_dict_from_file(self, path):
        slot_set = {}
    
        file = open(path, 'r')
        index = 0
        for line in file: 
            slot_set[line.strip('\n').strip('\r')] = index
            index += 1
        
        self.dict = slot_set
        
        
    def load_dict_from_array(self, array):
        slot_set = {}
        for index, ele in enumerate(array):
            slot_set[ele.strip('\n').strip('\r')] = index
        
        self.dict = slot_set
