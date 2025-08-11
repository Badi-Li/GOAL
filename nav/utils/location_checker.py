import numpy as np 
class LocationChecker(object):
    def __init__(self, length, deadlockthr):
        super().__init__()
        self.locations = []
        self.length = length 
        self.deadlockthr = deadlockthr
    
    def reset(self):
        self.locations = []
    
    def insert(self, loc):
        if len(self.locations) >= self.length:
            self.locations.pop(0)
        self.locations.append(np.array(loc))

    def deadlock(self):
        if len(self.locations) < self.length:
            return False 

        base = self.locations[0]
        max_dist = max(np.linalg.norm(loc - base) for loc in self.locations)

        return max_dist < self.deadlockthr