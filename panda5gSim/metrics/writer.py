# This model contains a DirectObject class that collects the metrics
# and writes them to files. The class discovers the buildings, user actors
# bs actors path nodes from the scene graph and calculates the metrics.

import csv
import os
# from panda3d.core import *
from direct.showbase.DirectObject import DirectObject
from direct.task import Task


from panda5gSim import ASSETS_DIR, OUTPUT_DIR
from panda5gSim.core.scene_graph import findNPbyTag, has_line_of_sight


class Writer:
    def __init__(self, file_name, csv_header):
        #
        self.file_name = file_name
        self.header = csv_header
        # open csv file and write the header
        directory = OUTPUT_DIR + '/metrics/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.csv_file = directory + f'{self.file_name}.csv'
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w') as csvfile:  
                writer = csv.DictWriter(csvfile, self.header)
                writer.writeheader()
                # writer.writerow(my_dict)
        
    def writerow(self, row):
        with open(self.csv_file, 'a') as csvfile:  
            writer = csv.DictWriter(csvfile, self.header)
            writer.writerow(row)
            
    

class Writer_old(DirectObject):
    def __init__(self, file_name, csv_header, simFixed = {}, period = 1):
        DirectObject.__init__(self)
        self.period = period
        self.file_name = file_name
        self.header = csv_header
        for k,v in simFixed.items():
            if k not in self.header:
                self.header.append(k)
        self.simFixed = simFixed
        self.findActors()
        # open csv file and write the header
        directory = OUTPUT_DIR + '/metrics/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.csv_file = directory + f'{file_name}.csv'
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w') as csvfile:  
                writer = csv.DictWriter(csvfile, self.header)
                writer.writeheader()
                # writer.writerow(my_dict)
        
        #
        self.accept("destroy", self.destroy)
        #
        self.taskMgr.doMethodLater(self.period, 
                    self.update, f"writeMetrics_{self.file_name}")
        
    def findActors(self):
        self.gUT = findNPbyTag('ground_actor', 'type')
        self.airUT = findNPbyTag('air_actor', 'type')
        self.airBS = findNPbyTag('air_bs', 'type')
        self.gBS = findNPbyTag('ground_bs', 'type')
        
    def writerow(self, row):
        with open(self.csv_file, 'a') as csvfile:  
            writer = csv.DictWriter(csvfile, self.header)
            writer.writerow(row)
            
    def update(self, task):
        # override this method
        return Task.cont
    
    def destroy(self):
        self.ignoreAll()
        # self.taskMgr.remove(f"writeMetrics_{self.file_name}")
        self.removeAllTasks()
        del self
        print('writer destroyed')
        
class WriteRLB(Writer):
    def __init__(self, file_name, simFixed = {}, period = 1):
        csv_header = [
            'Time',
            'User name',
            'User type'
            'BS name', 
            'BS type',
            'User position',
            'BS position',
            'Distance',
            'Breakpoint',
            'LoS probability calculated',
            'LoS probability',
            'LoS real',
            
        ]
        
        Writer.__init__(self, file_name, csv_header, simFixed, period)
        
    def update(self, task):
        # combine the ground bs and air bs lists in one list
        bs_list = self.gBS + self.airBS
        user_list = self.gUT + self.airUT
        for bs in bs_list:
            for user in user_list:
                # calculate the distance between the user and the bs
                distance = getDistance3DPos(user.getPos(), bs.getPos())
                # check if the user is in LoS with the bs
                LoS = self.rayTest(user, bs)
                # calculate the LoS probability
                LoS_probability = self.LoSProbability(distance)
                
                # real LoS
                LoS_real = has_line_of_sight(bs, user)
                # write the row
                row = {
                    'Time': globalClock.getFrameTime(),
                    'User name': user.getName(),
                    'User type': user.getTag('type'),
                    'BS name': bs.getName(),
                    'BS type': bs.getTag('type'),
                    'User position': user.getPos(),
                    'BS position': bs.getPos(),
                    'Distance': distance,
                    'Breakpoint': LoS,
                    'LoS probability calculated': LoS_probability,
                    'LoS probability': LoS_probability,
                    'LoS real': LoS_real,
                }
                self.writerow(row)
        return Task.cont