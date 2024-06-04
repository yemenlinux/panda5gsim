import os
import sys

# add panda5gSim to path
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
sys.path.append(PARENT_DIR)
# panda5gSim modules

from panda5gSim.simMgr import SimManager


        
building_dist = 'Variant'
app = SimManager(building_distribution=building_dist,
                alpha = 0.5,
                beta = 300,
                gamma = 50,
                output_transform_filename= f'transforms_{building_dist}.csv')
# Generate Environment
app.genEnvironment()
app.run() 
