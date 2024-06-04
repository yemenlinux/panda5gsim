import os
import sys

# add panda5gSim to path
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
sys.path.append(PARENT_DIR)
# panda5gSim modules
# from .los_with_ordinary_distribution import *
from examples.los_with_ordinary_distribution import TransformCollector


        
        
app = TransformCollector(building_distribution='2DPPP',
                        filename='experiment_grid_2DPPP.csv')
app.run()