import os
import sys
import numpy as np

# add panda5gSim to path
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
sys.path.append(PARENT_DIR)
# panda5gSim modules

from panda5gSim.simMgr import (
    SimManager,
    
)

Simulation_time_per_scenario = 60
num_environments = 10
sim_delay = 5
airBS_heights = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

num = 4
        
print('Panda5gSim: Directional LOS with Mobility Simulation')
print('Starting Simulation')

building_dist = 'Variant'
app = SimManager(building_distribution=building_dist,
                # alpha = 0.5,
                # beta = 300,
                # gamma = 50,
                metric_filename = f'test.csv',
                airBS_heights = airBS_heights,
                num_rec_collect = 200,
                output_transform_filename= f'transforms_{building_dist}.csv')
# Generate Environments
# generate ITU-R P.1410-5 environments
app.environments = [(0.5, 300, 50), (0.5, 300, 20), (0.3, 500, 15)]
for alpha in np.linspace(0.1, 0.7, num).round(2):
    for beta in np.linspace(100, 750, num, dtype=int):
        for gamma in np.linspace(8, 50, num, dtype=int):
            app.environments.append((alpha, beta, gamma))
# np.random.shuffle(app.environments)

# app.environments = [(0.5, 300, 50), (0.5, 300, 20), (0.3, 500, 15), 
#                     (0.1, 750, 8), ]
# app.genEnvironment()
app.sort_by_complex_argument()

app.Simulate()
app.run() 

