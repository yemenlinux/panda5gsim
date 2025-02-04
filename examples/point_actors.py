import os
import sys
import numpy as np
np.random.seed(123)
# add panda5gSim to path
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
sys.path.append(PARENT_DIR)
# panda5gSim modules

from panda5gSim.simmgrs.simmgr import (
    SimManager,
    
)

# Simulation_time_per_scenario = 60
# sim_delay = 5

# airBS heights
# airBS_heights = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
airBS_heights = [100, 100]
# airBS_heights = [100]

# Number of environments to generate = 3^num + 3 
# 3 is the standard environments as in the ITU-R P.1410-5 model
num = 4
        
print('Panda5gSim: Directional LOS with Mobility Simulation')
print('Starting Simulation')

building_dist = 'Variant'
app = SimManager(#building_distribution=building_dist,
                # alpha = 0.5,
                # beta = 300,
                # gamma = 50,
                # num_gUEs = 1,
                metric_filename = f'balanced_env_soj_120s_1.csv',
                airBS_heights = airBS_heights,
                num_rec_collect = 120,
                point_actors = True,
                # output_transform_filename= f'transforms_{building_dist}.csv'
                )
# Generate Environments
# generate ITU-R P.1410-5 standard environments
# app.environments = [(0.5, 300, 50), (0.5, 300, 20), (0.3, 500, 15)]
# for alpha in np.linspace(0.1, 0.7, num).round(2):
#     for beta in np.linspace(100, 750, num, dtype=int):
#         for gamma in np.linspace(8, 50, num, dtype=int):
#             app.environments.append((alpha, beta, gamma))
# np.random.shuffle(app.environments)

app.environments = [[0.41, 100, 36], [0.51, 363, 17], [0.65, 633, 12], [0.31, 692, 18], [0.76, 329, 22], [0.74, 702, 18], [0.72, 319, 31],
[0.75, 217, 44], [0.47, 579, 26], [0.59, 656, 30], [0.59, 392, 44], [0.79, 675, 47], [0.49, 621, 40], [0.43, 676, 39],
[0.33, 511, 37], [0.35, 634, 43], [0.29, 479, 42], [0.27, 743, 34], [0.25, 661, 23], [0.24, 646, 30], [0.19, 735, 47],
[0.21, 398, 43], [0.18, 586, 41], [0.15, 692, 43], [0.2, 422, 36], [0.16, 721, 33], [0.16, 293, 49], [0.11, 460, 47],
[0.13, 477, 40], [0.2, 682, 24], [0.15, 447, 35], [0.11, 499, 37], [0.1, 566, 34], [0.13, 266, 44], [0.15, 428, 32],
[0.18, 273, 36], [0.14, 399, 31], [0.19, 283, 33], [0.16, 312, 32], [0.16, 668, 21], [0.15, 710, 20], [0.1, 634, 20],
[0.15, 526, 21], [0.12, 201, 31], [0.16, 114, 40], [0.12, 625, 14], [0.17, 217, 26], [0.13, 232, 16], [0.17, 105, 28],
[0.19, 149, 23], [0.21, 673, 12], [0.24, 359, 23], [0.23, 129, 24], [0.24, 649, 10], [0.25, 525, 19], [0.26, 588, 10],
[0.28, 308, 8], [0.28, 387, 12], [0.29, 506, 11], [0.3, 274, 15], [0.27, 236, 27], [0.32, 695, 9], [0.32, 530, 12],
[0.33, 236, 18], [0.3, 168, 29], [0.38, 287, 13], [0.34, 713, 12], [0.43, 494, 9], [0.52, 207, 8], [0.51, 473, 8],
[0.51, 198, 15], [0.56, 532, 9], 
# [0.10, 750, 8], [0.5, 300, 50], [0.5, 300, 20], [0.3, 500, 15], [0.55, 680, 12.69]
]

# app.environments = [[0.51, 473, 8]]
# app.genEnvironment()
# app.sort_by_complex_argument()

app.Simulate()
app.run() 

