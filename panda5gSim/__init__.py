#
# Get the base directory of the project
import os
import sys
# if sys.version_info >= (3, 0):
import builtins
# else:
#     import __builtin__ as builtins
from .simConfig import SimulationData


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.realpath(os.path.join(BASE_DIR, os.pardir))
OUTPUT_DIR = os.path.realpath(os.path.join(PARENT_DIR, 'output'))
ASSETS_DIR = os.path.realpath(os.path.join(PARENT_DIR, 'assets'))
NATIVE_MODULES_DIR = os.path.join(BASE_DIR, 'native_modules')
# contributed modules directory (plugins)
CONTRIB_MODULES_DIR = os.path.join(PARENT_DIR, 'plugins')
# add the contrib modules directory to the python path
# check if it exists
if not os.path.isdir(CONTRIB_MODULES_DIR):
    os.mkdir(CONTRIB_MODULES_DIR)
# add the contrib modules directory to the python path
# import sys
# sys.path.append(CONTRIB_MODULES_DIR)
    
# The main Simulation Data Structure
builtins.SimData = {} 

# update the simulation data structure with the default values
SimData.update(SimulationData) # type: ignore


