# this file shows the environment only

import os
import sys
import time
import numpy as np
# Panda3D modules
from direct.showbase.ShowBase import ShowBase

from panda3d.core import (
    Filename,
    LVector3,
    Texture,
)

# add panda5gSim to path
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from panda5gSim import ASSETS_DIR, OUTPUT_DIR#, SimData
from panda5gSim.core.streets import UrbanCityMap
from panda5gSim.core.buildings import Building
from panda5gSim.users.receiver import Device
from panda5gSim.metrics.trans_updator import TransformReader
# from panda5gSim.metrics.manager import TransMgr

class Environments(ShowBase):
    def __init__(self, **kwargs):
        ShowBase.__init__(self)
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.loadTerrain()
        self.loadScenario()
        
    def loadScenario(self):
        # load scenario
        scenario = self.scenario
        bbox = SimData['Simulation area boundaries per m']
        self.genStreetMap(
            alpha = SimData['Scenarios'][scenario]['Alpha'],
            beta = SimData['Scenarios'][scenario]['Beta'],
            gamma = SimData['Scenarios'][scenario]['Gamma'],
            bbox=bbox,
            varStWidth = self.building_distribution,
        )
        # buildings
        self.genBuildings()
        
    def destroyScenario(self):
        # destroy buildings
        for b in self.buildings:
            b.destroy()
            del b
        
    def loadTerrain(self, street_map=None):
        self.terrain = loader.loadModel(ASSETS_DIR + '/models/plane.egg')
        self.terrain.reparentTo(render)
        
        bbox = SimData['Simulation area boundaries per m'] 
        tbbox = self.terrain.getTightBounds()
        print(f'tbbox: {tbbox}')
        self.terrain.setScale(
            (bbox[2] - bbox[0]) / (tbbox[1][0] - tbbox[0][0]),
            (bbox[3] - bbox[1]) / (tbbox[1][1] - tbbox[0][1]),
            1
        )
        # self.terrain.setScale(1000)
        # self.terrain.setHpr(0, -90, 0)
        self.terrain.setPos(0, 0, 0)
        #
        skydome_scale = 30 # was 15
        # load skydome
        skydome_model = ASSETS_DIR + '/models/skydome.egg'
        self.skydome1 = loader.loadModel(skydome_model)      
        self.skydome1.reparentTo(render)
        self.skydome1.setPos(0,0,0)
        self.skydome1.setScale(skydome_scale) #8
        # 
        self.skydome2 = loader.loadModel(skydome_model)
        self.skydome2.reparentTo(render)
        self.skydome2.setP(180)
        self.skydome2.setH(270)
        self.skydome2.setScale(skydome_scale) # 8
        # tags
        self.terrain.setTag('type', 'terrain')
        self.skydome1.setTag('type', 'skydome')
        self.skydome2.setTag('type', 'skydome')
        
        base.cam.setPos(-0, -1600, 500)
        base.cam.lookAt(0, 0, 0)
        # base.cam.setHpr(0, -32, 0)
        
    def genStreetMap(self, 
                    alpha,
                    beta,
                    gamma,
                    bbox = (-600, -600, 600, 600),
                    street_map=None, 
                    varStWidth = 'Variant',
                    varPercent = 0.25,
                    height_dist = 'Rayleigh',
                    navMesh_stepSize=20):
        if not street_map:
            street_map = OUTPUT_DIR + '/street_map.jpg'
        #
        filename = OUTPUT_DIR + f'/street_map_a{alpha}b{beta}g{gamma}_{varStWidth}'.replace('.','_')+'.jpg'
        self.city = UrbanCityMap(bounding_area = bbox, 
                                alpha=alpha, 
                                beta=beta, 
                                gamma=gamma,
                                varStWidth=varStWidth,
                                varPercent=varPercent,
                                height_dist=height_dist,
                                stepSize=navMesh_stepSize,
                                filename=filename)
        
        #
        shape = self.city.streetMap.shape
        tex = Texture()
        tex.setCompression(Texture.CM_off)
        tex.setup2dTexture(shape[0], shape[1], Texture.T_unsigned_byte, Texture.F_rgb8)
        tex.setRamImage(self.city.getStreetMap())
        # tex.setRamImage(self.city.buildingsMap)
        # tex = loader.loadTexture(street_map)
        self.terrain.setTexture(tex)
        
    def genBuildings(self, n_buildings = None, quads = None, Propability = None):
        if not hasattr(self, 'city'):
            raise ValueError('CityMap object not found. Call readStreetMap() first.')
        # generate buildings
        buildingData = self.city.getBuildingData()
        self.buildings = []
        b_tex = ['b0.jpg', 'b1.jpg', 'b2.jpg', 'b3.jpg']
        r_tex = ['r0.jpg', 'r1.jpg', 'r2.jpg']
        height_list = []
        # name, centerPos, width, depth, height, side_texture_path, roof_texture_path
        for i in range(len(buildingData)):
            bname = f'Building{i}'
            # b = self.getBuildingData[i] 
            center, width, depth, height = buildingData[i] 
            texture_path = ASSETS_DIR + '/textures/' + np.random.choice(b_tex)
            roof_texture_path = ASSETS_DIR + '/textures/' + np.random.choice(r_tex)
            b = Building(
                bname, 
                center, 
                width, 
                depth, 
                height, 
                texture_path, 
                roof_texture_path)
            self.buildings.append(b)
            # collect heights
            height_list.append(height)
        _min = min(height_list)
        _max = max(height_list)
        _mean = np.mean(height_list)
        _std = np.std(height_list)
        print(f"""Building heights: 
              min: {_min}, 
              max: {_max}, 
              mean: {_mean}, 
              std: {_std}
              """)
        
    
#['High-rise Urban', 'Dense Urban', 'Urban', 'Suburban']:
# run with 2DPPP distribution
# app = Environments(scenario = 'High-rise Urban',
#                     building_distribution='2DPPP',
#                     filename='experiment_grid_2DPPP.csv')
# app.run()

# app = Environments(scenario = 'Dense Urban',
#                     building_distribution='2DPPP',
#                     filename='experiment_grid_2DPPP.csv')
# app.run()

# app = Environments(scenario = 'Urban',
#                     building_distribution='2DPPP',
#                     filename='experiment_grid_2DPPP.csv')
# app.run()

# app = Environments(scenario = 'Suburban',
#                     building_distribution='2DPPP',
#                     filename='experiment_grid_2DPPP.csv')
# app.run()

# run with even distribution with 25% variant
app = Environments(scenario = 'High-rise Urban',
                    building_distribution='Variant',
                    filename='experiment_grid_Variant.csv')
app.run()

app = Environments(scenario = 'Dense Urban',
                    building_distribution='Variant',
                    filename='experiment_grid_Variant.csv')
app.run()

app = Environments(scenario = 'Urban',
                    building_distribution='Variant',
                    filename='experiment_grid_Variant.csv')
app.run()

app = Environments(scenario = 'Suburban',
                    building_distribution='Variant',
                    filename='experiment_grid_Variant.csv')
app.run()




# app = Environments(scenario = 'High-rise Urban',
#                     building_distribution='Regular',
#                     filename='experiment_grid_Variant.csv')
# app.run()

# app = Environments(scenario = 'Dense Urban',
#                     building_distribution='Regular',
#                     filename='experiment_grid_Regular.csv')
# app.run()

# app = Environments(scenario = 'Urban',
#                     building_distribution='Regular',
#                     filename='experiment_grid_Regular.csv')
# app.run()

# app = Environments(scenario = 'Suburban',
#                     building_distribution='Regular',
#                     filename='experiment_grid_Regular.csv')
# app.run()