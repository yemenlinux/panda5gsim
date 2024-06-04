
import os
import sys
import time
import numpy as np
# Panda3D modules
from direct.showbase.ShowBase import ShowBase

from panda3d.core import (
    Filename,
    Point3,
    LVector3,
    NativeWindowHandle,
    CollisionTraverser, 
    CollisionHandlerEvent,
    WindowProperties,
    MeshDrawer,
    NodePath,
    Texture,
    DirectionalLight,
    AmbientLight,
    PointLight,
)

# add panda5gSim to path
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from panda5gSim import ASSETS_DIR, OUTPUT_DIR#, SimData
from panda5gSim.simConfig import SCENARIOS
from panda5gSim.core.streets import UrbanCityMap
from panda5gSim.core.buildings import Building
from panda5gSim.users.receiver import Device
from panda5gSim.metrics.trans_updator import TransformReader
# from panda5gSim.metrics.manager import TransMgr

class TransformCollector(ShowBase):
    def __init__(self, **kwargs):
        ShowBase.__init__(self)
        # set window title
        # base.win.set_title("Panda5gSim")
        # set window size
        # base.win.set_size(800, 600)
        self.gpos_step_size = 50
        self.num_gpos = 50
        self.apos_step_size = 100
        self.num_apos = 5
        # self.airBS_heights = [51.5, 101.5, 201.5, 501.5]
        self.airBS_heights = [21.5] + np.arange(51.5, 1001.5, 50).tolist()
        self.building_distribution = 'Variant'
        self.filename = 'experiment_grid.csv'
        #
        for k, v in kwargs.items():
            setattr(self, k, v)
            print(f'{k}: {v}')
        print(f'kwargs: {kwargs}')
        print(f'Building distribution: {self.building_distribution}')
        print(f'filename: {self.filename}')
        
        self.load_flag = True
        self.collect_flag = False
        self.loadTerrain()
        
        self.scenarios = list(SimData['Scenarios'].keys())
        # taskMgr.add(self.collect,'Collect_task')
        self.taskMgr.add(self.simulate, 'Simulate')
        self.taskMgr.add(self.exit_watcher,'Exit_watcher', delay=20)
        
        
        
    def simulate(self, task):
        if len(self.scenarios) == 0:
            print('Simulation completed')
            return task.done
        # check if there is a running task
        # has_task = self.taskMgr.hasTaskNamed('Collect_task')
        # print(f'has_task: {has_task}')
        if self.load_flag:
            # if hasattr(self, 'loaded'):
            #     self.destroyScenario()
            #     del self.loaded
            self.scenario = self.scenarios.pop()
            print(f'Running scenario: {self.scenario}')
            # self.loadScenario(scenario)
            # add task and check if it is done
            self.loadScenario()
            # self.taskMgr.add(self.loadScenario, 
            #                 'Generate_task',
            #                 appendTask=True,
            #                 extraArgs=[self.scenario])
            self.load_flag = False
            self.collect_flag = True
            self.taskMgr.add(self.collect,'Collect_task', delay=3)
        
                
        return task.cont
        
    def exit_watcher(self, task):
        if (not self.taskMgr.hasTaskNamed('Collect_task')
            and len(self.scenarios) == 0
            and not self.taskMgr.hasTaskNamed('Simulate')):
            print('Simulation user exit')
            self.destroyScenario()
            self.userExit()
            return task.done
        return task.cont
        
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
        # load devices
        gPos = self.city.getGridPos(self.num_gpos, filter_indoor=True)
        aPos = self.city.getGridPos(self.num_apos, filter_indoor=False)
        print(f'Number of generated ground users: {len(gPos)}')
        print(f'Number of generated air users: {len(aPos)}')
        self.gNodes = []
        self.aNodes = []
        # for i in range(len(gPos)):
        #     x, y = gPos[i]
        for x,y in gPos:
            # pos = LVector3(x*self.gpos_step_size+bbox[0], 
            #             y*self.gpos_step_size+bbox[1], 1.5)
            pos = LVector3(x, y, 1.5)
            node = Device(f'gUT')
            node.setPos(pos)
            node.reparentTo(render)
            self.gNodes.append(node)
        # for i in range(len(aPos)):
        #     x, y = aPos[i]
        #     pos = LVector3(x*self.apos_step_size+bbox[0], 
        #                 y*self.apos_step_size+bbox[1], self.airBS_heights[0])
        for x,y in aPos:
            pos = LVector3(x, y, self.airBS_heights[0])
            node = Device(f'airBS')
            node.setPos(pos)
            node.reparentTo(render)
            self.aNodes.append(node)
        # if not hasattr(self, 'TrasReader'):
        self.TrasReader = TransformReader(
                ['gBS_Tx', 'airBS_Tx'],
                ['gUT_Rx', 'airUT_Rx'])
        
        self.loaded = True
        
        # return task.done
        #
        
    def collect(self, task):
        if self.collect_flag:
            scenario = self.scenario
            print(f'TransformReader created: {scenario}')
            nrx = len(self.TrasReader.RxNodes)
            ntx = len(self.TrasReader.TxNodes)
            print(f'{ntx} Tx nodes, {nrx} Rx nodes, Total: {ntx*nrx}')
            for h in self.airBS_heights:
                print(f'Collecting transforms for {scenario} at height {h-1.5}')
                for node in self.aNodes:
                    node.setZ(h)
                # collect transforms
                self.collectTransforms(scenario)
                print(f'Collected transforms for {scenario} at height {h}')
            self.collect_flag = False
            if len(self.scenarios) > 0:
                self.destroyScenario()
                self.load_flag = True
            #     return task.cont
            # else:
            return task.done
        return task.cont
        
    def destroyScenario(self):
        # del self.TrasReader
        # destroy devices
        for node in self.gNodes:
            node.removeNode()
            del node
        for node in self.aNodes:
            node.removeNode()
            del node
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
        
        base.cam.setPos(0, -1500, 600)
        base.cam.lookAt(0, 400, 20)
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
        
    def collectTransforms(self, scenario):
        alpha = SimData['Scenarios'][scenario]['Alpha'] # type: ignore
        beta = SimData['Scenarios'][scenario]['Beta'] # type: ignore
        gamma = SimData['Scenarios'][scenario]['Gamma'] # type: ignore
        # freq = SimData['Frequency (GHz)'] # type: ignore
        #
        TransformDF = self.TrasReader.getTransformsDF()
        TransformDF['Environment'] = scenario
        TransformDF['Alpha'] = alpha
        TransformDF['Beta'] = beta
        TransformDF['Gamma'] = gamma
        # TransformDF['Frequency'] = self.freq
        # write metrics
        file_dir = OUTPUT_DIR + '/metrics/'
        filename = Filename(file_dir , self.filename)
        filename.make_dir()
        if not os.path.isfile(filename):
            TransformDF.to_csv(filename, index=False, header=True)
        else: # else it exists so append without writing the header
            TransformDF.to_csv(filename,
                            index=False, mode='a', header=False)
        return True
        
    
# run with even distribution with 25% variant
# app = TransformCollector(building_distribution='Variant',
#                         filename='experiment_skewed_grid.csv')
# app.run()

# run with 2DPPP distribution
# app = TransformCollector(building_distribution='2DPPP',
#                         filename='experiment_2DPPP.csv')
# app.run()

# run with 2DPPP distribution
app = TransformCollector(building_distribution='Regular',
                        filename='experiment_grid.csv')
app.run()