# Simulation mangager using panda3d gui

import os
import time
import numpy as np

# Panda3D modules
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import Thread
from panda3d.core import (
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
# from direct.interval.IntervalGlobal import Sequence

# P5gSim modules
from panda5gSim import ASSETS_DIR, OUTPUT_DIR
from panda5gSim.camcontrols.cameraControl import CameraControl
from panda5gSim.core.streets import UrbanCityMap
from panda5gSim.core.buildings import Building
from panda5gSim.users.ut import GroundUser
from panda5gSim.users.uav import UAVuser
from panda5gSim.users.air_bs import UAVBs
from panda5gSim.users.bs import GBS
from panda5gSim.core.ai_world import AiWorld
from panda5gSim.core.scene_graph import findNPbyTag, has_line_of_sight
# from panda5gSim.metrics.collectors import RadioLinkBudgetMetricsCollector
from panda5gSim.metrics.manager import TransMgr
from panda5gSim.core.mobility_mgr import MobilityMgr
from panda5gSim.metrics.trans_updator import TransformReader

class SimManager(ShowBase):
    def __init__(self, 
                **kwargs
                ):
        ShowBase.__init__(self)
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        self.setup()
        
    def setup(self):
        if not hasattr(self, 'PANDA_WINDOW_WIDTH'):
            self.PANDA_WINDOW_WIDTH = 1200   
        if not hasattr(self, 'PANDA_WINDOW_HEIGHT'):
            self.PANDA_WINDOW_HEIGHT = 960
        if not hasattr(self, 'PANDA_WINDOW_TITLE'):
            self.PANDA_WINDOW_TITLE = 'Panda5gSim Simulation'
        if not hasattr(self, 'building_distribution'):
            self.building_distribution = '2DPPP'
        if not hasattr(self, 'street_variance_percentage'):
            self.street_variance_percentage = 0.25
        if not hasattr(self, 'alpha'):
            self.alpha = round(np.random.uniform(0.1, 0.8),1)
        if not hasattr(self, 'beta'):
            self.beta = np.random.randint(100,750)
        if not hasattr(self, 'gamma'):
            self.gamma = np.random.randint(8,50)
        if not hasattr(self, 'scenario'):
            if (self.alpha == 0.5
                and self.beta == 300
                and self.gamma == 50):
                self.scenario = 'High-rise Urban'
            elif (self.alpha == 0.5
                and self.beta == 300
                and self.gamma == 20):
                self.scenario = 'Dense Urban'
            elif (self.alpha == 0.3
                and self.beta == 500
                and self.gamma == 15):
                self.scenario = 'Urban'
            elif (self.alpha == 0.1
                and self.beta == 750
                and self.gamma == 8):
                self.scenario = 'Suburban'
            else:
                self.scenario = f'Urban_{self.alpha}_{self.beta}_{self.gamma}'
        if not hasattr(self, 'num_gUEs'):
            self.num_gUEs = 20
        if not hasattr(self, 'num_airBSs'):
            self.num_airBSs = 2
        if not hasattr(self, 'num_gBSs'):
            self.num_gBSs = 2
        if not hasattr(self, 'num_airUEs'):
            self.num_airUEs = 2
        if not hasattr(self, 'airBS_height'):
            self.airBS_height = 100
        if not hasattr(self, 'gBS_height'):
            self.gBS_height = 20
        if not hasattr(self, 'airUE_height'):
            self.airUE_height = 20
        if not hasattr(self, 'gUE_height'):
            self.gUE_height = 1.5
        if not hasattr(self, 'num_mobility_PoIs'):
            self.num_mobility_PoIs = 5
        if not hasattr(self, 'sim_boundaries'):
            self.sim_boundaries = [-600, -600, 600, 600]
        if not hasattr(self, 'sim_duration'):
            self.sim_duration = 60
        if not hasattr(self, 'navMesh_stepSize'):
            self.navMesh_stepSize = 20
        if not hasattr(self, 'building_heights_distribution'):
            self.building_heights_distribution = 'Rayleigh'
        if not hasattr(self, 'frequency_GHz'):
            self.frequency_GHz = 28 # GHz
        if not hasattr(self, 'communication_channel'):
            self.communication_channel = 'GA2GA'
        if not hasattr(self, 'link_direction'):
            self.link_direction = 'downlink'
        if not hasattr(self, 'metric_filename'):
            self.metric_filename = 'metrics.csv'
        
        # 
        if not hasattr(self, 'TxPower'):
            self.TxPower = 30 # dBm
        if not hasattr(self, 'building_penetration_loss'):
            if self.frequency_GHz < 6:
                self.building_penetration_loss = 20
            elif self.frequency_GHz < 20:
                self.building_penetration_loss = 25
            else:
                self.building_penetration_loss = 55
        
            
        if not hasattr(self, 'tx_antenna_gain'):
            self.tx_antenna_gain = 0
        if not hasattr(self, 'rx_antenna_gain'):
            self.rx_antenna_gain = 0
        
        # Setup window
        self.wp = WindowProperties()
        self.wp.setOrigin(0, 0)
        self.wp.setSize(self.PANDA_WINDOW_WIDTH, self.PANDA_WINDOW_HEIGHT)
        self.openDefaultWindow(props=self.wp)
        # Collision detection
        traverser = CollisionTraverser('traverser')
        base.cTrav = traverser # type: ignore
        
        # method of drawing points 
        self.render.setRenderModePerspective(True)
        # load terrain
        # self.load_flag = True
        # self.collect_flag = False
        self.loadTerrain()
        
        
        # # taskMgr.add(self.collect,'Collect_task')
        # self.taskMgr.add(self.simulate, 'Simulate')
        # self.taskMgr.add(self.exit_watcher,'Exit_watcher', delay=20)
        # CameraControl
        # self.cameraControl = CameraControl()
        
        
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
        
        bbox = self.sim_boundaries 
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
        
        base.cam.setPos(0, -500, 100)
        base.cam.lookAt(0, -32, 0)
        # base.cam.setHpr(0, -32, 0)
        
    def genStreetMap(self, 
                    alpha = 0.5,
                    beta = 300,
                    gamma = 50,
                    street_map=None, 
                    varStWidth = 'Variant',
                    varPercent = 0.25,
                    height_dist = 'Rayleigh',
                    navMesh_stepSize=20):
        if not street_map:
            street_map = OUTPUT_DIR + '/street_map.jpg'
        # bounds = SimData['Simulation area boundaries per m']
        bbox = self.sim_boundaries
        # alpha = SimData['Scenarios'][self.gui.scenario]['Alpha'] # type: ignore
        # beta = SimData['Scenarios'][self.gui.scenario]['Beta'] # type: ignore
        # gamma = SimData['Scenarios'][self.gui.scenario]['Gamma'] # type: ignore
        # scenario = self.gui.scenarioDict[self.gui.scenario]
        # print(f'Scenario:{self.gui.scenario}, {scenario}')
        filename = OUTPUT_DIR + f'/street_map_a{alpha}b{beta}g{gamma}_{varStWidth}'.replace('.','_')+'.jpg'
        self.city = UrbanCityMap(bounding_area = bbox, 
                                alpha=alpha, 
                                beta=beta, 
                                gamma=gamma,
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
        #
        # generate and save navigation mesh
        self.city.writeNavMesh()
        for h in [self.airBS_height, self.airUE_height]:
            # filename = f"navmesh_a{alpha}b{beta}g{gamma}h{h}".replace('.','_')+".csv"
            self.city.writeNavMesh(height = h)
        
    def writeNavMesh(self, 
                    step = 5, 
                    height = 0):
        if height > 0:
            fname = f'navmesh{height}.csv'
        else:
            fname = 'navmesh.csv'
        fpath = OUTPUT_DIR + '/' + fname
        if not os.path.exists(fpath):
            self.city.writeNavMesh(
                step = step,
                height = height,
                filename = fname)
        
    def genBuildings(self):
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
        
    def genGroundUsers(self):
        # ground users
        # self.number_of_gUT = self.calculate_n_gUT()
        # print(f'Number of ground users: {self.number_of_gUT}')
        self.ground_users = []
        self.ground_users_locations = self.city.getGPositions(height = 0)
        indices = np.random.choice(range(len(self.ground_users_locations)), 
                                    self.num_gUEs)
        positions = [self.ground_users_locations[i] for i in indices]
        # print(f'Number of ground users: = {len(positions)}')
        # print(f'Ground user positions: = {positions}')
        # a,b,g = self.getEnvParam()
        nav_mesh_file = f"navmesh_a{self.alpha}b{self.beta}g{self.gamma}h{0}".replace('.','_')+".csv"
        nav_mesh_path = OUTPUT_DIR + '/navMesh/' + nav_mesh_file
        for i in range(len(positions)):
            p = positions[i]
            param = {
                'ID': i,
                'name': f'UE{i}',
                'type': 'UE',
                'device_type': 'gUT',
            }
            # param.update(SimData['UT defaults'])
            ue = GroundUser(**param)
            ue.setPos(p)
            ue.setNavMesh(nav_mesh_path)
            self.ground_users.append(ue)
    
    def genAirUsers(self):
        self.number_of_airUT = self.num_airUEs
        flight_height = self.airUE_height
        self.air_users_locations = self.city.getGPositions(height = flight_height)
        indices = np.random.choice(range(len(self.air_users_locations)), self.number_of_airUT)
        positions = [self.air_users_locations[i] for i in indices]
        # print(f'Number of air users: = {len(positions)}')
        # print(f'positions: = {positions}')
        # a,b,g = self.getEnvParam()
        self.air_users = []
        for i in range(len(positions)):
            p = positions[i]
            param = {
                'ID': i,
                'name': f'UU{i}',
                'type': 'UU',
                # 'position': positions[i],
                'device_type': 'airUT',
            }
            au = UAVuser(**param)
            au.setPos(p)
            nav_mesh_file = f"navmesh_a{self.alpha}b{self.beta}g{self.gamma}h{flight_height}".replace('.','_')+".csv"
            nav_mesh_path = OUTPUT_DIR + '/navMesh/' + nav_mesh_file
            au.setNavMesh(nav_mesh_path)
            self.air_users.append(au)
            # add mobility parameters
            # self.air_users[i].reparentTo(render)
            # self.air_users[i].Actor.setPos(positions[i])
            # self.air_users[i].Actor.setZ(20.0)
    
    def genAirBS(self):
        # self.number_of_airBS = SimData['Number of air BSs']
        airBS_heights = self.airBS_height
        self.number_of_airBS = self.num_airBSs
        flight_height = self.airBS_height
        self.air_bs_locations = self.city.getGPositions(height = 0)
        indices = np.random.choice(range(len(self.air_bs_locations)), 
                                    self.num_airBSs)
        positions = [self.air_bs_locations[i] for i in indices]
        # print(f'Number of air bs: = {len(positions)}')
        a,b,g = self.alpha, self.beta, self.gamma
        
        self.air_bs = []
        for i in range(len(positions)):
            p = (positions[i][0], positions[i][1], self.airBS_height)
            param = {
                'ID': i,
                'name': f'UB{i}',
                'type': 'UB',
                # 'position': positions[i],
                'device_type': 'airBS',
            }
            au = UAVBs(**param)
            au.setPos(p)
            nav_mesh_file = f"navmesh_a{a}b{b}g{g}h{self.airBS_height}".replace('.','_')+".csv"
            nav_mesh_path = OUTPUT_DIR + '/navMesh/' + nav_mesh_file
            au.setNavMesh(nav_mesh_path)
            self.air_bs.append(au)
            #
            # self.air_users[i].reparentTo(render)
            # self.air_bs[i].Actor.setPos(positions[i])
    
    def genGroundBS(self):
        self.gBS_pos = []
        print(f'remaining pos: {len(self.gBS_pos)}, {self.gBS_pos[0]}')
        self.gBS = []
        ngBS = 1
        height = 20
        cell_type = 'directional'
        pos = self.gBS_pos[np.random.choice(range(len(self.gBS_pos)), ngBS)[0]]
        #
        name = f'gBS{0}'
        bs = GBS(name)
        bs.setPos(pos[0])
        bs.setTowerHeight(height)
        bs.addCell(cell_type, num_of_sectors = 3)
        self.gBS.append(bs)
    
    def setAI(self):
        self.aiWorld = AiWorld(self)
    
    def destroyBuildings(self):
        for b in self.buildings:
            b.destroy()
            del b
        self.buildings = []
    
    def destroyGBS(self):
        for bs in self.gBS:
            bs.destroy()
            del bs
        self.gBS = []
    
    def genEnvironment(self):
        self.genStreetMap(
            alpha = self.alpha,
            beta = self.beta,
            gamma = self.gamma,
            varStWidth = self.building_distribution,
            navMesh_stepSize = self.navMesh_stepSize,
            height_dist = self.building_heights_distribution,
            varPercent = self.street_variance_percentage,
        )
        #
        self.genBuildings()
        self.genGroundUsers()
        self.genAirUsers()
        self.genAirBS()
        # self.genGroundBS()
        # Generate Ai
        self.setAI()
        # self.makeLight()
        #
        positions = self.city.getGPositions(height = 0)
        self.mobMgr = MobilityMgr(positions)
        # collect transforms ot Tx and Rx
        self.TrasReader = TransformReader(
                ['gBS_Tx', 'airBS_Tx'],
                ['gUT_Rx', 'airUT_Rx'])
        # self.collectTransforms()
        
        # method: automatically
        # timing of changing environment scenario periodically
        # scenario_delay = 20 # seconds
        # self.num_iterations = 0
        # self.taskMgr.doMethodLater(scenario_delay,  
        #                     self.transAutoUpdate, 
        #                     'SimTiming_task')
        
    def reGenEnvironment(self):
        self.transMgr.collecting = False
        time.sleep(1.5)
        self.taskMgr.remove('SimTiming_task')
        self.transMgr.destroy()
        # 
        self.mobMgr.destroy()
        del self.mobMgr
        # self.destroyGBS()
        self.aiWorld.destroy()
        del self.aiWorld
        for g in self.ground_users:
            g.destroy()
            del g
        for a in self.air_users:
            a.destroy()
            del a
        for a in self.air_bs:
            a.destroy()
            del a
        self.destroyBuildings()
        self.genEnvironment()
        
    def collectMetrics(self, task):
        #
        #
        TransformDF = self.TrasReader.getTransformsDF()
        env = f'Urban_{self.alpha}_{self.beta}_{self.gamma}'
        TransformDF['Environment'] = env
        # time stamp
        
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
        
    