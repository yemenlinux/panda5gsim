# 
import os
import time
import numpy as np
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
# Panda3D modules
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import Thread
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
from direct.interval.IntervalGlobal import Sequence

# P5gSim modules
from panda5gSim import ASSETS_DIR, OUTPUT_DIR
from panda5gSim.camcontrols.cameraControl import CameraControl
from panda5gSim.core.streets import CityMap, UrbanCityMap
from panda5gSim.core.buildings import Building
from panda5gSim.users.ut import GroundUser
from panda5gSim.users.uav import UAVuser
from panda5gSim.users.air_bs import UAVBs
from panda5gSim.users.bs import GBS
from panda5gSim.core.ai_world import AiWorld
from panda5gSim.core.scene_graph import findNPbyTag, has_line_of_sight
# from panda5gSim.metrics.collectors import RadioLinkBudgetMetricsCollector
from panda5gSim.metrics.manager import TransMgr, MetricsManager_old
from panda5gSim.core.mobility_mgr import MobilityMgr

class SimManager(ShowBase):
    def __init__(self, Gui):
        ShowBase.__init__(self)
        self.gui = Gui
        #
        screen_width = base.win.get_x_size()
        screen_height = base.win.get_y_size()

        print("Screen size: {}x{}".format(screen_width, screen_height))
        
        #
        self.PANDA_WINDOW_WIDTH = self.gui.panda_drawing_area.get_allocated_width()   # Width of panda window in GTK
        self.PANDA_WINDOW_HEIGHT = self.gui.panda_drawing_area.get_allocated_height()  # Height of panda window in GTK
        # Setup window
        self.wp = WindowProperties()
        self.wp.setOrigin(0, 0)
        self.wp.setSize(self.PANDA_WINDOW_WIDTH, self.PANDA_WINDOW_HEIGHT)
        # Get drawing area and set its size
        print(f'panda_drawing_area width: {self.gui.panda_drawing_area.get_allocated_width()}, height: {self.gui.panda_drawing_area.get_allocated_height()}')
        
        # Panda should not open own top level window but use the window of the drawing area in GTK
        handle = NativeWindowHandle.makeInt(self.gui.panda_drawing_area.get_property('window').get_xid())
        self.wp.setParentWindow(handle)
        # Open panda window
        self.openDefaultWindow(props=self.wp)
        # connect resize event
        self.gui.connect("size_allocate", self.on_window_resize)
        #
        # Create task to update GUI
        def gtk_iteration(task):
            """
            Handles the gtk events and lets as many GUI iterations run as needed.
            """
            while Gtk.events_pending():
                Gtk.main_iteration_do(False)
            return task.cont
        #
        self.taskMgr.add(gtk_iteration, "gtk")
        #
        # Collision detection
        traverser = CollisionTraverser('traverser')
        base.cTrav = traverser # type: ignore
        #traverser.addCollider(fromObject, handler)
        #
        self.render.setRenderModePerspective(True)
        # load terrain
        self.loadTerrain()
        # CameraControl
        self.cameraControl = CameraControl()
        
        # accepted messages
        self.num_iterations = 0
        self.accept("scenario_changed", self.reGenEnvironment)
        # self.accept('Simulation_Start', self.Simulate)
        # self.accept('Simulation_Stop', self.StopSimulate)
        # self.taskMgr.add(self.simulate, 'SimMgr', delay=2)
        
        
        

    def on_window_resize(self, window, allocation):
        """Get the size of the panda drawing area and the size of the frame
            in case of a resize event
            if the frame is not the same size as the drawing area then the drawing area is resized to match the frame
        """
        width, height = allocation.width, allocation.height
        if hasattr(self, 'gui_width') and hasattr(self, 'gui_height'):
            if self.gui_width == width and self.gui_height == height:
                return
        # get the size ot the combo box and the status bar
        self.gui.middle_height = \
            height - self.gui.top_vbox_size - self.gui.bottom_vbox_size
        self.gui.vpaned.set_position(self.gui.top_vbox_size)
        self.gui.vpaned2.set_position(self.gui.middle_height)
        
        # 
        print(f'width: {width}, height: {height}')
        self.PANDA_WINDOW_WIDTH = \
            self.gui.panda_drawing_area.get_allocated_width()   # Width of panda window in GTK
        self.PANDA_WINDOW_HEIGHT = \
            self.gui.panda_drawing_area.get_allocated_height()  # Height of panda window in GTK
        #
        self.wp = WindowProperties()
        self.wp.setOrigin(0, 0)
        # 
        self.wp.setSize(self.PANDA_WINDOW_WIDTH, self.PANDA_WINDOW_HEIGHT)
        # Panda should not open own top level window but use 
        # the window of the drawing area in GTK
        handle = NativeWindowHandle.makeInt(window.panda_drawing_area.get_property('window').get_xid())
        self.wp.setParentWindow(handle)
        # Open panda window
        self.openDefaultWindow(props=self.wp)
        
        self.gui_width, self.gui_height = width, height
        
        # 
        base.cam.setPos(0, -500, 100)
        # base.cam.lookAt(0, 0, 0)
        base.cam.setHpr(0, -32, 0)
    
    def setSimBounds(self, bounds):
        """
        Set the simulation area boundaries
        as a tuple of length 4 (xmin, ymin, xmax, ymax)
        or as a tuple of length 2 (x, y) which will be 
        converted to (-x/2, -y/2, x/2, y/2)
        Usage:
            setSimBounds((-500, -500, 500, 500))
            setSimBounds((1000, 1000))
        """
        if isinstance(bounds, tuple) and len(bounds) == 4:
            SimData['Simulation area boundaries per m'] = bounds
            SimData['Simulation area per square km'] = \
                (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / 1e6
        elif isinstance(bounds, tuple) and len(bounds) == 2:
            x, y = bounds
            SimData['Simulation area boundaries per m'] = \
                (-x/2, -y/2, x/2, y/2)
            SimData['Simulation area per square km'] = \
                (x) * (y) / 1e6
        else:
            raise ValueError('Bounds must be a tuple of length 4 or 2')
    
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
        
    def genStreetMap_old(self, street_map=None):
        if not street_map:
            street_map = OUTPUT_DIR + '/street_map.jpg'
        # bounds = SimData['Simulation area boundaries per m']
        bbox = SimData['Simulation area boundaries per m'] # type: ignore
        alpha = SimData['Scenarios'][self.gui.scenario]['Alpha'] # type: ignore
        beta = SimData['Scenarios'][self.gui.scenario]['Beta'] # type: ignore
        gamma = SimData['Scenarios'][self.gui.scenario]['Gamma'] # type: ignore
        scenario = self.gui.scenarioDict[self.gui.scenario]
        # print(f'Scenario:{self.gui.scenario}, {scenario}')
        self.city = CityMap(bounding_area = bbox, alpha=alpha, beta=beta, gamma=gamma)
        if street_map:
            street_map = street_map
        else:
            street_map = ASSETS_DIR + '/textures/street_map.jpg'
        if os.path.exists(street_map):
            self.city.readImage(street_map)
        else:
            self.city.genStreets()
            # self.city.writeImage2d(street_map)
        #
        shape = self.city.streetMap.shape
        tex = Texture()
        tex.setCompression(Texture.CM_off)
        tex.setup2dTexture(shape[0], shape[1], Texture.T_unsigned_byte, Texture.F_rgb8)
        tex.setRamImage(self.city.streetMap)
        # tex.setRamImage(self.city.buildingsMap)
        # tex = loader.loadTexture(street_map)
        self.terrain.setTexture(tex)
        #
        # generate and save navigation mesh
        self.city.genNavMesh()
        self.writeNavMesh()
        for h in SimData['Allowed flight range']:
            self.writeNavMesh(step = 10, height = h)
        
    def genStreetMap(self, 
                    street_map=None, 
                    varStWidth = 'Variant',
                    varPercent = 0.25,
                    height_dist = 'Rayleigh',
                    navMesh_stepSize=20):
        if not street_map:
            street_map = OUTPUT_DIR + '/street_map.jpg'
        # bounds = SimData['Simulation area boundaries per m']
        bbox = SimData['Simulation area boundaries per m'] # type: ignore
        alpha = SimData['Scenarios'][self.gui.scenario]['Alpha'] # type: ignore
        beta = SimData['Scenarios'][self.gui.scenario]['Beta'] # type: ignore
        gamma = SimData['Scenarios'][self.gui.scenario]['Gamma'] # type: ignore
        scenario = self.gui.scenarioDict[self.gui.scenario]
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
        for h in SimData['Allowed flight range']:
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
        
        
    def genBuildings_old(self, n_buildings = None, quads = None, Propability = None):
        if not hasattr(self, 'city'):
            raise ValueError('CityMap object not found. Call readStreetMap() first.')
        # generate buildings
        _buildingData = self.city.getBuildingData()
        print(f'Number of generated Buildings: {len(_buildingData)}')
        _indeces = range(len(_buildingData))
        if n_buildings:
            indeces = np.random.choice(_indeces, n_buildings, replace=False)
        # elif 'Number of buildings per square km' in SimData['Building Defaults']:
        #     n = SimData['Building Defaults']['Number of buildings per square km']
        #     indeces = np.random.choice(indeces, n)
        elif hasattr(self.city, 'num_buildings'):
            n = self.city.num_buildings
            indeces = np.random.choice(_indeces, n, replace=False)
            
        print(f'Number of Building pos: {len(_buildingData)}, number of buildings: {n}')
        
        # 
        buildingData = []
        self.gBS_pos = []
        for i in _indeces:
            if i in indeces:
                buildingData.append(_buildingData[i])
            else:
                self.gBS_pos.append(_buildingData[i])
        # buildingData = [_buildingData[i] for i in indeces]
        
        # all 
        # r = [i for i in _indeces if not i in indeces]
        # self.gBS_pos = [_buildingData[i] for i in r]
        # buildings
        self.buildings = []
        b_tex = ['b0.jpg', 'b1.jpg', 'b2.jpg', 'b3.jpg']
        r_tex = ['r0.jpg', 'r1.jpg', 'r2.jpg']
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
        print(f'Number of Building pos: {len(self.buildings)}')
        
    def genBuildings(self, n_buildings = None, quads = None, Propability = None):
        if not hasattr(self, 'city'):
            raise ValueError('CityMap object not found. Call readStreetMap() first.')
        # generate buildings
        buildingData = self.city.getBuildingData()
        self.buildings = []
        b_tex = ['b0.jpg', 'b1.jpg', 'b2.jpg', 'b3.jpg']
        r_tex = ['r0.jpg', 'r1.jpg', 'r2.jpg']
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
        
            
    def setAI(self):
        self.aiWorld = AiWorld(self)
        
    def calculate_n_gUT(self):
        """ Generates number of users to be placed in the simulation.
        The locations are generated in a rectangular area. 
        The area is divided into a grid of cells based on 
        the density parameter.
        number of locations = density * area 
        density = is the number of locations per square meter.
            if you have the density of UEs per square kilometer, 
            then you need to divide it by 1e6 to convert the 
            density to UEs per square meter
            0.1 means 1 location per 10 square meters of area.
            Telephony user density can be calculated for a 
            country or specific city as population density 
            (people per a square kilometer) times the telephony 
            density (number of users per person)
        """
        if 'Population density per sq. km' in SimData:
            pop_density = SimData['Population density per sq. km']
        else:
            pop_density = 650
        if 'Telephony density per population' in SimData:
            telephony_density = SimData['Telephony density per population']
        else:
            telephony_density = 0.8
        if 'Number of ground users' in SimData:
            return SimData['Number of ground users']
        else:
            return int(pop_density * telephony_density * SimData['Simulation area per square km'])
        
    def calculate_n_gUT_per_building(self, building):
        """ Generates number of users to be placed in the simulation.
        The locations are generated in a rectangular area. 
        The area is divided into a grid of cells based on 
        the density parameter.
        number of locations = density * area 
        density = is the number of locations per square meter.
            if you have the density of UEs per square kilometer, 
            then you need to divide it by 1e6 to convert the 
            density to UEs per square meter
            0.1 means 1 location per 10 square meters of area.
            Telephony user density can be calculated for a 
            country or specific city as population density 
            (people per a square kilometer) times the telephony 
            density (number of users per person)
        """
        building_bbox = building.getTightBounds()
        building_area = (building_bbox[1][0] - building_bbox[0][0]) * (building_bbox[1][1] - building_bbox[0][1])
        building_area_in_km = building_area / 1e6
        level_height = SimData['Building Defaults']['Floor height']
        number_of_levels = (building_bbox[1][2] - building_bbox[0][2]) // level_height
        if 'Population density per sq. km' in SimData:
            pop_density = SimData['Population density per sq. km']
        else:
            pop_density = 650
        if 'Telephony density per population' in SimData:
            telephony_density = SimData['Telephony density per population']
        else:
            telephony_density = 0.8
        return max(1, round(pop_density 
                            * telephony_density 
                            * building_area_in_km 
                            * number_of_levels))
    
    def genGroundUsers(self):
        # ground users
        self.number_of_gUT = self.calculate_n_gUT()
        # print(f'Number of ground users: {self.number_of_gUT}')
        self.ground_users = []
        self.ground_users_locations = self.city.getGPositions(height = 0)
        indices = np.random.choice(range(len(self.ground_users_locations)), 
                                    self.number_of_gUT)
        positions = [self.ground_users_locations[i] for i in indices]
        # print(f'Number of ground users: = {len(positions)}')
        # print(f'Ground user positions: = {positions}')
        a,b,g = self.getEnvParam()
        nav_mesh_file = f"navmesh_a{a}b{b}g{g}h{0}".replace('.','_')+".csv"
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
        self.number_of_airUT = SimData['Number of air users']
        flight_height = SimData['Allowed flight range'][0]
        self.air_users_locations = self.city.getGPositions(height = flight_height)
        indices = np.random.choice(range(len(self.air_users_locations)), self.number_of_airUT)
        positions = [self.air_users_locations[i] for i in indices]
        # print(f'Number of air users: = {len(positions)}')
        # print(f'positions: = {positions}')
        a,b,g = self.getEnvParam()
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
            nav_mesh_file = f"navmesh_a{a}b{b}g{g}h{flight_height}".replace('.','_')+".csv"
            nav_mesh_path = OUTPUT_DIR + '/navMesh/' + nav_mesh_file
            au.setNavMesh(nav_mesh_path)
            self.air_users.append(au)
            # add mobility parameters
            # self.air_users[i].reparentTo(render)
            # self.air_users[i].Actor.setPos(positions[i])
            # self.air_users[i].Actor.setZ(20.0)
    
    
    def genAirBS(self):
        # self.number_of_airBS = SimData['Number of air BSs']
        airBS_heights = SimData['Allowed flight range'][1:]
        self.number_of_airBS = len(airBS_heights)
        flight_height = SimData['Allowed flight range'][1]
        self.air_bs_locations = self.city.getGPositions(height = 0)
        indices = np.random.choice(range(len(self.air_bs_locations)), 
                                    self.number_of_airBS)
        positions = [self.air_bs_locations[i] for i in indices]
        # print(f'Number of air bs: = {len(positions)}')
        a,b,g = self.getEnvParam()
        
        self.air_bs = []
        for i in range(len(positions)):
            p = (positions[i][0], positions[i][1], airBS_heights[i])
            param = {
                'ID': i,
                'name': f'UB{i}',
                'type': 'UB',
                # 'position': positions[i],
                'device_type': 'airBS',
            }
            au = UAVBs(**param)
            au.setPos(p)
            nav_mesh_file = f"navmesh_a{a}b{b}g{g}h{airBS_heights[i]}".replace('.','_')+".csv"
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
            
    
    def makeLight(self):
        # First we create an ambient light. All objects are affected by ambient
        # light equally
        # Create and name the ambient light
        self.ambientLight = render.attachNewNode(AmbientLight("ambientLight"))
        # Set the color of the ambient light
        self.ambientLight.node().setColor((.1, .1, .1, 1))
        # add the newly created light to the lightAttrib
        render.setLight(self.ambientLight)
        # Now we create a directional light. Directional lights add shading from a
        # given angle. This is good for far away sources like the sun
        # self.directionalLight = render.attachNewNode(
        #     DirectionalLight("directionalLight"))
        # self.directionalLight.node().setColor((1, 1, 1, 1))
        # # The direction of a directional light is set as a 3D vector
        # self.directionalLight.node().setDirection(LVector3(-1, 0, -90))
        # # These settings are necessary for shadows to work correctly
        # # self.directionalLight.setZ(200)
        # dlens = self.directionalLight.node().getLens()
        # dlens.setFilmSize(41, 21)
        # #dlens.setNearFar(50, 75)
        # dlens.setNearFar(50, 75)
        # #self.directionalLight.node().showFrustum()
        # render.setLight(self.directionalLight)
        
        self.PointLight = render.attachNewNode(
            PointLight("redPointLight"))
        self.PointLight.setPos(500, 0, 200)
        self.PointLight.setScale(2)
        self.PointLight.node().setColor((1, 1, 1, 1))
        # self.PointLight.node().setAttenuation(LVector3(.1, 0.04, 0.0))
        render.setLight(self.PointLight)
        
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
        
    def collectTransforms(self):
        alpha = SimData['Scenarios'][self.gui.scenario]['Alpha'] # type: ignore
        beta = SimData['Scenarios'][self.gui.scenario]['Beta'] # type: ignore
        gamma = SimData['Scenarios'][self.gui.scenario]['Gamma'] # type: ignore
        freq = SimData['Frequency (GHz)'] # type: ignore
        #
        if not hasattr(self, 'metricMgr'):
            self.transMgr = TransMgr(taskchain = 'MetricsChain')
        # set parameters
        self.transMgr.setParameters(
            self.gui.scenario, 
            alpha,
            beta,
            gamma,
            freq,
            channel= 'GA2GA',
            link = 'downlink', 
            filename=None)
        
    def updateTransforms(self):
        alpha = SimData['Scenarios'][self.gui.scenario]['Alpha'] # type: ignore
        beta = SimData['Scenarios'][self.gui.scenario]['Beta'] # type: ignore
        gamma = SimData['Scenarios'][self.gui.scenario]['Gamma'] # type: ignore
        # 
        # set environment
        self.transMgr.updateEnv(
            self.gui.scenario, 
            alpha,
            beta,
            gamma,
            )
        
    def transUpdate(self, task):
        # manual update of scenario 
        # call this task
        # update transform collecting by changing 
        # the scenario manually
        if self.gui.scenario != self.currnt_scenario:
            self.currnt_scenario = self.gui.scenario
            # self.collectTransforms()
            self.updateTransforms()
        return task.again
    
    def transAutoUpdate(self, task):
        # update transform collecting by changing 
        # the scenario automatically
        # # panda3d timing
        # if not hasattr(self, 'sim_start_time'):
        #     self.sim_start_time = globalClock.getFrameTime()
        # # check if two minutes passed change scenario
        # if globalClock.getFrameTime() - self.sim_start_time > 120:
        #     self.sim_start_time = globalClock.getFrameTime()
        print(f'Transform Collecting: iterations: {self.num_iterations+1}')
        if self.gui.scenario == 'Suburban':
            self.gui.setScenario('Urban')
            self.updateTransforms()
        elif self.gui.scenario == 'Urban':
            self.gui.setScenario('Dense Urban')
            self.updateTransforms()
        elif self.gui.scenario == 'Dense Urban':
            self.gui.setScenario('High-rise Urban')
            self.updateTransforms()
        elif self.gui.scenario == 'High-rise Urban':
            self.gui.setScenario('Suburban')
            self.updateTransforms()
            self.num_iterations += 1
        #
        
        #
        if self.num_iterations >= 10:
            self.transMgr.collecting = False
            return task.done
        else:
            return task.again
    
    def genEnvironment(self):
        self.genStreetMap(
                    varStWidth = 'Variant',
                    varPercent = 0.25,
                    height_dist = 'Rayleigh',
                    navMesh_stepSize=20)
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
        self.collectTransforms()
        
        # method: automatically
        # timing of changing environment scenario periodically
        scenario_delay = 20 # seconds
        # self.num_iterations = 0
        self.taskMgr.doMethodLater(scenario_delay,  
                            self.transAutoUpdate, 
                            'SimTiming_task')
        
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
        
    def getEnvParam(self):
        alpha = SimData['Scenarios'][self.gui.scenario]['Alpha'] # type: ignore
        beta = SimData['Scenarios'][self.gui.scenario]['Beta'] # type: ignore
        gamma = SimData['Scenarios'][self.gui.scenario]['Gamma'] # type: ignore
        return (alpha, beta, gamma)
        
    def collectRLP(self):
        # create sequence of tasks to collect metrics
        # tasks are executed one at a time in sequence
        # get scenario from gui
        scenario = self.gui.scenarioDict[self.gui.scenario]
        print(f'SimMgr: simulation {scenario}')
        print(f'Before collecting')
        print(f'SimMgr: tasks: \n{taskMgr.getTasks()}')
        self.metricMgr1 = MetricsManager_old(taskchain = 'MetricsChain')
        
        # create a sequence to change airBS height then
        # collect metrics
        self.metricMgr1.addCollector(
                        scenario, 
                        'RLB', 
                        channel= 'GA2GA',
                        link = 'downlink', 
                        filename=None)
        print(f'after collecting')
        print(f'SimMgr: tasks: \n{taskMgr.getTasks()}')
        
    def Simulate(self):
        # create sequence of tasks to collect metrics
        # tasks are executed one at a time in sequence
        # get scenario from gui
        scenario = self.gui.scenarioDict[self.gui.scenario]
        print(f'SimMgr: simulation {scenario}')
        print(f'Before collecting')
        print(f'SimMgr: tasks: \n{taskMgr.getTasks()}')
        
    def collect_P_LoS(self):
        alpha = SimData['Scenarios'][self.gui.scenario]['Alpha'] # type: ignore
        beta = SimData['Scenarios'][self.gui.scenario]['Beta'] # type: ignore
        gamma = SimData['Scenarios'][self.gui.scenario]['Gamma'] # type: ignore
        freq = SimData['Frequency (GHz)']
        #
        self.metricMgr = TransMgr(taskchain = 'MetricsChain')
        # set parameters
        self.metricMgr.setParameters(
            self.gui.scenario, 
            alpha,
            beta,
            gamma,
            freq,
            channel= 'GA2GA',
            link = 'downlink', 
            filename=None)
            