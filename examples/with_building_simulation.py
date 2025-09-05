import os
import sys
import numpy as np
np.random.seed(124)
# add panda5gSim to path
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
sys.path.append(PARENT_DIR)
# panda5gSim modules
from panda3d.core import (
    # NodePath,
    LVecBase3f,
    # LMatrix4f,
    # LVector3f,
    # Filename,
    # BitMask32,
    # Spotlight,
    # LVector3, 
    # Lens, 
    # VBase4, 
    # PointLight,
    # PerspectiveLens,
    # AmbientLight,
    # DirectionalLight,
    # TextureStage,
    Texture,
    #
    # LineSegs,
    # CardMaker,
)
# import opencv to draw circles and lines on numpy arrays
import cv2

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
# from direct.actor.Actor import Actor
# from direct.showbase.DirectObject import DirectObject

from panda5gSim import ASSETS_DIR, OUTPUT_DIR
# from panda5gSim.users.receiver import Device, PointNode

from panda5gSim.core.streets import UrbanCityMap
from panda5gSim.core.buildings import Building
from panda5gSim.users.uav import PointActor
from panda5gSim.users.air_bs import UAVBs
from panda5gSim.mobility.autopilot import AutoPilotMove
from panda5gSim.simmgrs.simmgr import SimManager
from panda5gSim.core.scene_graph import findNPbyTag
from panda5gSim.core.mobility_mgr import MobilityMgr
from panda5gSim.metrics.trans_updator import TransformReader
from panda5gSim.camcontrols.cameraControl import CameraMgr

def calculate_next_position(actor, step_size):
    """Calculates the next position of an actor based on its current 
    position, HPR, and a step size.

    Args:
        actor: The Panda3D actor.
        step_size: The distance to move in the actor's forward direction.

    Returns:
        A Panda3D Vec3 representing the next position.  Returns None if
        the actor doesn't have a valid transform.
    """

    if not actor or not actor.get_transform():  # Check for valid actor/transform
        return None
    #
    direction = actor.getQuat().getForward()
    current_pos = actor.getPos()
    next_pos = current_pos + direction * step_size
    return next_pos



class Sim(SimManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 
        self.omni_angle = 80
        self.omnidirectional_coverage_color = (114, 220, 200, 0.5)
        self.directional_coverage_color = (0, 220, 0, 0.5)
        
        # self.accept('r', self.create_coverage_map)
        self.taskMgr.add(self.update_coverage_map,'update_coverage_map', delay=0.1)
        
        
    def update_terrain_texture(self, numpy_image):
        shape = numpy_image.shape
        tex = Texture()
        tex.setCompression(Texture.CM_off)
        tex.setup2dTexture(shape[0], shape[1], Texture.T_unsigned_byte, Texture.F_rgb8)
        tex.setRamImage(numpy_image)
        tex.setRamImage(numpy_image)
        # tex = loader.loadTexture(street_map)
        self.terrain.setTexture(tex)
        
    
    
    def create_coverage_map(self):
        # create image of terrain size
        thickness = -1
        size = self.city.getAreaXY()
        self.coverage_array = np.zeros(shape=[size[0], size[1], 3], 
                                       dtype=np.uint8)
        self.coverage_array.fill(255)
        # find airBSs in scene graph
        for nodepath in render.findAllMatches('**'): # type: ignore
            if nodepath.getTag('type') == 'air_actor':
                
                pos = nodepath.getPos()
                hpr = nodepath.getHpr()
                x = int(pos.getX())
                y = int(pos.getY())
                altitude = pos.getZ()
                # convert the possition to the image coordinates
                x = x + size[0] // 2
                y = y + size[1] // 2
                # draw a circle around the airBS and fill it with color
                coverage_radius = int(altitude * np.tan(np.deg2rad(self.omni_angle)) 
                                      * np.sin(np.deg2rad(self.omni_angle)))
                
                cv2.circle(self.coverage_array, (x, y), 
                           coverage_radius, 
                           self.omnidirectional_coverage_color, thickness)
        #####
        for nodepath in render.findAllMatches('**'): # type: ignore
            if nodepath.getTag('type') == 'air_actor':
                pos = nodepath.getPos()
                hpr = nodepath.getHpr()
                x = int(pos.getX())
                y = int(pos.getY())
                altitude = pos.getZ()
                # 
                
                phi_coverage = int(
                    altitude * np.tan(np.deg2rad(self.antenna_phi_3dB))
                    )
                theta_coverage = int(
                    altitude * np.tan(np.deg2rad(self.antenna_theta_3dB))
                    # + altitude * np.tan(np.deg2rad(self.antenna_theta_3dB/2))
                    )
                # 
                axesLength = (phi_coverage, 
                              theta_coverage)
                # # directional coverage
                # x_offset = (altitude / np.cos(np.deg2rad(90 - self.antenna_theta_3dB))
                #             * np.sin(np.deg2rad(90 - self.antenna_theta_3dB)) ) # Forward/backward offset
                # # 
                # step_size = -(x_offset + theta_coverage)
                # center_coordinates = calculate_next_position(nodepath, step_size)
                # center_coordinates = (int(center_coordinates.getX() + size[0] // 2),
                #                       int(center_coordinates.getY() + size[1] // 2)
                #                       )
                
                # calculate the center of the antenna's main lobe 
                # projection on ground
                offset_angle = np.deg2rad(90 
                                        - self.antenna_theta_3dB/2 
                                        - self.antenna_down_tilt
                                        - nodepath.getP())
                if offset_angle != 0:
                    x_offset = altitude / np.tan(offset_angle)
                else:
                    x_offset = 0
                # calculate the center of coverage ellipse
                c_coverage = x_offset + theta_coverage /2
                heading_angle = np.deg2rad(nodepath.getH() - 90)
                heading_vector = LVecBase3f(np.cos(heading_angle), 
                                            np.sin(heading_angle), 0)
                center_coordinates = nodepath.getPos() + heading_vector * c_coverage
                center_coordinates = (int(center_coordinates.getX() + size[0] // 2),
                                    int(center_coordinates.getY() + size[1] // 2)
                                    )
                
                angle = hpr[0]
                startAngle = 0
                endAngle = 360
                
                cv2.ellipse(self.coverage_array, 
                            center_coordinates, 
                            axesLength, 
                            angle, 
                            startAngle, endAngle, 
                            self.directional_coverage_color, 
                            thickness)
                # test
                # print(f'direction: {nodepath.getQuat().getForward()}, hpr: {nodepath.getHpr()}')
                # print(f'pos: {nodepath.getPos()}, c: {center_coordinates}, offset: {x_offset}, theta: {theta_coverage} ')
                # nnp = LVector3(center_coordinates[0], center_coordinates[1], 0)
                # print(f'diff: {nnp - nodepath.getPos()}')
        # update the terrain texture
        self.update_terrain_texture(self.coverage_array)
    
    def update_coverage_map(self, task):
        self.create_coverage_map()
        return Task.again
    
    # override this method to generate the environment
    # without buildings
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
        self.buildings = []
        #
        self.genGroundUsers()
        # self.genAirUsers()
        self.air_users = []
        self.genAirBS()
        # self.genGroundBS()
        # Generate Ai
        self.setAI()
        # self.makeLight()
        
        # Init Mobility manager
        # if self.enable_mobility_manager:
        positions = self.city.getGPositions(height = 0)
        # positions = self.get_positions(0)
        np.random.shuffle(positions)
        self.mobMgr = MobilityMgr(positions)
        #
        # base.cam.setPos(0, -1000, 100) # type: ignore
        # base.cam.lookAt(0, -32, 0) # type: ignore
        self.camMgr = CameraMgr(base.cam) # type: ignore
        # Init Transform reader
        # collect transforms ot Tx and Rx
        self.TrasReader = TransformReader(
                ['gBS_Tx', 'airBS_Tx'],
                ['gUT_Rx', 'airUT_Rx'])
        # 
        
        self.taskMgr.doMethodLater(self.collect_interval,  
                            self.collectMetrics, 
                            'Collect_task')
        
    
    def Simulate1(self):
        # load next scenario
        self.load_next_scenario()
        #
        # self.air_bs[0].Actor.setPos(300, -300, self.air_bs[0].Actor.getZ())
        # self.air_bs[1].Actor.setPos(-115, -100, self.air_bs[1].Actor.getZ())
        # 
        # self.air_bs[0].Actor.setHpr(0, 0, 0)
        # self.air_bs[1].Actor.setHpr(0, 0, 0)
        # 
        # self.ground_users[0].Actor.setPos(-115, -300, 0)
        # self.ground_users[1].Actor.setPos(100, -300, 0)
        

app = Sim(#building_distribution=building_dist,
                # alpha = 0.5,
                # beta = 300,
                # gamma = 50,
                num_gUEs = 10,
                metric_filename = f'test.csv',
                airBS_heights = [100, 100],
                num_rec_collect = 120,
                point_actors = True,
                # output_transform_filename= f'transforms_{building_dist}.csv'
                )

# camera
# base.cam.setPos(300, -500, 150)
# base.cam.lookAt(300, -400, 15)

# print(f'camera Fov: {base.cam.node().getLens().getFov()}')
# get the camera lens and expand its frustum
lens = base.cam.node().getLens()
lens.setFov(70,30)
# lens.setNearFar(1, 1000)

# app.environments = [[0.17, 105, 28]]
# app.environments = [
#     [0.1, 750, 8], 
#     [0.3, 500, 15], 
#     [0.5, 300, 20],
#     [0.3, 300, 50],
#     # [0.51, 198, 15],
#     ]

# random list of environments
# app.environments = [[0.7, 750, 50], [0.7, 750, 36], [0.7, 750, 22], [0.7, 750, 8], [0.7, 533, 50], [0.7, 533, 36], 
# [0.7, 533, 22], [0.7, 533, 8], [0.7, 316, 50], [0.7, 316, 36], [0.7, 316, 22], [0.7, 316, 8],
# [0.7, 100, 50], [0.7, 100, 36], [0.7, 100, 22], [0.7, 100, 8], [0.5, 750, 50], [0.5, 750, 36], 
# [0.5, 750, 22], [0.5, 750, 8], [0.5, 533, 50], [0.5, 533, 36], [0.5, 533, 22], [0.5, 533, 8], 
# [0.5, 316, 50], [0.5, 316, 36], [0.5, 316, 22], [0.5, 316, 8], [0.5, 100, 50], [0.5, 100, 36],
# [0.5, 100, 22], [0.5, 100, 8], [0.3, 750, 50], [0.3, 750, 36], [0.3, 750, 22], [0.3, 750, 8],
# [0.3, 533, 50], [0.3, 533, 36], [0.3, 533, 22], [0.3, 533, 8], [0.3, 316, 50], [0.3, 316, 36],
# [0.3, 316, 22], [0.3, 316, 8], [0.3, 100, 50], [0.3, 100, 36], [0.3, 100, 22], [0.3, 100, 8],
# [0.1, 750, 50], [0.1, 750, 36], [0.1, 750, 22], [0.1, 750, 8], [0.1, 533, 50], [0.1, 533, 36],
# [0.1, 533, 22], [0.1, 533, 8], [0.1, 316, 50], [0.1, 316, 36], [0.1, 316, 22], [0.1, 316, 8],
# [0.1, 100, 50], [0.1, 100, 36], [0.1, 100, 22], [0.1, 100, 8], [0.3, 500, 15], [0.5, 300, 20],
# [0.5, 300, 50]]

# balanced list of environments
app.environments = [[0.41, 100, 36], [0.51, 363, 17], [0.65, 633, 12], [0.31, 692, 18], [0.76, 329, 22],
[0.74, 702, 18], [0.72, 319, 31], [0.75, 217, 44], [0.47, 579, 26], [0.59, 656, 30],
[0.59, 392, 44], [0.79, 675, 47], [0.49, 621, 40], [0.43, 676, 39], [0.33, 511, 37],
[0.35, 634, 43], [0.29, 479, 42], [0.27, 743, 34], [0.25, 661, 23], [0.24, 646, 30],
[0.19, 735, 47], [0.21, 398, 43], [0.18, 586, 41], [0.15, 692, 43], [0.2, 422, 36],
[0.16, 721, 33], [0.16, 293, 49], [0.11, 460, 47], [0.13, 477, 40], [0.2, 682, 24],
[0.15, 447, 35], [0.11, 499, 37], [0.1, 566, 34], [0.13, 266, 44], [0.15, 428, 32],
[0.18, 273, 36], [0.14, 399, 31], [0.19, 283, 33], [0.16, 312, 32], [0.16, 668, 21],
[0.15, 710, 20], [0.1, 634, 20], [0.15, 526, 21], [0.12, 201, 31], [0.16, 114, 40],
[0.12, 625, 14], [0.17, 217, 26], [0.13, 232, 16], [0.17, 105, 28], [0.19, 149, 23],
[0.21, 673, 12], [0.24, 359, 23], [0.23, 129, 24], [0.24, 649, 10], [0.25, 525, 19],
[0.26, 588, 10], [0.28, 308, 8], [0.28, 387, 12], [0.29, 506, 11], [0.3, 274, 15],
[0.27, 236, 27], [0.32, 695, 9], [0.32, 530, 12], [0.33, 236, 18], [0.3, 168, 29],
[0.38, 287, 13], [0.34, 713, 12], [0.43, 494, 9], [0.52, 207, 8], [0.51, 473, 8],
[0.51, 198, 15], [0.56, 532, 9]]



app.Simulate()
# base.cam.lookAt(app.air_bs[0].Actor)

app.run()