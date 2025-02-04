import os
import sys
import numpy as np
np.random.seed(123)
# add panda5gSim to path
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
sys.path.append(PARENT_DIR)
# panda5gSim modules
from panda3d.core import (
    NodePath,
    LVecBase3f,
    LMatrix4f,
    LVector3f,
    Filename,
    BitMask32,
    Spotlight,
    LVector3, 
    Lens, 
    VBase4, 
    PointLight,
    PerspectiveLens,
    AmbientLight,
    DirectionalLight,
    TextureStage,
    Texture,
    #
    LineSegs,
    CardMaker,
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
        
    def create_coverage_map1(self):
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
                # self.antenna_phi_3dB = 45
                # self.antenna_theta_3dB = 65
                phi_coverage = int(altitude * np.tan(np.deg2rad(self.antenna_phi_3dB)))
                theta_coverage = int(altitude * np.tan(np.deg2rad(self.antenna_theta_3dB)))
                # 
                axesLength = (phi_coverage, 
                              theta_coverage)
                # directional coverage
                x_offset = (altitude / np.cos(np.deg2rad(90 - self.antenna_theta_3dB))
                            * np.sin(np.deg2rad(90 - self.antenna_theta_3dB)) ) # Forward/backward offset
                # 
                step_size = -(x_offset + theta_coverage)
                center_coordinates = calculate_next_position(nodepath, step_size)
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
        # update the terrain texture
        self.update_terrain_texture(self.coverage_array)
        
    
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
                self.antenna_phi_3dB = 45
                # self.antenna_theta_3dB = 65
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
                num_gUEs = 2,
                metric_filename = f'test.csv',
                airBS_heights = [50],
                num_rec_collect = 400,
                point_actors = True,
                # output_transform_filename= f'transforms_{building_dist}.csv'
                )

# camera
# base.cam.setPos(300, -500, 150)
# base.cam.lookAt(300, -400, 15)

# print(f'camera Fov: {base.cam.node().getLens().getFov()}')
# get the camera lens and expand its frustum
lens = base.cam.node().getLens()
lens.setFov(90,30)
# lens.setNearFar(1, 1000)

# app.environments = [[0.17, 105, 28]]
app.environments = [
    # [0.1, 750, 8], 
    [0.3, 500, 15], 
    # [0.5, 300, 20],
    # [0.3, 300, 50],
    # [0.51, 198, 15],
    ]
app.Simulate1()
# base.cam.lookAt(app.air_bs[0].Actor)

app.run()