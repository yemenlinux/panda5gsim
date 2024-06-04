# Base station (eNB) class

import numpy as np
from panda3d.core import (
    LRotation,
    NodePath,
    LVecBase3f,
    Filename,
    BitMask32,
)


from direct.task import Task
# from direct.actor.Actor import Actor
from direct.showbase.DirectObject import DirectObject

from panda5gSim import ASSETS_DIR, OUTPUT_DIR
from panda5gSim.users.receiver import Device


# OMNI_cell = loader.loadModel(ASSETS_DIR+"/models/Top_Antenna.bam")
# Directional_cell = loader.loadModel(ASSETS_DIR+"/models/Top_Antenna.bam")

class GBS(DirectObject):
    def __init__(self, name):
        DirectObject.__init__(self)
        self.tower = loader.loadModel(ASSETS_DIR+"/models/Tower.bam")
        self.tower.reparentTo(render)
        self.tower.name = name
        self.cells = []
        self.device_type = 'gBS'
        
    def setPos(self, pos):
        pos = LVecBase3f(pos[0], 
                        pos[1], 
                        pos[2])
        self.tower.setPos(pos)
        self.tower.setHpr(0,0,0)
        
    def setTowerHeight(self, height):
        width = 2.0
        depth = 2.0
        bounding_box = self.tower.getTightBounds()
        # model_height = self.tower.getTightBounds()[1][2]
        x_scale = width / (bounding_box[1][0] - bounding_box[0][0])
        y_scale = depth / (bounding_box[1][1] - bounding_box[0][1])
        z_scale = height / (bounding_box[1][2] - bounding_box[0][2])
        self.tower.setScale(x_scale, y_scale, z_scale)
        
        
    def addCell(self, cell_type, num_of_sectors = 3):
        if cell_type == 'omni':
            device = Device(self.device_type, mimo = (1,1))
            device.reparentTo(self.tower)
            device.setZ(self.tower.getTightBounds()[1][2])
            self.cells.append(device)
        elif cell_type == 'directional':
            origin = self.tower.getPos().getXy()
            point = (self.getTowerRadius(), 0)
            for i in range(num_of_sectors):
                h_angle = - i * (360 / num_of_sectors)
                pos_angle = 90 + h_angle 
                x, y = self.rotate(origin, point, np.deg2rad(pos_angle))
                device = Device(self.device_type, mimo = (1,1))
                device.reparentTo(self.tower)
                device.setPos(LVecBase3f(x, y, self.tower.getTightBounds()[1][2]))
                tilt = 0
                quad = LRotation((1, 0, 0), tilt) * LRotation((0, 0, 1), h_angle)
                device.setQuat(quad)
                #
                self.cells.append(device)
                
        
    def getTowerRadius(self):
        b = self.tower.getTightBounds()[1] - self.tower.getTightBounds()[0]
        diameter = b.getXy().length()
        return diameter / 2
        
    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy
    
    def destroy(self):
        self.tower.removeNode()