
import numpy as np
# from pathlib import Path
from panda3d.core import (
    Vec3,
    LVecBase3f,
    Filename,
    CollisionTraverser, 
    CollisionNode,
    CollisionHandlerQueue, 
    CollisionRay,
    CollisionHandlerPusher, 
    CollisionSphere,
    CollideMask, 
    BitMask32,
)


from direct.task import Task
from direct.showbase.DirectObject import DirectObject

from panda5gSim import ASSETS_DIR, OUTPUT_DIR
from panda5gSim.users.receiver import Device
from panda5gSim.users.uav import AirActor




class UAVBs(AirActor):
    def __init__(self, **kwargs):
        AirActor.__init__(self, **kwargs)
        self.type = "UB"
        self.Actor.setTag('actor', 'airBS_actor')
        self.Actor.setColor(1,0,0,1)
        # add device UAV
        if not hasattr(self, 'device_type'):
            self.device_type = 'airBS'
        self.device = Device(self.device_type, mimo = (1,1))
        self.device.reparentTo(self.Actor)
        #
        self.accept(f'{self.type}_setAltitude', self.setAltitude)
        
    def destroy(self):
        self.device.removeNode()
        del self.device
        AirActor.destroy(self)
        del self
        
            