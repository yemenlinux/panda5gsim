#
import os
import numpy as np
from pathlib import Path
from panda3d.core import (
    NodePath,
    LVecBase3f,
    Filename,
    BitMask32,
)


from direct.task import Task
from direct.actor.Actor import Actor
from direct.showbase.DirectObject import DirectObject

from panda5gSim import ASSETS_DIR, OUTPUT_DIR
from panda5gSim.users.receiver import Device
# module_type = 'network_device'
# module_name = 'UAVuser'
# module_description = 'UAV user module of any network_device'

# working_dir = os.getcwd()


class DroneActor(Actor):
    """ Unmanned Aerial Vehicle (UAV) """
    def __init__(self):
        #
        shell_model = Filename(ASSETS_DIR + "/models/uav/quad-shell.glb")
        #Default propeller models
        # if propellers is None:
        prop_model_cw = Filename(ASSETS_DIR + "/models/uav/propeller.glb")
        prop_model_ccw = prop_model_cw #TODO: Temporary, create a filpped model

        propellers = {
            "PropellerJoint1": prop_model_ccw,
            "PropellerJoint2": prop_model_ccw,
            "PropellerJoint3": prop_model_cw,
            "PropellerJoint4": prop_model_cw
        }
        # if propeller_spin is None:
        propeller_spin = {
            "PropellerJoint1": -1,
            "PropellerJoint2": -1,
            "PropellerJoint3": 1,
            "PropellerJoint4": 1
        }
        #
        #Prefix so that it doesn't clash with original bone node
        propeller_parts = {'p_%s'%k:v for k,v in propellers.items()}
        self.joints = {'propellers': {}}
        # init super
        super().__init__( {
            'modelRoot': shell_model,
            **propeller_parts
        }, anims={'modelRoot': {}}) #To use the multipart w/o LOD loader (this is the way to do it)
        for bone in propellers.keys():
            #Make node accessible
            self.exposeJoint(None, 'modelRoot', bone)
            self.attach('p_%s'%bone, "modelRoot", bone)
            control_node = self.controlJoint(None, 'modelRoot', bone)
            #
            self.joints['propellers'][bone] = {
                'bone': control_node,
                'spinDir': propeller_spin[bone]
            }
            #Random rotation
            control_node.setH(np.random.randint(0,360))
        
        # set tags
        # self.setTag('type', 'air_actor')
        
        # set collide mask
        self.setCollideMask(BitMask32.bit(0))
        
        
        # 
        self.spin_propeller = False
        self.animation_on = False
        # self.addTask(self.propellers_run,  'propellers_run')
        taskMgr.add(self.propellers_run,  'propellers_run') # type: ignore
        
    def propellers_run(self, task):
        # if self.getZ(render) > 0 or self.spin_propeller == True: # type: ignore
        if self.spin_propeller == True: # type: ignore
            prop_vel = 100 #TODO: Get from physics
            for bone in self.joints['propellers'].values():
                #Rotate with respect to spin direction and thrust
                bone['bone'].setH(bone['bone'].getH() + prop_vel*bone['spinDir'])
        return Task.cont
    
    def destroy(self):
        taskMgr.remove('propellers_run')
        self.cleanup()
        self.removeNode()
        # self.joints = None
        # self.spin_propeller = None
        # self.animation_on = None
        
        

class AirActor(DirectObject):
    def __init__(self, **kwargs):
        DirectObject.__init__(self)
        #
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.type = "UU"
        #
        self.Actor = DroneActor()
        self.Actor.setName(self.name)
        #
        self.Actor.reparentTo(render)
        self.Actor.setScale(0.31)
        # self.Actor.setPos(self.position)
        # set tags
        self.Actor.setTag('type', 'air_actor')
        self.Actor.setTag('actor', 'air_actor')
        self.Actor.setPythonTag("subclass", self)
        
        # # add device UAV
        # if not hasattr(self, 'device_type'):
        #     self.device_type = 'airUT'
        # self.device = Device(self.device_type, mimo = (1,1))
        # self.device.reparentTo(self.Actor)
        
        # 
        self.moveStatus = 'init'
        self.mobility_on = False
        # self.nav_mesh_filename = os.path.abspath(os.path.join(working_dir, 'output', self.env.ground_nav_mesh))
        self.ai_type = 'pathfollow'
        self.target_PoIs = []
        self.target_stay_time = []
        self.target_index = 0
        self.mobility_mode = 'random'
        # accept messages
        self.accept(f'{self.name}_move', self.move)
        
        # self.accept('move_ON', self.toggle_mobility)
        # self.accept('move_OFF', self.toggle_mobility)
        # # tasks
        # taskMgr.add(self.moveUAV, 'moveUAV')
        
    def setStayTime(self, weighted_list):
        self.target_stay_time = weighted_list
        
    def setMobilityMode(self, mode):
        self.mobility_mode = mode
        
    def setPoIs(self, poi_list):
        self.target_PoIs = []
        for t in poi_list:
            v = LVecBase3f(t[0], t[1], t[2])
            self.target_PoIs.append(v)
        
    async def moveUAV(self, task):
        elapsed = globalClock.getDt()
        if self.Actor.getZ(render) == 0:
            self.Actor.spin_propeller = False
        else:
            self.Actor.spin_propeller = True
        #
        if self.mobility_on:
            if self.moveStatus == 'init':
                # ai_type = self.ai_type
                current_idx = self.target_index
                print(f'uav {self.name} PoIs: {len(self.target_PoIs)}, {self.target_PoIs}')
                # target = self.target_PoIs[current_idx]
                # pois_list = self.target_PoIs
                wait_time = self.target_stay_time[current_idx] * elapsed
                #
                # self.moveTask = taskMgr.add(self.setMove(ai_type, target, pois_list, wait_time))
                self.setMove(self.ai_type,
                            self.target_PoIs[current_idx])
            #
            self.moveStatus = self.AIbehaviors.behaviorStatus(self.ai_type)
            if self.moveStatus == 'done':
                
                current_idx = self.target_index
                wait_time = self.target_stay_time[current_idx] * elapsed
                # print(f'{self.name} second task, wait_time: {wait_time}')
                await Task.pause(wait_time)
                
                self.target_index += 1
                if self.target_index >= len(self.target_PoIs):
                    self.target_index = 0
                #
                # ai_type = self.ai_type
                # current_idx = self.target_index
                # target = self.target_PoIs[current_idx]
                # pois_list = self.target_PoIs
                # 
                self.setMove(
                    self.ai_type,
                    self.target_PoIs[current_idx])
            else:
                await Task.pause(1)
        #
        return Task.cont
    
    def setMove(self, ai_type, target):
        if ai_type == 'pathfollow':
            self.AIbehaviors.removeAi('all')
            if not self.load_nav_mesh:
                self.AIbehaviors.initPathFind(self.nav_mesh_filename)
                self.load_nav_mesh = True
            # 
            self.AIbehaviors.pathFindTo(target, "addPath")
            self.Actor.spin_propeller = True
            
        if ai_type == 'seek':
            self.AIbehaviors.removeAi('all')
            self.AIbehaviors.seek(target)
            self.AIbehaviors.obstacleAvoidance(1.0)
            self.Actor.spin_propeller = True
        
    def move(self, target):
        if not self.load_nav_mesh:
            self.AIbehaviors.initPathFind(self.nav_mesh_filename)
            self.load_nav_mesh = True
        #
        # self.AIbehaviors.removeAi('all')
        self.AIbehaviors.pathFindTo(target, "addPath")
        self.Actor.spin_propeller = True
        # if self.type == 'UB':
        #     print(f'{self.name} move to {target}')
        
    def init_nav_mesh(self):
        if not self.load_nav_mesh:
            self.AIbehaviors.initPathFind(self.nav_mesh_filename)
            self.load_nav_mesh = True
        
    def toggle_mobility(self, user_id = 'all'):
        if user_id == 'all' or user_id == self.ID:
            self.mobility_on = not self.mobility_on
            if self.mobility_on:
                # self.Actor.spin_propeller = True
                self.Actor.animation_on = True
                self.moveStatus = 'init'
            else:
                # self.Actor.spin_propeller = False
                self.Actor.animation_on = False
                self.moveStatus = 'done'
        
    def set_id(self, ID):
        self.ID = ID
        self.name = f'{self.type}{self.ID}'
        # self.accept(f'{self.name}_seek', self.seek)
        
    def message_handler(self, command):
        if command == 'setPfGuide':
            # setPfGuide
            self.AIchar.setPfGuide(True)
            
    def setNavMesh(self, path):
        self.nav_mesh_filename = path
        self.load_nav_mesh = False
        # print(self.nav_mesh_filename)
        
    def setPos(self, pos):
        if len(pos) == 2:
            self.position = LVecBase3f(pos[0], pos[1], 0)
        else:
            self.position = LVecBase3f(pos[0], pos[1], pos[2])
        self.Actor.setPos(self.position)
        
    def updatePos(self):
        self.position = self.Actor.getPos()
        
    def getPos(self):
        self.updatePos()
        return self.position
        
    def destroy(self):
        # remove AI behaviors
        self.AIbehaviors.removeAi('all')
        self.ignoreAll()
        self.removeAllTasks()
        # self.Actor.cleanup()
        self.Actor.destroy()
        del self

    def resetToLastPoS(self):
        if hasattr(self, 'AIbehaviors'):
            self.AIbehaviors.removeAi('all')
        # if lastPos is NodePath get positions
        if isinstance(self.target_PoIs[-1], NodePath):
            lastPos = self.target_PoIs[-1].getPos()
        else:
            lastPos = self.target_PoIs[-1]
        self.Actor.setPos(lastPos)
        self.target_index = 0
        self.moveStatus = 'init'
        
    def setAltitude(self, altitude):
        print(f'UAVBs.setAltitude({altitude}), Pos: {self.Actor.getPos()}')
        if hasattr(self, 'AIbehaviors'):
            self.AIbehaviors.removeAi('all')
        self.mobility_on = False
        self.moveStatus = 'init'
        for i in range(len(self.target_PoIs)):
            self.target_PoIs[i].setZ(altitude)
            # if self.target_index == i:
            #     if isinstance(self.target_PoIs[i], NodePath):
            #         self.Actor.setPos(self.target_PoIs[i].getPos())
            #     else:
            #         self.Actor.setPos(self.target_PoIs[i])
            
        # self.resetToLastPoS()
        self.Actor.setZ(altitude)
        self.setNavMesh(OUTPUT_DIR + f'/navmesh{altitude}.csv')
        self.mobility_on = True
        print(f'UAVBs.setAltitude({altitude}), Pos: {self.Actor.getPos()}')
        


class UAVuser(AirActor):
    def __init__(self, **kwargs):
        AirActor.__init__(self, **kwargs)
        self.type = "UU"
        self.Actor.setTag('actor', 'airUT_actor')
        # add device UAV
        if not hasattr(self, 'device_type'):
            self.device_type = 'airUT'
        self.device = Device(self.device_type, mimo = (1,1))
        self.device.reparentTo(self.Actor)
        
        self.accept(f'{self.type}_setAltitude', self.setAltitude)
        
    def destroy(self):
        self.device.removeNode()
        del self.device
        AirActor.destroy(self)
        del self