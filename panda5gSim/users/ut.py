import numpy as np
from direct.actor.Actor import Actor
import os
from pathlib import Path
from panda3d.core import (
    Vec3,
    LVecBase3f,
    CollisionTraverser, 
    CollisionNode,
    CollisionHandlerQueue, 
    CollisionRay,
    CollisionHandlerPusher, 
    CollisionSphere,
    CollideMask,
    BitMask32
)
from direct.showbase.DirectObject import DirectObject
from panda3d.ai import *
from direct.task import Task
from panda5gSim import ASSETS_DIR, OUTPUT_DIR
from panda5gSim.users.receiver import Device

Ralph_model = ASSETS_DIR + '/models/ralph/ralph'
Ralph_animations = {
    "run": ASSETS_DIR + "/models/ralph/ralph-run",
    "walk": ASSETS_DIR + "/models/ralph/ralph-walk"
}

# GroundUser
class GroundUser(DirectObject):
    def __init__(self, **kwargs):
        DirectObject.__init__(self)
        #
        for key, value in kwargs.items():
            setattr(self, key, value)
        #
        self.Actor = Actor(Ralph_model, Ralph_animations)
        self.Actor.setName(self.name)
        #
        self.Actor.reparentTo(render) # type: ignore
        self.Actor.setScale(0.31)
        # self.Actor.setPos(self.position)
        # set tags
        self.Actor.setTag('type', 'ground_actor')
        self.Actor.setTag('actor', 'ground_actor')
        self.Actor.setPythonTag("subclass", self)
        
        # Now we will expose the joint the hand joint. 
        # ExposeJoint allows us to get the position of a joint 
        # while it is animating. This is different than
        # controlJonit which stops that joint from animating 
        # but lets us move it.
        # This is particularly useful for putting an object 
        # (like a cell phone) in an actor's hand
        self.rightHand = self.Actor.exposeJoint(None, 'modelRoot', 'RightHand')
        # add device to the hand
        if not hasattr(self, 'device_type'):
            self.device_type = 'gUT'
        self.device = Device(self.device_type, mimo = (1,1))
        self.device.reparentTo(self.rightHand)
        # set height of device to 1.5 m above ground
        self.device.setZ(render, 1.5)
        
        # accept messages
        self.moveStatus = 'init'
        self.mobility_on = False
        # self.aiParent = None
        self.animation = 'run'
        #
        self.ai_type = 'pathfollow'
        self.target_PoIs = []
        self.target_stay_time = []
        self.target_index = 0
        self.mobility_mode = 'random'
        # self.accept(f'{self.name}_seek', self.seek)
        # self.accept('move_ON/OFF', self.toggle_mobility)
        # use: messenger.send('move_ON/OFF', [self.ID])
        # or messenger.send('move_ON/OFF', ['all'])
        
        self.accept(f'{self.name}_move', self.move)
        
        # set collide mask
        self.Actor.setCollideMask(BitMask32.bit(0))
        
        # tasks
        # uncomment to use task
        # self.accept('move_ON', self.toggle_mobility)
        # self.accept('move_OFF', self.toggle_mobility)
        # taskMgr.add(self.moveUE, 'moveUE') # type: ignore
        # taskMgr.add(self.update_position, 'update_position')
        # move task
        #taskMgr.add(self.move, 'move')
        # taskMgr.doMethodLater(2, self.move, 'move')
        
    def setStayTime(self, weighted_list):
        self.target_stay_time = weighted_list
        
    def setMobilityMode(self, mode):
        self.mobility_mode = mode
        
    def setPoIs(self, poi_list):
        self.target_PoIs = []
        for t in poi_list:
            v = LVecBase3f(t[0], t[1], 0)
            self.target_PoIs.append(v)
        
    async def moveUE(self, task):
        elapsed = globalClock.getDt() # type: ignore
        if self.mobility_on:
            if self.moveStatus == 'init':
                ai_type = self.ai_type
                current_idx = self.target_index
                # print(f'indx: {current_idx}, pois {len(self.target_PoIs)}, {self.target_PoIs}')
                # print(f'self.target_stay_time: {len(self.target_stay_time)}, {self.target_stay_time}')
                target = self.target_PoIs[current_idx]
                # pois_list = self.target_PoIs
                wait_time = self.target_stay_time[current_idx] * elapsed
                #
                # self.moveTask = taskMgr.add(self.setMove(ai_type, target, pois_list, wait_time))
                self.setMove(ai_type, target, self.target_PoIs)
            #
            self.moveStatus = self.AIbehaviors.behaviorStatus(self.ai_type)
            if self.moveStatus == 'done':
                self.Actor.stop(self.animation)
                current_idx = self.target_index
                wait_time = self.target_stay_time[current_idx] * elapsed
                # print(f'{self.name} second task, wait_time: {wait_time}')
                await Task.pause(wait_time)
                
                if self.mobility_mode == 'Random':
                    self.target_index = np.random.randint(0, len(self.target_PoIs))
                else:
                    self.target_index += 1
                    if self.target_index >= len(self.target_PoIs):
                        self.target_index = 0
                #
                ai_type = self.ai_type
                current_idx = self.target_index
                target = self.target_PoIs[current_idx]
                pois_list = self.target_PoIs
                # wait_time = self.target_stay_time[current_idx] * elapsed
                #
                # self.moveTask = taskMgr.add(self.setMove(ai_type, target, pois_list, wait_time))
                self.setMove(ai_type, target, pois_list)
            else:
                await Task.pause(1)
        #
        return Task.cont
    
    def setMove(self, ai_type, target, pois_list):
        if ai_type == 'pathfollow':
            self.AIbehaviors.removeAi('all')
            if not self.load_nav_mesh:
                self.AIbehaviors.initPathFind(self.nav_mesh_filename)
                self.load_nav_mesh = True
            #
            self.AIbehaviors.pathFindTo(target, "addPath")
            self.Actor.loop(self.animation)
            
        if ai_type == 'seek':
            self.AIbehaviors.removeAi('all')
            self.AIbehaviors.seek(target)
            self.AIbehaviors.obstacleAvoidance(1.0)
            self.Actor.loop(self.animation)
    
    def move(self, target):
        if not self.load_nav_mesh:
            self.AIbehaviors.initPathFind(self.nav_mesh_filename)
            self.load_nav_mesh = True
        #
        self.AIbehaviors.pathFindTo(target, "addPath")
        self.Actor.loop(self.animation)
        
    def toggle_mobility(self, user_id = 'all'):
        if user_id == 'all' or user_id == self.ID:
            self.mobility_on = not self.mobility_on
    
    def init_nav_mesh(self):
        if not self.load_nav_mesh:
            self.AIbehaviors.initPathFind(self.nav_mesh_filename)
            self.load_nav_mesh = True
    def destroy(self):
        self.AIbehaviors.removeAi('all')
        self.ignoreAll()
        self.removeAllTasks()
        self.device.removeNode()
        del self.device
        self.Actor.cleanup()
        self.Actor.removeNode()
        del self
        
    def setNavMesh(self, path):
        self.nav_mesh_filename = path
        self.load_nav_mesh = False
        
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
    
    

