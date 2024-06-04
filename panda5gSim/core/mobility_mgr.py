# Mobility manger class coordinates the movements of al actors in the simulation.
# MobilityMgr is subclass of DirectObject.

import numpy as np
from direct.showbase.DirectObject import DirectObject
from panda3d.core import (
    LVecBase3
)
from direct.task import Task

class MobilityMgr(DirectObject):
    def __init__(self, positions):
        DirectObject.__init__(self)
        
        self.positions = positions
        #
        self.status_types = ['disabled', 'done', 'paused']
        #
        # 10% of the positions are airBS positions
        self.n_airBS_pos = int(len(self.positions) * 0.1)
        indices = np.random.choice(
                    range(len(self.positions)), 
                    self.n_airBS_pos)
        self.airBS_pos = [self.positions[p] for p in indices]
        
        # find all actors in scene
        self.actors = self.findActors()
        
        # print(f'positions: {len(self.positions)}, {self.positions[0]}')
        # for a in self.actors:
        #     print(a)
        #     print(f'name: {a.name}')
        #     print(f'z: {a.getZ()}')
        #     print(f'subclass: {a.getPythonTag("subclass")}')
        #     print(f'subclass: {a.getPythonTag("subclass").AIbehaviors.behaviorStatus("pathfollow")}')
            
        # update mobility task
        # Create Metrics task chain
        self.TaskChainName = 'MobilityChain'
        taskMgr.setupTaskChain(self.TaskChainName, numThreads = 2) # type: ignore
        task_delay = 5
        taskMgr.doMethodLater(task_delay,  # type: ignore
                            self.updateMobility, 
                            f'Update_Mobility',
                            taskChain = self.TaskChainName)
        
    def setPositions(self, positions):
        self.positions = positions
        # 10% of the positions are airBS positions
        self.n_airBS_pos = int(len(self.positions) * 0.1)
        indices = np.random.choice(
                    range(len(self.positions)), 
                    self.n_airBS_pos)
        self.airBS_pos = [self.positions[p] for p in indices]
        
    def findActors(self):
        actors = []
        actor_types = ['ground_actor', 'air_actor']
        for nodepath in render.findAllMatches('**'): # type: ignore
            if nodepath.getTag('type') in actor_types:
                actors.append(nodepath)
        return actors
    
    def updateMobility(self, task):
        for actor in self.actors:
            actor_type = actor.getTag('type')
            if actor_type == 'ground_actor':
                self.updateGroundActor(actor)
            elif actor_type == 'air_actor':
                air_actor_type = actor.getTag('actor')
                if air_actor_type == 'airBS_actor':
                    self.updateAirBSActor(actor)
                elif air_actor_type == 'airUT_actor':
                    self.updateAirUEActor(actor)
        return task.again
    
    def send(self, name, msg):
        messenger.send(name, msg) # type: ignore
    
    def updateGroundActor(self, actor):
        name = actor.name
        D_obj = actor.getPythonTag('subclass')
        if D_obj.AIbehaviors.behaviorStatus("pathfollow") in self.status_types:
            n = np.random.randint(0, len(self.positions))
            p = self.positions[n]
            a_pos = actor.getPos()
            if a_pos.getX() == p[0] and a_pos.getY() == p[1]:
                if n == len(self.positions) - 1:
                    p = self.positions[n-1]
                else:
                    p = self.positions[n+1]
            target = LVecBase3(p[0], p[1], actor.getZ())
            self.send(f'{name}_move', [target])
        
    def updateAirUEActor(self, actor):
        name = actor.name
        D_obj = actor.getPythonTag('subclass')
        if D_obj.AIbehaviors.behaviorStatus("pathfollow") in self.status_types:
            n = np.random.randint(0, len(self.positions))
            p = self.positions[n]
            a_pos = actor.getPos()
            if a_pos.getX() == p[0] and a_pos.getY() == p[1]:
                if n == len(self.positions) - 1:
                    p = self.positions[n-1]
                else:
                    p = self.positions[n+1]
            target = LVecBase3(p[0], p[1], actor.getZ())
            self.send(f'{name}_move', [target])
            
        
    def updateAirBSActor(self, actor):
        name = actor.name
        D_obj = actor.getPythonTag('subclass')
        ai_beh = D_obj.AIbehaviors.behaviorStatus("pathfollow")
        # print(f'{name}, {ai_beh}')
        if ai_beh in self.status_types:
            n = np.random.randint(0, len(self.airBS_pos))
            p = self.airBS_pos[n]
            a_pos = actor.getPos()
            if a_pos.getX() == p[0] and a_pos.getY() == p[1]:
                if n == len(self.airBS_pos) - 1:
                    p = self.airBS_pos[n-1]
                else:
                    p = self.airBS_pos[n+1]
            target = LVecBase3(p[0], p[1], actor.getZ())
            self.send(f'{name}_move', [target])
            # print(f'{name}, {ai_beh}, {target}')
            
    def destroy(self):
        self.ignoreAll()
        taskMgr.remove(f'Update_Mobility')
        self.removeAllTasks()
        # taskMgr.removeTaskChain(self.TaskChainName)
        self.actors = []
        self.positions = []
        self.airBS_pos = []
        self.n_airBS_pos = 0