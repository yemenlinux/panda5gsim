import numpy as np
# np.random.seed(0)

# from panda3d.core import CollisionTraverser,CollisionNode
# from panda3d.core import CollisionHandlerQueue,CollisionRay
from panda3d.core import (
    Vec3,
    LVecBase3f,
    LPoint3,
    NodePath,
    Vec4,
    BitMask32,
    CollisionTraverser,
    CollisionNode,
    CollisionHandlerQueue,
    CollisionBox,
    CollisionRay,
    BoundingBox,
)
from direct.showbase.DirectObject import DirectObject
from panda3d.ai import *
from direct.task import Task


from panda5gSim import ASSETS_DIR, OUTPUT_DIR
from panda5gSim.core.scene_graph import findNPbyTag


def destroyNP(obj_list):
    for obj in obj_list:
        obj.detachNode()
        obj.removeNode()
        # obj.delete()
        del obj
    del obj_list
    
def destroyDirectObj(obj_list):
    for obj in obj_list:
        obj.ignoreAll()
        obj.removeAllTasks()
        # obj.model.detachNode()
        # obj.model.removeNode()
        obj.detachNode()
        del obj
    del obj_list
    
# AiWorld
class AiWorld(DirectObject):
    def __init__(self, parent):
        DirectObject.__init__(self)
        #
        self.parent = parent
        # 
        # self.TaskChainName = 'AiWorldTaskChain'
        # taskMgr.setupTaskChain(self.TaskChainName, numThreads = 2)
        # AI world
        self.AIworld = AIWorld(render)
        # search for actors
        # self.findActors()
        #
        self.cycle_time = 300
        # set AI environment
        self.setAI()
        # accepted messages
        self.accept('DestroyAI', self.destroy)
    
    def setAI(self):
        # if not hasattr(self.parent, 'ground_users'):
        #     self.parent.genGroundUsers()
        # if not hasattr(self.parent, 'air_users'):
        #     self.parent.genAirUsers()
        # if not hasattr(self.parent, 'air_bs'):
        #     self.parent.genAirBS()
        #
        mobility_mode = SimData['Mobility']['Mobility modes']
        mobility_prob = SimData['Mobility']['Mobility modes probability']
        self.mobility_mode = np.random.choice(mobility_mode, 
                                            size = len(self.parent.ground_users), 
                                            p = mobility_prob)
        #
        mass = 50.0
        movt_force = 30.0
        max_force = 75.0
        airBS_force = 30.0 #  N (kg.m/s^2)
        airut_force = 5.0 #  N (kg.m/s^2)
        gut_force = 3.0 #  N (kg.m/s^2)
        print(f'OUTPUT_DIR: {OUTPUT_DIR}')
        # ground users
        # for user in self.parent.ground_users:
        if hasattr(self.parent, 'ground_users'):
            for i in range(len(self.parent.ground_users)):
                # set mobility mode
                self.parent.ground_users[i].setMobilityMode(self.mobility_mode[i])
                self.parent.ground_users[i].AIchar = \
                    AICharacter(self.parent.ground_users[i].name, 
                                self.parent.ground_users[i].Actor, 
                                mass, 
                                gut_force*0.5, 
                                gut_force)
                                
                self.AIworld.addAiChar(self.parent.ground_users[i].AIchar)
                self.parent.ground_users[i].AIbehaviors = \
                    self.parent.ground_users[i].AIchar.getAiBehaviors()
                if self.mobility_mode[i] == 'PoI':
                    stay_time = self.gen_n_weighted_pois(2, 6) * self.cycle_time
                else:
                    stay_time = self.gen_n_weighted_pois(2, 21) * self.cycle_time
                # print(f'stay time: {stay_time}, {list(stay_time)}')
                self.parent.ground_users[i].setStayTime(stay_time)
                indices = np.random.choice(range(len(self.parent.ground_users_locations)), 
                                        len(stay_time))
                self.parent.ground_users[i].setPoIs(
                    [self.parent.ground_users_locations[p] for p in indices]
                )
                # 
                # self.parent.ground_users[i].init_nav_mesh()
                # if hasattr(self.parent, 'buildings'):
                #     for building in self.parent.buildings:
                #         self.parent.ground_users[i].AIbehaviors.addStaticObstacle(building)
            
            # air users
            if hasattr(self.parent, 'air_users'):
                for i in range(len(self.parent.air_users)):
                    # set mobility mode
                    self.parent.air_users[i].setMobilityMode('Random')
                    self.parent.air_users[i].AIchar = \
                        AICharacter(self.parent.air_users[i].name, 
                                    self.parent.air_users[i].Actor, 
                                    mass, 
                                    airut_force*0.5, 
                                    airut_force)
                    self.AIworld.addAiChar(self.parent.air_users[i].AIchar)
                    self.parent.air_users[i].AIbehaviors = \
                        self.parent.air_users[i].AIchar.getAiBehaviors()
                    stay_time = self.gen_n_weighted_pois(2, 11) * self.cycle_time
                    # print(f'stay time: {stay_time}, {list(stay_time)}')
                    self.parent.air_users[i].setStayTime(stay_time)
                    indices = np.random.choice(range(len(self.parent.air_users_locations)), 
                                            len(stay_time))
                    self.parent.air_users[i].setPoIs(
                        [self.parent.air_users_locations[p] for p in indices]
                    )
            
            # AirBS
            stay_time = self.gen_n_weighted_pois(5, 30) * self.cycle_time
            for i in range(len(self.parent.air_bs)):
                # set mobility mode
                self.parent.air_bs[i].setMobilityMode('Random')
                self.parent.air_bs[i].AIchar = \
                    AICharacter(self.parent.air_bs[i].name, 
                                self.parent.air_bs[i].Actor, 
                                mass, 
                                airBS_force*0.5, 
                                airBS_force)
                                
                self.AIworld.addAiChar(self.parent.air_bs[i].AIchar)
                self.parent.air_bs[i].AIbehaviors = \
                    self.parent.air_bs[i].AIchar.getAiBehaviors()
                # stay_time = self.gen_n_weighted_pois(5, 30) * self.cycle_time
                # print(f'stay time: {stay_time}, {list(stay_time)}')
                self.parent.air_bs[i].setStayTime(stay_time)
                indices = np.random.choice(range(len(self.parent.air_bs_locations)), 
                                        len(stay_time))
                # 
                _pois = [self.parent.air_bs_locations[p] for p in indices]
                airbs_height = self.parent.air_bs[i].Actor.getZ()
                pois = []
                for j in range(len(_pois)):
                    p = _pois[j]
                    pois.append((p[0], p[1], airbs_height))
                self.parent.air_bs[i].setPoIs(pois)
                #
                # self.parent.air_bs[i].init_nav_mesh()
                # if hasattr(self.parent, 'buildings'):
                #     for building in self.parent.buildings:
                #         self.parent.air_bs[i].AIbehaviors.addStaticObstacle(building)
            
            
        # update AIWorld
        taskMgr.add(self.AIUpdate, 
                    "AIUpdate")
        # taskMgr.add(self.AIUpdate, 
        #             "AIUpdate",
        #             taskChain = self.TaskChainName)
        
    def AIUpdate(self, task):
        # self.handle_ai()
        self.AIworld.update()
        return Task.cont
        
    def destroy(self):
        # remove aichar
        if hasattr(self.parent, 'ground_users'):
            for i in range(len(self.parent.ground_users)):
                self.AIworld.removeAiChar(self.parent.ground_users[i].name)
        if hasattr(self.parent, 'air_users'):
            for i in range(len(self.parent.air_users)):
                self.AIworld.removeAiChar(self.parent.air_users[i].name)
        if hasattr(self.parent, 'air_bs'):
            for i in range(len(self.parent.air_bs)):
                self.AIworld.removeAiChar(self.parent.air_bs[i].name)
        # delete the AI world
        self.ignoreAll()
        # remove tasks
        self.removeAllTasks()
        # taskMgr.remove(self.TaskChainName)
        # print(f'{self.TaskChainName}: Destroyed.')
        del self
        
    ### generate ground user's mobility
    def gen_n_weighted_pois(self, min_pois = 2, max_pois = 11):
        # n = np.random.randint(min_pois, max_pois)
        # weights = np.round(np.random.dirichlet(np.ones(n)), decimals=2)
        while True:
            n = np.random.randint(min_pois, max_pois)
            weighted_pois = np.round(np.random.dirichlet(np.ones(n)), decimals=2)
            if np.sum(weighted_pois) == 1.0:
                result = weighted_pois
                break
        return result
        
    

