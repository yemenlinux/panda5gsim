# AutoPilot class for 3D mobility simulation of UAVs
# Author: Basheer Raddwan
# Date: 2024-10-23

# Importing libraries
import numpy as np
from direct.task import Task
from direct.showbase.DirectObject import DirectObject
from panda3d.core import (
    NodePath,
    #
    LVector3,
)
# from panda5gSim.users.receiver import PointNode
from panda5gSim.mobility.detectors import obstacle_bounding


""" AutoPilot class:
    AutoPilot class takes the UAV actor object and a target location
    as input and moves the UAV actor to the target location.
    
"""
class AutoPilotMove(DirectObject):
    def __init__(self, actor, target_location, speed = 30, debug = False):
        DirectObject.__init__(self)
        self.Actor = actor.Actor
        self.target_location = LVector3(*target_location)
        self.speed = speed
        self.frame_rate = 60
        self.dt = 1 / self.frame_rate
        self.deltad = self.speed * self.dt
        #
        self.arrived = False
        #
        self.TaskChainName = 'MobilityChain'
        # taskMgr.setupTaskChain(self.TaskChainName, numThreads = 2) # type: ignore
        # self.addTask(self.move,
        #             name='move',
        #             taskChain = self.TaskChainName,
        #             )
        #
        self.debug = debug
        
        #
        # self.collisionNode = NodePath('node2')
        # self.collisionNode.reparentTo(render) # type: ignore
        self.stack = []
    
    def move(self, task):
        # Calculate the direction vector
        direction = self.target_location - self.Actor.getPos()
        direction.normalize()
        # Move the actor towards the target location
        current_location = self.Actor.getPos()
        delta = direction * self.deltad
        next_location = current_location + delta
        #
        if self.debug:
            if np.random.random() > 0.8:
                print(f'UAV direction: {self.Actor.getHpr()}')
                
        if (self.target_location - current_location).length() < self.deltad:
            self.Actor.setPos(self.target_location)
            self.arrived = True
            return Task.done
        else:
            phi = np.rad2deg(np.arcsin(direction.get_y()/direction.get_xy().length()))
            theta = np.rad2deg(np.arccos(direction.get_xy().length()/direction.length()))
            self.Actor.setH(phi)
            self.Actor.setPos(next_location)
            return Task.cont
    
    def move_RWP(self, task):
        # random way point (RWP) mobility
        if not hasattr(self, 'max_rw_points'):
            self.max_rw_points = 10 # set this
            self.num_rw_points = 0
        if not hasattr(self, 'rw_point'):
            self.rw_point = LVector3(
                np.random.randint(-600, 600),
                np.random.randint(-600, 600),
                np.random.randint(0, 200)
            )
            self.Actor.lookAt(self.rw_point)
        # 
        self.frame_rate = globalClock.getAverageFrameRate() # type: ignore
        if self.frame_rate == 0:
            self.frame_rate = 60
            # print('Frame rate is zero')
        self.dt = 1 / self.frame_rate
        self.deltad = self.speed * self.dt
        #
        if (self.rw_point - self.Actor.getPos()).length() < self.deltad:
            self.Actor.setPos(self.rw_point)
            self.arrived = True
            self.num_rw_points += 1
            if self.num_rw_points == self.max_rw_points:
                return Task.done
            # generate New point
            del self.rw_point
            return Task.cont
        else:
            direction = self.Actor.getQuat().getForward()
            next_location = self.Actor.getPos() + direction * self.deltad
            self.Actor.setPos(next_location)
            return Task.cont
    
    def move_RWP_with_collision_detection(self, task):
        # random way point (RWP) mobility
        if not hasattr(self, 'max_rw_points'):
            self.max_rw_points = 100 # set this
            self.num_rw_points = 0
        if not hasattr(self, 'rw_point'):
            if len(self.stack) > 0:
                self.rw_point = self.stack.pop()
            else:
                self.rw_point = LVector3(
                    np.random.randint(-600, 600),
                    np.random.randint(-600, 600),
                    np.random.randint(0, 50)
                )
                print(f'New rw_point: {self.rw_point}')
            self.Actor.lookAt(self.rw_point)
        else:
            self.Actor.lookAt(self.rw_point)
        # 
        self.frame_rate = globalClock.getAverageFrameRate() # type: ignore
        if self.frame_rate == 0:
            self.frame_rate = 60
        self.dt = 1 / self.frame_rate
        self.deltad = self.speed * self.dt
        #
        if (self.rw_point - self.Actor.getPos()).length() <= self.deltad:
            self.Actor.setPos(self.rw_point)
            self.arrived = True
            if len(self.stack) == 0:
                self.num_rw_points += 1
            if self.num_rw_points == self.max_rw_points:
                return Task.done
            # generate New point
            del self.rw_point
            return Task.cont
        else:
            direction = self.Actor.getQuat().getForward()
            next_location = self.Actor.getPos() + direction * self.deltad
            # check for collisions
            # self.collisionNode.setPos(next_location)
            collisions = obstacle_bounding(self.Actor.getPos(), 
                                           next_location)
            if collisions is not None:
                # stack current rw_point
                self.stack.append(self.rw_point)
                # generate new rw_point
                tb = collisions
                self.rw_point = self.get_new_point(tb, next_location.z)
                self.Actor.lookAt(self.rw_point)
                direction = self.Actor.getQuat().getForward()
                next_location = self.Actor.getPos() + direction * self.deltad
                self.Actor.setPos(next_location)
                return Task.cont
                 
            self.Actor.setPos(next_location)
            return Task.cont
        
    def get_new_point(self, tb, z):
        # 
        gap = 5
        p1 = LVector3(tb[0][0]-gap, tb[0][1]-gap, z)
        p2 = LVector3(tb[1][0]+gap, tb[0][1]-gap, z)
        p3 = LVector3(tb[0][0]-gap, tb[1][1]+gap, z)
        p4 = LVector3(tb[1][0]+gap, tb[1][1]+gap, z)
        d1 = (p1 - self.Actor.getPos()).length()
        d2 = (p2 - self.Actor.getPos()).length()
        d3 = (p3 - self.Actor.getPos()).length()
        d4 = (p4 - self.Actor.getPos()).length()
        l = sorted([d1, d2, d3, d4])
        return p1
        if l[0] == d1:
            return p1
        elif l[0] == d2:
            return p2
        elif l[0] == d3:
            return p3
        else:
            return p4
        
    
def set_direction(self, task):
    # check if has angular velocity or set angular velocity
    if not hasattr(self, 'angular_velocity'):
        # rotate 1 degree per frame
        self.angular_velocity = 1
        # get panda3d frame rate
        # self.frame_rate = globalClock.getAverageFrameRate() # type: ignore
        # self.dt = 1 / self.frame_rate
    # get current heading
    current_heading = self.Actor.getH()
    # update heading
    # Calculate the direction vector
    direction = self.target_location - self.Actor.getPos()
    direction.normalize()
    # calculate angle between current heading and direction
    phi = np.rad2deg(np.arcsin(direction.get_y()/direction.get_xy().length()))
    theta = np.rad2deg(np.arccos(direction.get_xy().length()/direction.length()))
    # update heading
    if current_heading < phi:
        current_heading += self.angular_velocity
    elif current_heading > phi:
        current_heading -= self.angular_velocity
    # set new heading
    if abs(current_heading - phi) <= self.angular_velocity:
        current_heading = phi
        self.Actor.setH(current_heading)
        return Task.done
    self.Actor.setH(current_heading)
    return Task.cont

def set_flight_height(self, task):
    # check if has flight constraints or set flight constraints
    if not hasattr(self, 'flight_height_range'):
        # set flight height range
        self.flight_height_range = (50, 200)
    if not hasattr(self, 'flight_height'):
        # select randomly a flight height
        self.flight_height = np.random.randint(*self.flight_height_range)
    # get current height
    current_height = self.Actor.getZ()
    # update height
    if current_height < self.flight_height_range[0]:
        current_height += 2
    elif current_height > self.flight_height_range[1]:
        current_height -= 2
    elif current_height < self.flight_height:
        current_height += 1
    elif current_height > self.flight_height:
        current_height -= 1
    # set new height
    if abs(current_height - self.flight_height) <= 1:
        current_height = self.flight_height
        self.Actor.setZ(current_height)
        return Task.done
    self.Actor.setZ(current_height)
    return Task.cont

def move_in_the_same_direction(self, task):
    # get the current direction
    direction = self.Actor.getHpr()