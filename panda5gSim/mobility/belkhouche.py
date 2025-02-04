# This file is the implementation of the Belkhouche study 2012
# Belkhouche, F., & Bendjilali, B. (2012). Reactive path planning for 3-D autonomous vehicles. IEEE transactions on control systems technology, 20(1), 249-256.

"""
The heading equation with time
$\phi (t) = N_1 \sigma_{xy} (t) + c_1 + b_1 \exp(- d_1 (t - t_0))$
The flight path angle (pich angle)
$\theta (t) = N_2 \sigma_{rz} (t) + c_2 + b_2 \exp(- d_2 (t - t_0))$

where $N_1, N_2 \ge 1$ are real numbers. $\sigma_{xy}$ is the the angle between the target and the heading angle of the UAV.
$\sigma_{rz}$ is the angle between the target and the flight angle of the UAV.
$c_1$ and $c_2$ are controlling the curvature of the navigation path of the UAV. 
$b_1$ and $b_2$ are are used to maintain the smoothness of the path.
$d_1$ and $d_2$ are controlling and characterizing the UAV's turning radius.

The collision avoidance mode is activated when an obstacle is within a given distance from the robot.
An obstacle is avoided by generating a new path by changing the values of the navigation parameters ($N_1, c_1, N_2, c_2$).
Deviation by changing the horizontal plane parameters is referred to as “zero-slope deviation,” and deviation by changing the 
vertical plane parameters is referred to as “infinite-slope deviation”.

Zero-slope deviation:
    Collision in the horizontal plane will take place when
    $\phi_{r} \notin \left[ \sigma_{xy} - \frac{\varepsilon_1}{2},  \sigma_{xy} + \frac{\varepsilon_1}{2}\right]$

Infinite-slope deviation:
    $\varphi_{r} \notin \left[ \sigma_{rz} - \frac{\varepsilon_2}{2},  \sigma_{rz} + \frac{\varepsilon_2}{2}\right]$

class Belkhouche is a subclass of DirectObject.
The Belkhouche class is used to implement the Belkhouche study 2012.
It accepts the following parameters:
    - actor: mobile actor
    - target: target location if not provided, 
    the the generate_target method is called to generate a random 
    target location within default simulation volume.
    - velocity: average velocity of the actor
    - flight_height_range: flight height range
    - N1, N2: real numbers greater than or equal to 1
    - c1, c2: controlling the curvature of the navigation path of the UAV
    default values are between $-\pi/2$ and $\pi/2$
    - b1, b2: maintaining the smoothness of the path
    - d1, d2: controlling and characterizing the UAV's turning radius
    
The Belkhouche class has the following methods:
    - setup: setup the default values of the Belkhouche class
    - generate_target: generate a random target location within 
    the default simulation volume
    - set_direction: set the direction of the actor
    - update: is a task that updates the actor's position, heading angle, and flight angle.
    It also checks the collision with obstacles (buildings) by calling 
    the obstacle_avoidance method.
    - obstacle_avoidance: get the tight bounds of obstacle and calculate
    the desired heading angle and flight angle to avoid the obstacle. it also 
    updates the N1, N2, c1, c2, b1, b2, d1, d2 parameters.
    
    
"""
import numpy as np
from direct.task import Task
from direct.showbase.DirectObject import DirectObject
from panda5gSim.mobility.detectors import obstacle_bounding

class Belkhouche(DirectObject):
    def __init__(self, actor, target=None, velocity=1, flight_height_range=(50, 200), 
                 N1=1, c1=None, b1=None, d1=None, N2=1, c2=None, b2=None, d2=None):
        self.actor = actor
        self.target = target
        self.velocity = velocity
        self.flight_height_range = flight_height_range
        self.N1 = N1
        self.c1 = c1
        self.b1 = b1
        self.d1 = d1
        self.N2 = N2
        self.c2 = c2
        self.b2 = b2
        self.d2 = d2
        self.setup()
        
    def setup(self):
        # setup the default values
        if self.target is None:
            self.generate_target()
        if self.c1 is None:
            self.c1 = np.random.uniform(-np.pi/2, np.pi/2)
        if self.b1 is None:
            self.b1 = np.random.uniform(-np.pi/2, np.pi/2)
        if self.d1 is None:
            self.d1 = np.random.uniform(-np.pi/2, np.pi/2)
        if self.c2 is None:
            self.c2 = np.random.uniform(-np.pi/2, np.pi/2)
        if self.b2 is None:
            self.b2 = np.random.uniform(-np.pi/2, np.pi/2)
        if self.d2 is None:
            self.d2 = np.random.uniform(-np.pi/2, np.pi/2)
            
    def generate_target(self):
        # generate a random target location within the default simulation volume
        self.target = np.random.uniform(-600, 600, 3)
        
    def set_direction(self, task):
        # set the direction of the actor
        # check if has angular velocity or set angular velocity
        if not hasattr(self, 'angular_velocity'):
            # rotate 1 degree per frame
            self.angular_velocity = 1
            # get panda3d frame rate
            # self.frame_rate = globalClock.getAverageFrameRate() # type: ignore
            # self.dt = 1 / self.frame_rate
        # get current heading
        current_heading = self.actor.getH()
        # update heading
        #
    
