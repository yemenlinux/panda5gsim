# This file contains receiver class
import numpy as np
from panda3d.core import (
    Geom,
    GeomVertexFormat,
    GeomVertexData,
    GeomVertexWriter,
    GeomPoints,
    GeomNode,
    NodePath,
)

from panda5gSim.core.scene_graph import (
    findNPbyTag,
    has_line_of_sight
)
from panda5gSim.users.antennas import get_3d_power_radiation_pattern


def makePoint():
    # Create a new GeomPoints object
    format = GeomVertexFormat.getV3()
    vdata = GeomVertexData('points', format, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, 'vertex')
    points = GeomPoints(Geom.UHStatic)

    # Add a point at position (0, 0, 0)
    vertex.addData3f(0, 0, 0)
    points.addVertex(0)

    # Create a Geom object to hold the points
    geom = Geom(vdata)
    geom.addPrimitive(points)

    # Create a GeomNode to store the Geom object
    node = GeomNode('points')
    node.addGeom(geom)
    return node


class PointNode(NodePath):
    def __init__(self, name):
        NodePath.__init__(self, name)
        # Attach the GeomNode to the render
        self.attachNewNode(makePoint())
        
class Antenna(NodePath):
    def __init__(self, name='Antenna', mimo = 1):
        # PointNode.__init__(self, name)
        NodePath.__init__(self, name)
        NodePath.setPythonTag(self, "subclass", self)
        # Attach the GeomNode to the render
        self.node = self.attachNewNode(makePoint())
        # stash the point node
        self.node.stash()
        #
        self.beamwidth = 65
        self.MIMO = mimo
        self.setTag('type', name)
        
        
        self.state = 1
        dev = self.name[:-3]
        if dev == 'airBS' or dev == 'gBS':
            self.antenna_type = '3GPP'
            self.max_gain_dB = 30 # in dB
            self.Tx_power = 40 # in dB
        else:
            self.antenna_type = 'omni'
            self.max_gain_dB = 4 # in dB
            self.Tx_power = 10*np.log10(0.200)
        
    def setBeamWidth(self, angle):
        # angle in degree
        self.beamwidth = angle
        
    def setOn(self):
        self.state = 1
        
    def setOff(self):
        self.state = 0
        
    def getOnOff(self):
        return self.state
    
    def setTxPoser(self, tx_power):
        self.Tx_power = tx_power
        
    def getTxPowerdB(self, phi = 0, theta = 0):
        # phi and theta in degree
        gain = self.getGain(phi, theta)
        # get the Tx_power in dBm
        return self.Tx_power + gain
    
    def getTxPower(self, phi = 0, theta = 0):
        # get Tx_power in W
        return 10**(self.getTxPowerdB(phi, theta)/10)
    
    def getRxPowerdB(self, in_power):
        # return the received power in dB after the antenna
        # gain is added
        gain = self.getGain()
        return in_power + gain
        
    def getRxPower(self, in_powerW, phi = 0, theta = 0):
        # return the received power in W after the antenna
        # gain is added
        in_power = self.convertW2dB(in_powerW)
        return 10**(self.getRxPowerdB(in_power, phi, theta)/10)
    
    def setGain(self, gain):
        self.max_gain_dB = gain
        
    def getGain(self, phi=0, theta=0):
        if self.antenna_type == 'omni':
            return self.max_gain_dB
        else:
            return get_3d_power_radiation_pattern(
                transformed_phi = phi,
                transformed_theta = theta,
                theta_3db = 65, 
                phi_3db = 65, 
                SLA_v = 30, 
                A_max = self.max_gain_dB)
        
    
    def setMIMO(self, mimo):
        self.MIMO = mimo
    
    def setDirection(self, az_angle, tilt_angle= 0):
        self.setHpr(render, az_angle, tilt_angle, 0)
    
    # def get_gain(self, phi, theta):
    #     """
    #     Return the antenna gain in dB for the given phi and theta angles.

    #     Parameters:
    #     phi : float
    #         The azimuth angle in radians.
    #     theta : float
    #         The elevation angle in radians.

    #     Returns:
    #     float
    #         The antenna gain in dB.
    #     """
    #     # Calculate the angle between the direction the antenna is pointing and the direction to the user
    #     angle_diff = np.sqrt(phi**2 + theta**2)
    #     # Calculate the gain using a simple model where the gain decreases as the angle increases
    #     gain_db = self.max_gain_dB - 12 * (angle_diff / self.beamwidth)**2
    #     # Ensure the gain is not negative
    #     gain_db = max(gain_db, 0)
    #     return gain_db
    
    def convertdB2dBm(self, dB):
        return dB + 30
    
    def convertW2dBm(self, W):
        return 10*np.log10(W) + 30
    def convertW2dB(self, W):
        return 10*np.log10(W)
    def convertdB2W(self, dB):
        return 10**(dB/10)
    
# class Rx(Antenna):
#     def __init__(self, name='Rx', mimo = 1):
#         Antenna.__init__(self, name, mimo)
#         # self.setTag('type', name)
        
# class Tx(Antenna):
#     def __init__(self, name='Tx', mimo = 1):
#         Antenna.__init__(self, name, mimo)
#         # self.Antenna = Antenna(mimo = mimo)
#         # self.setTag('type', name)
        
        
class Device(NodePath):
    def __init__(self, name, mimo = (1,1)):
        NodePath.__init__(self, name)
        self.name = name
        self.mimo = mimo
        # self.Rx = Rx(f'{name}_Rx', mimo = mimo[0])
        # self.Tx = Tx(f'{name}_Tx', mimo = mimo[1])
        self.Rx = Antenna(f'{name}_Rx', mimo = mimo[0])
        self.Tx = Antenna(f'{name}_Tx', mimo = mimo[1])
        self.Tx.reparentTo(self)
        self.Rx.reparentTo(self)
        self.setTag('type', name)
        self.setPythonTag("subclass", self)
        self.RxFrom = []
        self.state = 1
        #
        self.connectedTo = None
        self.distanceTo = None
        self.currentISNR = None
        self.sinr = None
        self.interference = None
        self.received_power = None
        self.path_loss = None
        self.SF = None
        self.los = None
    
    def getVelocity(self):
        # get the subclass of parent class
        parent = self.getParent()
        if parent.name == 'RightHand':
            parent = parent.getParent().getParent()
        #
        actor = parent.getPythonTag("subclass")
        # print(parent.name, actor.AIchar.getVelocity())
        return actor.AIchar.getVelocity()
    
    def getActorName(self):
        # get the subclass of parent class
        parent = self.getParent()
        if parent.name == 'RightHand':
            parent = parent.getParent().getParent()
        #
        actor = parent.getPythonTag("subclass")
        # print(parent.name, actor.AIchar.getVelocity())
        return actor.name
    
        
class GroundUE(Device):
    def __init__(self, name = 'gUT', mimo = (1,1)):
        Device.__init__(self, name, mimo)
        self.name = name
        self.RxFrom = ['gBS', 'airBS']
        
class GroundBS(Device):
    def __init__(self, name= 'gBS', mimo = (1,1)):
        Device.__init__(self, name, mimo)
        self.RxFrom = ['gUE', 'airUT']
        
class AirUE(Device):
    def __init__(self, name = 'airUT', mimo = (1,1)):
        Device.__init__(self, name, mimo)
        self.RxFrom = ['gBS', 'airBS']
        
class AirBS(Device):
    def __init__(self, name = 'airBS', mimo = (1,1)):
        # super().__init__(name)
        Device.__init__(self, name, mimo)
        self.RxFrom = ['gUT', 'airUT']