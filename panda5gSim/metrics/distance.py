# This module contains the functions to calculate 
# the distance between two points in the 3D space.
# The distance is calculated using the Euclidean 
# distance formula.

def getDistance3D(node1, node2):
    # calculate the distance between two path nodes.
    return node1.getDistance(node2)

def getDistance2D(node1, node2):
    # calculate the distance between two path nodes.
    p1 = node1.getPos()
    p2 = node2.getPos()
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def getDistance2DPos(p1, p2):
    # calculate the distance between two points.
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def getDistance3DPos(p1, p2):
    # calculate the distance between two points.
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5

