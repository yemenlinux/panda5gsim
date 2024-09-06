from panda3d.core import (
    NodePath,
    CollisionTraverser,
    CollisionNode,
    CollisionRay,
    CollisionSegment,
    CollisionHandlerQueue,
    BitMask32
)
# This module contains helper functions for the scene graph search.

def findNPbyTag(np_name, tag_name='type'):
    NP_list = []
    for nodepath in render.findAllMatches('**'):
        if nodepath.getTag(tag_name) == np_name:
            NP_list.append(nodepath)
    return NP_list

def isContain(bbox, point):
    # return True if a the point inside the bounding box
    b1, b2 = bbox
    cx = point[0] >= b1[0] and point[0] <= b2[0]
    cy = point[1] >= b1[1] and point[1] <= b2[1]
    cz = point[2] >= b1[2] and point[2] <= b2[2]
    if cx and cy and cz:
        return True
    else:
        return False

def has_line_of_sight(node1, node2):
    # Create a collision traverser
    cTrav = CollisionTraverser()

    # Create a collision handler
    cHandler = CollisionHandlerQueue()

    # Create a collision ray
    cRay = CollisionRay()
    cRay.setOrigin(node1.getPos(render))  # Set the origin of the ray to the position of node1
    cRay.setDirection(node2.getPos(render) - node1.getPos(render))  # Set the direction of the ray towards node2

    # ax, ay, az = node1.getPos(render)
    # bx, by, bz = node2.getPos(render)
    # cRay = CollisionSegment(ax, ay, az, bx, by, bz)
    
    # Create a collision node
    cNode = CollisionNode('sightRay')
    cNode.addSolid(cRay)
    cNode.setFromCollideMask(BitMask32.bit(0))
    cNode.setIntoCollideMask(BitMask32.allOff())

    # Attach the collision node to a new NodePath
    cNP = node1.attachNewNode(cNode)

    # Add the collision node to the traverser
    cTrav.addCollider(cNP, cHandler)
    # Now check for collisions.
    cTrav.traverse(render)
    
    # get a list of collided objects
    # print(cHandler.getNumEntries())
    #
    if cHandler.getNumEntries() == 0:
        return 1
    else:
        return 0
    
def getRayLoS(node1, node2):
    # node1 = Tx, node2 = Rx
    # Create a collision traverser
    cTrav = CollisionTraverser()

    # Create a collision handler
    cHandler = CollisionHandlerQueue()

    # Create a collision ray
    cRay = CollisionRay()
    cRay.setOrigin(node1.getPos(render))  # Set the origin of the ray to the position of node1
    cRay.setDirection(node2.getPos(render) - node1.getPos(render))  # Set the direction of the ray towards node2

    # ax, ay, az = node1.getPos(render)
    # bx, by, bz = node2.getPos(render)
    # cRay = CollisionSegment(ax, ay, az, bx, by, bz)
    
    # Create a collision node
    cNode = CollisionNode('sightRay')
    cNode.addSolid(cRay)
    cNode.setFromCollideMask(BitMask32.bit(0))
    cNode.setIntoCollideMask(BitMask32.allOff())
    # Attach the collision node to a new NodePath
    cNP = node1.attachNewNode(cNode)
    # Add the collision node to the traverser
    cTrav.addCollider(cNP, cHandler)
    # Now check for collisions.
    cTrav.traverse(render)
    
    # get a list of collided objects
    # print(cHandler.getNumEntries())
    #
    #
    if cHandler.getNumEntries() == 0:
            return 1
    else:
        return 0
    
def getRayLoS_nBuildings(node1, node2):
    # node1 = Tx, node2 = Rx
    # Create a collision traverser
    cTrav = CollisionTraverser()

    # Create a collision handler
    cHandler = CollisionHandlerQueue()

    # Create a collision ray
    cRay = CollisionRay()
    cRay.setOrigin(node1.getPos(render))  # Set the origin of the ray to the position of node1
    cRay.setDirection(node2.getPos(render) - node1.getPos(render))  # Set the direction of the ray towards node2

    # ax, ay, az = node1.getPos(render)
    # bx, by, bz = node2.getPos(render)
    # cRay = CollisionSegment(ax, ay, az, bx, by, bz)
    
    # Create a collision node
    cNode = CollisionNode('sightRay')
    cNode.addSolid(cRay)
    cNode.setFromCollideMask(BitMask32.bit(0))
    cNode.setIntoCollideMask(BitMask32.allOff())
    # Attach the collision node to a new NodePath
    cNP = node1.attachNewNode(cNode)
    # Add the collision node to the traverser
    cTrav.addCollider(cNP, cHandler)
    # Now check for collisions.
    cTrav.traverse(render)
    
    # get a list of collided objects
    # print(cHandler.getNumEntries())
    #
    #
    if cHandler.getNumEntries() == 0:
            return (1, cHandler.getNumEntries())
    else:
        return (0, cHandler.getNumEntries())
    
def verify_point_is_outdoor(point):
    # create nodes
    node1 = NodePath('node1')
    node2 = NodePath('node2')
    node1.setPos(point)
    node2.setPos(point)
    node1.setZ(1000)
    node2.setZ(-10)
    node1.reparentTo(render)
    node2.reparentTo(render)
    # 
    # Create a collision traverser
    cTrav = CollisionTraverser()
    # Create a collision handler
    cHandler = CollisionHandlerQueue()
    # Create a collision ray
    cRay = CollisionRay()
    cRay.setOrigin(node1.getPos(render))  # Set the origin of the ray to the position of node1
    cRay.setDirection(node2.getPos(render) - node1.getPos(render))  # Set the direction of the ray towards node2
    # ax, ay, az = node1.getPos(render)
    # bx, by, bz = node2.getPos(render)
    # cRay = CollisionSegment(ax, ay, az, bx, by, bz)
    
    # Create a collision node
    cNode = CollisionNode('sightRay')
    cNode.addSolid(cRay)
    cNode.setFromCollideMask(BitMask32.bit(0))
    cNode.setIntoCollideMask(BitMask32.allOff())
    # Attach the collision node to a new NodePath
    cNP = node1.attachNewNode(cNode)
    # Add the collision node to the traverser
    cTrav.addCollider(cNP, cHandler)
    # Now check for collisions.
    cTrav.traverse(render)
    
    # get a list of collided objects
    # print(cHandler.getNumEntries())
    #
    #
    if cHandler.getNumEntries() == 0:
            return None
    else:
        return point