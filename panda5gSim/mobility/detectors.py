# This file contains the collision detectors
from panda3d.core import (
    NodePath,
    CollisionTraverser,
    CollisionNode,
    CollisionRay,
    CollisionSegment,
    CollisionHandlerQueue,
    BitMask32,
    #
    LVector3,
)

def detect_collisions(node1, node2):
    # node1 = Tx, node2 = Rx
    # Create a collision traverser
    cTrav = CollisionTraverser()

    # Create a collision handler
    cHandler = CollisionHandlerQueue()

    # Create a collision ray
    cRay = CollisionRay()
    cRay.setOrigin(node1.getPos(render))  # Set the origin of the ray to the position of node1
    cRay.setDirection(node2.getPos(render) - node1.getPos(render))  # Set the direction of the ray towards node2

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
    
    #
    if cHandler.getNumEntries() == 0:
        return None
        # return (1, cHandler.getNumEntries())
    else:
        return cHandler.getEntries()[0].getIntoNodePath()
        # return (0, cHandler.getNumEntries())
        
def obstacle_bounding(p1, p2):
    # 
    node1 = NodePath('node1')
    node1.setPos(p1)
    node1.reparentTo(render) # type: ignore
    node2 = NodePath('node2')
    node2.setPos(p2)
    node2.reparentTo(render) # type: ignore
    # Create a collision traverser
    cTrav = CollisionTraverser()

    # Create a collision handler
    cHandler = CollisionHandlerQueue()

    # Create a collision ray
    cRay = CollisionRay()
    cRay.setOrigin(node1.getPos(render))  # Set the origin of the ray to the position of node1
    cRay.setDirection(node2.getPos(render) - node1.getPos(render))  # Set the direction of the ray towards node2

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
    
    #
    if cHandler.getNumEntries() == 0:
        return None
        # return (1, cHandler.getNumEntries())
    else:
        return cHandler.getEntries()[0].getIntoNodePath().getParent().getTightBounds()
        # return (0, cHandler.getNumEntries())