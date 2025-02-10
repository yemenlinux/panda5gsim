
from panda3d.core import (
    lookAt,
    Vec3,
    LVecBase3f,
    GeomVertexFormat, 
    GeomVertexData,
    Geom, 
    GeomTriangles, 
    GeomVertexWriter,
    Texture, 
    GeomNode,
    PerspectiveLens,
    TextNode,
    LVector3,
    NodePath,
    TextureStage,
    TexGenAttrib,
    TransformState,
    PNMImage,
    BoundingSphere,
    BoundingBox,
    Point3,
    CardMaker,
    Light,
    Spotlight,
    BitMask32,
    CollisionBox,
    CollisionNode,
    CollisionPolygon,
)
import numpy as np

from panda5gSim.core.geometry import normalized, makeSquare


def makePlaneNP(name, p1, p2, texture_path = None):
    # pu, mu, pv, mv, pw, mw
    geom = makeSquare(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]) # right
    lp = Point3(p1[0], p1[1], p1[2])
    up = Point3(p2[0], p2[1], p2[2])
    snode = GeomNode(name)
    snode.addGeom(geom)
    snode.setBoundsType(3)
    snode.setBounds(BoundingBox(lp, up))
    snode.setFinal(True)
    #
    nodePath = NodePath(name)
    nodePath.attachNewNode(snode)
    # np.reparentTo(render)
    # nodePath.node().setBoundsType(3)
    # nodePath.node().setFinal(True)
    # np.showBounds()
    
    # texture
    texture = base.loader.loadTexture(texture_path)
    texture.setWrapU(Texture.WM_repeat)
    texture.setWrapV(Texture.WM_repeat)
    texture.setWrapW(Texture.WM_repeat)
    
    nodePath.setTexture(texture, 1)
    nodePath.setTexProjector(TextureStage.getDefault(), render, nodePath)
    
    return nodePath
    
class Building(NodePath):
    def __init__(self, name, centerPos, width, depth, height, side_texture_path, roof_texture_path):
        NodePath.__init__(self, name)
        self.name = name
        self.centerPos = centerPos
        self.width = width
        self.depth = depth
        self.height = height
        self.side_texture_path = side_texture_path
        self.roof_texture_path = roof_texture_path
        # self.nodePathList = []
        
        self.setPos(0,0,0)
        self.type = 'building'
        # 
        
        # minx, miny = x - 0.5 * width, y - 0.5*depth
        # maxx, maxy = x + 0.5 * width, y + 0.5 * depth
        # minz, maxz = z, 1 * height
        minx, miny = - 0.5 * width,  - 0.5*depth
        maxx, maxy =  0.5 * width,   0.5 * depth
        minz, maxz = 0, 1 * height
        self.buildingSides = {
            'pu': [(maxx, miny, minz), (maxx, maxy, maxz)],
            'mu': [(minx, maxy, minz), (minx, miny, maxz)],
            'pv': [(maxx, maxy, minz), (minx, maxy, maxz)],
            'mv': [(minx, miny, minz), (maxx, miny, maxz)],
            'pw': [(maxx, miny, minz), (minx, maxy, minz)],
            'mw': [(minx, miny, maxz), (maxx, maxy, maxz)],
        }
        
        # buildingNP = NodePath(self.name)
        
        
        
        # self.nodePathList.append(buildingNP)
        
        self.build()
        # self.setPos(x, y, z)
        self.reparentTo(render)
        self.setPos(*self.centerPos)
        
        # set into collision
        # bb = self.getTightBounds()
        # cBox = CollisionBox(bb[0], bb[1] )
        # self.setCollideMask(BitMask32.allOff())
        # cNode = CollisionNode(f'{self.name}')
        # cNode.addSolid(cBox)
        # cNode.setCollideMask(BitMask32.allOff())
        # cNode.setIntoCollideMask(BitMask32.bit(0))
        # cNode.setTag('type', 'building')
        # self.attachNewNode(cNode)
        self.make_collision_polygon()
        
        self.setTag('building', 'yes')
        # self.setCollideMask(BitMask32.bit(0))
        # 
        
        
    def build(self):
        # choos one side
        doorside =  np.random.choice(['pu', 'mu', 'pv', 'mv'])
        for k, v in self.buildingSides.items():
            side_name = f'{self.name}_{k}'
            if k in ['pw', 'mw']:
                nodePath = makePlaneNP(side_name, v[0], v[1], self.roof_texture_path)
                nodePath.setTag('type', 'roof')
            else:
                nodePath = makePlaneNP(side_name, v[0], v[1], self.side_texture_path)
                nodePath.setTag('type', 'wall')
            if k in [doorside, 'mw']:
                nodePath.setTag('obstacle', 'no')
            else:
                nodePath.setTag('obstacle', 'yes')
            nodePath.reparentTo(self)
            nodePath.setCollideMask(BitMask32.allOff())
            # np.setTwoSided(True)
            # self.nodePathList.append(nodePath)
            
    # def render(self):
    #     for nodePath in self.nodePathList:
    #         nodePath.reparentTo(render)
    #         nodePath.setPos(self.centerPos)
            
    def set2Sides(self):
        for nodePath in self.nodePathList[1:]:
            nodePath.setTwoSided(True)
    
    def destroy(self):
        self.removeNode()
        
    def make_collision_polygon(self):
        p0, p5 = self.getTightBounds()
        p1 = Point3(p5[0], p0[1], p0[2])
        p2 = Point3(p0[0], p0[1], p5[2])
        p3 = Point3(p5[0], p0[1], p5[2])
        p4 = Point3(p5[0], p5[1], p0[2])
        p6 = Point3(p0[0], p5[1], p0[2])
        p7 = Point3(p0[0], p5[1], p5[2])
        #
        list_of_polys = [
            p0, p1, p2, 
            p2, p1, p3,
            p3, p1, p4,
            p4, p3, p5,
            p5, p4, p6,
            p6, p5, p7,
            p7, p6, p0,
            p0, p7, p2,
            p2, p3, p5,
            p5, p2, p7]
        #
        cNode = CollisionNode(f'{self.name}')
        #
        for i in range(0, len(list_of_polys), 3):
            collision_mesh = CollisionPolygon(
                list_of_polys[i], 
                list_of_polys[i+1], 
                list_of_polys[i+2]
            )
            cNode.addSolid(collision_mesh)
            
        #
        cNode.setCollideMask(BitMask32.allOff())
        cNode.setIntoCollideMask(BitMask32.bit(0))
        cNode.setTag('type', 'building')
        self.attachNewNode(cNode)
        
