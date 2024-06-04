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
)

# You can't normalize inline so this is a helper function
def normalized(*args):
    myVec = LVector3(*args)
    myVec.normalize()
    return myVec

# helper function to make a square given the Lower-Left-Hand and
# Upper-Right-Hand corners

def makeSquare(x1, y1, z1, x2, y2, z2):
    format = GeomVertexFormat.getV3n3cpt2()
    vdata = GeomVertexData('square', format, Geom.UHDynamic)

    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    color = GeomVertexWriter(vdata, 'color')
    texcoord = GeomVertexWriter(vdata, 'texcoord')

    # make sure we draw the sqaure in the right plane
    if x1 != x2:
        vertex.addData3(x1, y1, z1)
        vertex.addData3(x2, y1, z1)
        vertex.addData3(x2, y2, z2)
        vertex.addData3(x1, y2, z2)

        normal.addData3(normalized(2 * x1 - 1, 2 * y1 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y1 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y2 - 1, 2 * z2 - 1))
        normal.addData3(normalized(2 * x1 - 1, 2 * y2 - 1, 2 * z2 - 1))

    else:
        vertex.addData3(x1, y1, z1)
        vertex.addData3(x2, y2, z1)
        vertex.addData3(x2, y2, z2)
        vertex.addData3(x1, y1, z2)

        normal.addData3(normalized(2 * x1 - 1, 2 * y1 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y2 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y2 - 1, 2 * z2 - 1))
        normal.addData3(normalized(2 * x1 - 1, 2 * y1 - 1, 2 * z2 - 1))

    # adding different colors to the vertex for visibility
    color.addData4f(1.0, 1.0, 1.0, 1.0)
    color.addData4f(1.0, 1.0, 1.0, 1.0)
    color.addData4f(1.0, 1.0, 1.0, 1.0)
    color.addData4f(1.0, 1.0, 1.0, 1.0)

    texcoord.addData2f(0.0, 0.0)
    texcoord.addData2f(1.0, 0.0)
    texcoord.addData2f(1.0, 1.0)
    texcoord.addData2f(0.0, 1.0)

    # Quads aren't directly supported by the Geom interface
    # you might be interested in the CardMaker class if you are
    # interested in rectangle though
    tris = GeomTriangles(Geom.UHDynamic)
    tris.addVertices(0, 1, 3)
    tris.addVertices(1, 2, 3)

    square = Geom(vdata)
    square.addPrimitive(tris)
    return square


class Cube(NodePath):
    def __init__(self, cube_name, pos, scale, texture_path, roof_texture_path ):
        NodePath.__init__(self, cube_name)
        
        
        self.pos = pos
        # self.setPos(0,0,0)
        self.scale = scale
        min_xz = min([scale[0], scale[1]])
        max_axis = max(scale)
        sx = max_axis /2
        sy = max_axis/2
        sz = max_axis/scale[2]
        
        
        
        
        # pu, mu, pv, mv, pw, mw
        square0_pu = makeSquare(0.5, -0.5, 0, 0.5, 0.5, 1) # right
        lp = Point3(0.5, -0.5, 0)
        up = Point3(0.5, 0.5, 1)
        self.addSide('pu', square0_pu, lp, up)
        square1_mu = makeSquare(-0.5, -0.5, 0, -0.5, 0.5, 1) # left
        lp = Point3(-0.5, -0.5, 0)
        up = Point3(-0.5, 0.5, 1)
        self.addSide('mu', square1_mu, lp, up)
        square2_pv = makeSquare(-0.5, 0.5, 0, 0.5, 0.5, 1) # back
        lp = Point3(-0.5, 0.5, 0)
        up = Point3(0.5, 0.5, 1)
        self.addSide('pv', square2_pv, lp, up)
        square3_mv = makeSquare(-0.5, -0.5, 0, 0.5, -0.5, 1) # front
        lp = Point3(-0.5, -0.5, 0)
        up = Point3(0.5, -0.5, 1)
        self.addSide('mv', square3_mv, lp, up)
        square4_pw = makeSquare(-0.5, 0.5, 1, 0.5, -0.5, 1) # top
        lp = Point3(-0.5, 0.5, 1)
        up = Point3(0.5, -0.5, 1)
        self.addSide('pw', square4_pw, lp, up)
        square5_mw = makeSquare(-0.5, 0.5, 0, 0.5, -0.5, 0) # bottom
        lp = Point3(-0.5, 0.5, 0)
        up = Point3(0.5, -0.5, 0)
        self.addSide('mw', square5_mw, lp, up)
        # square4_pw = makeSquare(-0.5, -0.5, 1, 0.5, 0.5, 1) # top
        # lp = Point3(-0.5, -0.5, 1)
        # up = Point3(0.5, 0.5, 1)
        # self.addSide('pw', square4_pw, lp, up)
        # square5_mw = makeSquare(-0.5, -0.5, 0, 0.5, 0.5, 0) # bottom
        # lp = Point3(-0.5, -0.5, 0)
        # up = Point3(0.5, 0.5, 0)
        # self.addSide('mw', square5_mw, lp, up)
        
        
        
        #
        # # self.node().clearBounds()
        # # self.flattenLight()
        # # self.node().setBounds(BoundingSphere(Point3(0,0,0.5), 0.5))
        lp = Point3(-0.5, -0.5, 0)
        up = Point3(0.5, 0.5, 1)
        self.node().setBoundsType(3)
        self.node().setBounds(BoundingBox(lp, up))
        self.node().setFinal(True)
        
        # print(f'boundingbox points: {BoundingBox(lp, up).getPlanes()}')
        
        
        
        self.reparentTo(render)
        self.setPos(self.pos)
        self.setScale(self.scale)
        
        
        
        self.load_side_textures(texture_path)
        self.load_roof_textures(roof_texture_path)
        self.setTexture(self.texture, 1)
        # self.setTexGen(TextureStage.getDefault(), TexGenAttrib.MEyeCubeMap)
        # self.setTexGen(TextureStage.getDefault(), TexGenAttrib.MWorldPosition)
        self.setTexProjector(TextureStage.getDefault(), render, self)
        self.setTexPos(TextureStage.getDefault(), 0, 0, 0)
        self.setTexScale(TextureStage.getDefault(), sx, sy, sz)
        
        top_bottom = ["pw", "mw"]
        for node in self.find_all_matches("**/*"):
            if node.getName() in top_bottom:
                node.setTexture(self.roof_texture, 1)
        
        # self.forceRecomputeBounds()
        # self.setBbox()
        
        self.setTwoSided(True)
        

    def addSide(self, side_name, geom, lp, up):
        snode = GeomNode(side_name)
        snode.addGeom(geom)
        snode.setBoundsType(3)
        snode.setBounds(BoundingBox(lp.normalize(), up.normalize()))
        # snode.setBounds(BoundingBox(lp, up))
        # snode.setFinal(True)
        # print(f'cube: {snode.getBounds()}')
        np = NodePath(side_name)
        np.attachNewNode(snode)
        np.reparentTo(self)
        
        
    def setBbox(self, nodePath = None, lp = None, up = None):
        if nodePath is None:
            nodePath = self.node()
        if lp is None or up is None:
            lp, up = self.getTightBounds()
        # self.node().setBounds(BoundingBox(lp.normalize(), up.normalize()))
        # lp = Point3(-0.5, -0.5, 0)
        # up = Point3(0.5, 0.5, 1)
        self.node().setBoundsType(3)
        # self.node().setBounds(BoundingBox(lp, up))
        self.node().setBounds(BoundingBox(lp.normalize(), up.normalize()))
        self.node().setFinal(True)
        
        
        # nodePath.final = True
        # self.node().setFinal(True) 
        
        print(f'lp, up {lp, up}')
        print(f'cube getBounds {self.getBounds()}')
        print(f'cube getTightBounds() {self.getTightBounds()}')
    
    def load_side_textures(self, side_texture_path):
        self.side_texture_path = side_texture_path
        self.texture = loader.loadTexture(side_texture_path)
        self.texture.setWrapU(Texture.WM_repeat)
        self.texture.setWrapV(Texture.WM_repeat)
        self.texture.setWrapW(Texture.WM_repeat)
        
    def load_roof_textures(self, roof_texture_path):
        self.roof_texture_path = roof_texture_path
        self.roof_texture = loader.loadTexture(roof_texture_path)
        self.roof_texture.setWrapU(Texture.WM_repeat)
        self.roof_texture.setWrapV(Texture.WM_repeat)
        self.roof_texture.setWrapW(Texture.WM_repeat)
        
    