import random
import numpy as np
from direct.showbase.DirectObject import DirectObject


from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    PerspectiveLens,
    TextNode,
    TexGenAttrib,
    TextureStage,
    TransparencyAttrib,
    LPoint3,
    LVector3,
    Texture,
)

class CameraControl(DirectObject):
    def __init__(self, **kwargs):
        DirectObject.__init__(self)
        #
        if 'number_of_cameras' in kwargs:
            self.number_of_cameras = kwargs['number_of_cameras']
        else:
            self.number_of_cameras = 3
        #
        self.xray_mode = False
        self.show_model_bounds = False
        
        self.camera_index = 0
        self.default_camera_Pos = None
        self.default_camera_Hpr = None
        self.save_default_camera = True
        # self.camera_list = []
        self.camera_parent = [
            render, # type: ignore
            self.selectRandomActor('ground_actor'), 
            self.selectRandomActor('air_actor')]
        
        # Display instructions
        # add_title("Panda3D Tutorial: Occluder Culling")
        # add_instructions(0.06, "[Esc]: Quit")
        # add_instructions(0.12, "[W]: Move Forward")
        # add_instructions(0.18, "[A]: Move Left")
        # add_instructions(0.24, "[S]: Move Right")
        # add_instructions(0.30, "[D]: Move Back")
        # add_instructions(0.36, "Arrow Keys: Look Around")
        # add_instructions(0.42, "[F]: Toggle Wireframe")
        # add_instructions(0.48, "[X]: Toggle X-Ray Mode")
        # add_instructions(0.54, "[B]: Toggle Bounding Volumes")
        
        # Setup controls
        self.keys = {}
        for key in ['arrow_left', 'arrow_right', 'arrow_up', 'arrow_down',
                    'a', 'd', 'w', 's']:
            self.keys[key] = 0
            self.accept(key, self.push_key, [key, 1])
            self.accept('shift-%s' % key, self.push_key, [key, 1])
            self.accept('%s-up' % key, self.push_key, [key, 0])
        # self.accept('f', self.toggleWireframe)
        self.accept('x', self.toggle_xray_mode)
        self.accept('b', self.toggle_model_bounds)
        self.accept('c', self.changeCamera)
        self.accept('n', self.nextActor)
        # self.accept('escape', __import__('sys').exit, [0])
        # base.disableMouse()
        
        # Setup camera
        # self.camera = base.cam
        self.lens = PerspectiveLens()
        self.lens.setFov(60)
        self.lens.setNear(0.01)
        self.lens.setFar(1000.0)
        base.cam.node().setLens(self.lens) # type: ignore
        # self.camera.setPos(-9, -0.5, 1)
        self.heading = 0.0
        self.pitch = 0.0
        
        taskMgr.add(self.update, 'mouse_loop') # type: ignore
        
    def findNP(self, np_tag):
        actor_list = []
        for obj in render.findAllMatches('**'): # type: ignore
            if obj.getTag('type') == np_tag:
                actor_list.append(obj)
        return actor_list
    
    def selectRandomActor(self, actor_tag):
        actor_list = self.findNP(actor_tag)
        if len(actor_list) > 0:
            return actor_list[random.randint(0, len(actor_list)-1)]
        else:
            return render # type: ignore
        
    def changeCamera(self):
        #
        if self.save_default_camera:
            if self.camera_index == 0:
                self.default_camera_Pos = base.cam.getPos()
                self.default_camera_Hpr = base.cam.getHpr()
                self.save_default_camera = not self.save_default_camera
        
        if self.camera_index + 1 >= self.number_of_cameras:
            self.camera_index = 0
        else:
            self.camera_index += 1
        #
        self.setCamera()
        
    def nextActor(self):
        if self.camera_index == 1:
            self.camera_parent[self.camera_index] = \
                self.selectRandomActor('ground_actor')
        if self.camera_index == 2:
            self.camera_parent[self.camera_index] = \
                self.selectRandomActor('air_actor')
        if self.camera_index > 0:
            self.setCamera()
        
    def setCamera(self):
        if self.camera_index > 0:
            base.cam.setPosHpr(0, 0, 0, 0, 0, 0)
            base.cam.reparentTo(self.camera_parent[self.camera_index])
            base.cam.setY(base.cam.getY() + 30)
            base.cam.setZ(base.cam.getZ() + 10)
            base.cam.setHpr(180,-15,0)
        else:
            base.cam.setPos(self.default_camera_Pos)
            base.cam.setHpr(self.default_camera_Hpr)
            base.cam.reparentTo(render)
            self.save_default_camera = not self.save_default_camera
            
    def push_key(self, key, value):
        """Stores a value associated with a key."""
        self.keys[key] = value

    def update(self, task):
        """Updates the camera based on the keyboard input."""
        if self.camera_index == 0:
            self.update_camera()
        return task.cont
    
    def update_camera(self):
        delta = base.clock.dt
        move_x = delta * 30 * -self.keys['a'] + delta * 30 * self.keys['d']
        move_z = delta * 30 * self.keys['s'] + delta * 30 * -self.keys['w']
        base.cam.setPos(base.cam, move_x, -move_z, 0)
        self.heading += (delta * 90 * self.keys['arrow_left'] +
                         delta * 90 * -self.keys['arrow_right'])
        self.pitch += (delta * 30 * self.keys['arrow_up'] +
                       delta * 30 * -self.keys['arrow_down'])
        base.cam.setHpr(self.heading, self.pitch, 0)
    
    def update_mouse_pos(self):
        """Updates the camera based on the mouse input."""
        if base.mouseWatcherNode.hasMouse():
            mouse_pos = base.mouseWatcherNode.getMouse()
            # self.heading -= mouse_pos.x * 10
            # self.pitch += mouse_pos.y * 10
            # self.camera.setHpr(self.heading, self.pitch, 0)

    def toggle_xray_mode(self):
        """Toggle X-ray mode on and off. This is useful for seeing the
        effectiveness of the occluder culling."""
        self.xray_mode = not self.xray_mode
        if self.xray_mode:
            for node in self.findNP('building'):
                node.setColorScale((1, 1, 1, 0.5))
                node.setTransparency(TransparencyAttrib.MDual)
        else:
            for node in self.findNP('building'):
                node.setColorScaleOff()
                node.setTransparency(TransparencyAttrib.MNone)
            

    def toggle_model_bounds(self):
        """Toggle bounding volumes on and off on the models."""
        self.show_model_bounds = not self.show_model_bounds
        find_list = ['building', 'wall', 'roof', 'ground_user', 'air_user']
        show_list = []
        
        for x in find_list:
            show_list += self.findNP(x)
        if self.show_model_bounds:
            # for model in self.findNP('building'):
            #     model.showBounds()
            for model in show_list:
                model.showBounds()
        else:
            # for model in self.findNP('building'):
            #     model.hideBounds()
            for model in show_list:
                model.hideBounds()
    
class CameraMgr(DirectObject):
    def __init__(self, camera):
        DirectObject.__init__(self)
        self.camera = camera
        #
        self.xray_mode = False
        self.show_model_bounds = False
        self.camera_follow_actor = False
        self.pf_gideshow = False
        self.current_actor = None
        #
        self.findActors()
        #
        self.accept('x', self.toggle_xray_mode)
        self.accept('b', self.toggle_model_bounds)
        self.accept('c', self.changeCamera)
        self.accept('n', self.nextActor)
        self.accept('p', self.setPfGider)
        
    def findActors(self):
        self.actors = []
        actor_types = ['ground_actor', 'air_actor']
        for nodepath in render.findAllMatches('**'): # type: ignore
            if nodepath.getTag('type') in actor_types:
                self.actors.append(nodepath)
        
    def findNP(self, np_tag):
        actor_list = []
        for obj in render.findAllMatches('**'): # type: ignore
            if obj.getTag('type') == np_tag:
                actor_list.append(obj)
        return actor_list
    
    def toggle_xray_mode(self):
        """Toggle X-ray mode on and off. This is useful for seeing the
        effectiveness of the occluder culling."""
        self.xray_mode = not self.xray_mode
        if self.xray_mode:
            for node in self.findNP('building'):
                node.setColorScale((1, 1, 1, 0.5))
                node.setTransparency(TransparencyAttrib.MDual)
        else:
            for node in self.findNP('building'):
                node.setColorScaleOff()
                node.setTransparency(TransparencyAttrib.MNone)
        
    def toggle_model_bounds(self):
        """Toggle bounding volumes on and off on the models."""
        self.show_model_bounds = not self.show_model_bounds
        find_list = ['building', 'wall', 'roof', 'ground_user', 'air_user']
        show_list = []
        
        for x in find_list:
            show_list += self.findNP(x)
        if self.show_model_bounds:
            # for model in self.findNP('building'):
            #     model.showBounds()
            for model in show_list:
                model.showBounds()
        else:
            # for model in self.findNP('building'):
            #     model.hideBounds()
            for model in show_list:
                model.hideBounds()
    
    def save_camera_position(self):
        self.camera_pos = self.camera.getPos()
        self.camera_hpr = self.camera.getHpr()
        self.camera_PosHpr = (LVector3(self.camera_pos.getX(),
                                       self.camera_pos.getY(),
                                       self.camera_pos.getZ()),
                              LVector3(self.camera_hpr.getX(),
                                       self.camera_hpr.getY(),
                                       self.camera_hpr.getZ()))
        
    def nextActor(self):
        if not self.camera_follow_actor:
            self.save_camera_position()
            self.camera_follow_actor = not self.camera_follow_actor
        if self.camera_follow_actor:
            n_actor = np.random.randint(0, len(self.actors)-1)
            
            self.camera.setPosHpr(0, 0, 0, 0, 0, 0)
            self.camera.reparentTo(self.actors[n_actor])
            if self.actors[n_actor].getTag('type') == 'ground_actor':
                self.camera.setY(self.camera.getY() + 30)
                self.camera.setZ(self.camera.getZ() + 5)
                self.camera.setHpr(180,-5,0)
            else:
                self.camera.setY(self.camera.getY() + 30)
                self.camera.setZ(self.camera.getZ() + 60)
                self.camera.setHpr(180,-60,0)
                # self.camera.lookAt(self.actors[n_actor])
    
    def changeCamera(self):
        if not hasattr(self, 'camera_PosHpr'):
            self.save_camera_position()
        self.camera_follow_actor = not self.camera_follow_actor
        if self.camera_follow_actor:
            self.nextActor()
        else:
            self.camera.reparentTo(render)
            self.camera.setPosHpr(*self.camera_PosHpr)
            
    def destroy(self):
        self.ignoreAll()
        self.camera = None
        self.actors = None
        self.xray_mode = None
        self.show_model_bounds = None
        self.camera_follow_actor = None
        self.camera_PosHpr = None
        self.camera_parent = None
        del self
        
    def setPfGider(self):
        if len(self.actors) < 6:
            self.pf_gideshow = not self.pf_gideshow
            for actor in self.actors:
                dj = actor.getPythonTag('subclass')
                dj.AIchar.setPfGuide(self.pf_gideshow)
            # render.ls()
        else:
            if self.current_actor is None :
                n_actor = np.random.randint(0, len(self.actors)-1)
                self.current_actor = self.actors[n_actor].getPythonTag('subclass')
                self.current_actor.AIchar.setPfGuide(True)
            else:
                self.current_actor.AIchar.setPfGuide(False)
                self.current_actor = None
        