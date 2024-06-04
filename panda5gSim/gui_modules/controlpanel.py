import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk


# source: https://www.blakerain.com/blog/overlays-with-custom-widgets-in-gtk
class ControlPanel(Gtk.Box):
    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        # Place the control panel in the top right
        # self.set_halign(Gtk.Align.END)
        self.set_halign(Gtk.Align.START)
        self.set_valign(Gtk.Align.START)
        
        

        # Add the .control-panel CSS class to this widget
        context = self.get_style_context()
        context.add_class("control-panel") 
        
    
    def add_group(self, group):
        # self.pack_start(group, False, False, 0)
        self.add(group)
        
    
