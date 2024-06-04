import sys
import os
import pdb

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
sys.path.append(PARENT_DIR)

from panda5gSim.gui import Gui
from panda5gSim.simManager import SimManager


settings = Gtk.Settings.get_default()
settings.set_property("gtk-theme-name", "Numix")
settings.set_property("gtk-application-prefer-dark-theme", False)  # if you want use dark theme, set second arg to True



def install_css():
    screen = Gdk.Screen.get_default()
    provider = Gtk.CssProvider()
    provider.load_from_data(b"""
    .control-panel {
        background-color: @bg_color;
        padding: 4px;
        border-bottom-left-radius: 4px;
    }
    """)
    Gtk.StyleContext.add_provider_for_screen(screen, provider, 600)

# Function to call when program is supposed to quit
def close_app(*args, **kw):
    # 
    # Gtk.main_quit(*args, **kw)
    # base.destroy()
    sys.exit(0)

# simulation function
# def Simulate():
#     pass
    

# class Panda5gSimApp(Gtk.Application):
#     def __init__(self):
#         Gtk.Application.__init__(self, 
#                                 application_id="Panda5gSim",
#                                 )
        

#     def do_activate(self):
#         win = Gui(self)
#         win.set_title("Main Application Window")
#         win.connect("destroy", close_app)
#         win.present()
        
    

# app = Panda5gSimApp()
# exit_status = app.run([])
# sys.exit(exit_status)

# if __name__ == "__main__":
#     app = Panda5gSimApp()
#     app.run(sys.argv)

if __name__ == "__main__":
    gui = Gui()
    gui.connect("destroy", close_app)
    gui.show_all()
    
    # start simulation
    app = SimManager(gui)
    # app.setSimBounds((-500, -500, 500, 500))
    # app.loadTerrain('/doc/code/MyPhdCode/pysimuav/panda5gSimv1_0/assets/textures/street_map.jpg')
    # app.loadTerrain()
    # generate street map
    # app.genStreetMap()
    # # generate Buildings
    # app.genBuildings()
    # # generate ground users
    # app.genGroundUsers()
    # # generate air users
    # app.genAirUsers()
    # # generate airBS
    # app.genAirBS()
    # # pdb.set_trace()
    # app.setAI()
    
    # Generate Environment
    app.genEnvironment()
    # app.collectRLP()
    # app.collect_P_LoS()
    app.run()
    # simulate
    
            
    
    
    
    
    
    

