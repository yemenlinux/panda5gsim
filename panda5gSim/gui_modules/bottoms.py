import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
# import panda3d messenger

class ToggleButton(Gtk.ToggleButton):
    def __init__(self, off_label, on_label):
        Gtk.ToggleButton.__init__(self, label=off_label)
        self.off_label = off_label
        self.on_label = on_label
        self.connect("toggled", self.on_toggled)

    def on_toggled(self, button):
        if button.get_active():
            button.set_label(self.on_label)
        else:
            button.set_label(self.off_label)
            
class MobilityToggleButton(ToggleButton):
    def __init__(self, off_label, on_label):
        ToggleButton.__init__(self, off_label, on_label)
        
    
    def on_toggled(self, button):
        if button.get_active():
            button.set_label(self.on_label)
            # self.Mobility_generator.mobility_on = True
            messenger.send('move_ON', ['all'])
            messenger.send('Collect_Metrics_Start')
            messenger.send('Simulation_Start')
        else:
            button.set_label(self.off_label)
            # self.Mobility_generator.mobility_on = False
            messenger.send('move_OFF', ['all'])
            messenger.send('Collect_Metrics_Stop')
            messenger.send('Simulation_Stop')
            
class MobilityToggleButton_old(ToggleButton):
    def __init__(self, off_label, on_label, mobility_generator):
        ToggleButton.__init__(self, off_label, on_label)
        self.Mobility_generator = mobility_generator
        self.connect("toggled", self.on_toggled)
    
    def on_toggled(self, button):
        if button.get_active():
            button.set_label(self.on_label)
            self.Mobility_generator.mobility_on = True
        else:
            button.set_label(self.off_label)
            self.Mobility_generator.mobility_on = False