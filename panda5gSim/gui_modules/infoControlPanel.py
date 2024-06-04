import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

from panda5gSim.gui_modules.ControlPanelGroup import ControlPanelGroup
from panda5gSim.gui_modules.controlpanel import ControlPanel

class TestControlPanel(ControlPanel):
    def __init__(self):
        super().__init__()
        self._first_panel = ControlPanelGroup("Some Buttons")
        self._first_panel.add_row(Gtk.Button(label="Button 1"))
        self._first_panel.add_row(Gtk.Button(label="Button 2"))
        self.add_group(self._first_panel)
        self._second_panel = ControlPanelGroup("Extra Settings")
        self._second_panel.add_row(Gtk.Button(label="Button 3"))
        self._second_panel.add_row(Gtk.Button(label="Button 4"))
        self._second_panel.add_row(
            Gtk.CheckButton.new_with_label("First checkbox"))
        self._second_panel.add_row(
            Gtk.CheckButton.new_with_label("Second checkbox"))
        combo = Gtk.ComboBoxText()
        combo.append("first", "First Choice")
        combo.append("second", "Second Choice")
        combo.append("third", "Third Choice")
        combo.append("forth", "This one is quite long")
        combo.set_active_id("first")
        self._second_panel.add_row(combo)
        self.add_group(self._second_panel) 
        
    

class InfoControlPanel(ControlPanel):
    def __init__(self, module, param):
        super().__init__()

        self.module = module
        if 'General' not in param:
            self.param = {'General': {}}
        else:
            self.param = {}
        for k, v in param.items():
            if isinstance(v, dict):
                # capitalize the first letter of the key
                self.param[k.capitalize()] = v
            else:
                self.param['General'][k] = v
        
        self.add_control_groups()
        
        
    def add_attribute(self, key, value):
        # add a Gtk.Box with a Gtk.Label and a Gtk.Entry
        label = Gtk.Label(label=key.capitalize())
        entry = Gtk.Entry()
        entry.set_text(str(value))
        entry.set_editable(True)
        #
        entry.connect("changed", self.on_entry_changed)
        # horizontal box
        # box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        # box.pack_start(label, False, False, 0)
        # box.pack_start(entry, False, False, 0)
        # vertical box
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        box.add(label)
        box.add(entry)
        
        return box
        
    def add_control_groups(self):
        for k, v in self.param.items():
            self.add_control_group(k, v)
            # group = ControlPanelGroup(k)
            # for k2, v2 in v.items():
            #     box = self.add_attribute(k2, v2)
            #     group.add_row(box)
            # self.add_group(group)
        
        
    def add_control_group(self, name, attr_dict):
        group = ControlPanelGroup(name)
        for k, v in attr_dict.items():
            box = self.add_attribute(k, v)
            group.add_row(box)
        self.add_group(group)
        
    def on_entry_changed(self, entry):
        # print(entry.get_text())
        # print(entry.get_parent())
        # print(entry.get_parent().get_children())
        # print(entry.
        pass
        
