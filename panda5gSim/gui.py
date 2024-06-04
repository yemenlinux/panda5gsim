import sys

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

settings = Gtk.Settings.get_default()
settings.set_property("gtk-theme-name", "Numix")
settings.set_property("gtk-application-prefer-dark-theme", False)  # if you want use dark theme, set second arg to True

# Panda3d
# from panda3d.core import Messenger

# from panda5gSim.pluginmanager import get_plugin_lists
from panda5gSim.gui_modules.infoControlPanel import InfoControlPanel, TestControlPanel
from panda5gSim.gui_modules.notebook import NoteBook
from panda5gSim.gui_modules.treeview import ParamTreeView, ParamTreestore, MobilityParamTreeView
from panda5gSim.gui_modules.bottoms import ToggleButton, MobilityToggleButton




    
class Gui(Gtk.ApplicationWindow):
    def __init__(self):
        Gtk.ApplicationWindow.__init__(self)
        #
        self.set_title("Panda5G")
        self.top_vbox_size = 40
        self.bottom_vbox_size = 40
        self.setWindowSize(1280, 960)
        # externally used variables
        self.scenario = None
        self.network_device = None
        self.mobility_pattern = None
        # 
        self.make_layout()
        # read parameters
        self.update_info_notebook(SimData)
        # self.add_mobility_toggle(mobility_generator)
        
        
    
        # 
    def make_layout(self):
        """
            Make the layout of the GUI
        """
        # create a vertical box
        self.vpaned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        self.vpaned.set_position(self.top_vbox_size)
        self.add(self.vpaned)
        #
        self.vpaned2 = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        self.vpaned2.set_position(self.middle_height)
        #
        self.hpaned = Gtk.Paned()
        self.hpaned.set_position(self.right_panel_size)
        
        # create a horizontal box on top of the window
        self.top_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=3)
        # add status bar
        self.status_bar = Gtk.Statusbar()
        
        # add widget to paned panels
        self.vpaned.add1(self.top_hbox)
        self.vpaned.add2(self.vpaned2)
        self.vpaned2.add1(self.hpaned)
        self.vpaned2.add2(self.status_bar)
        
        # 
        # 
        # info notebook
        self.info_notebook = NoteBook()
        self.info_notebook.set_scrollable(True)
        self.hpaned.add1(self.info_notebook)
        
        # drawing area for banda
        self.drawing_notebook = NoteBook()
        self.drawing_notebook.set_scrollable(True)
        self.hpaned.add2(self.drawing_notebook)
        # add tab to the drawing notebook
        
        self.panda_drawing_area = Gtk.DrawingArea()
        # connect the signal to the handler
        # self.panda_drawing_area.connect("key-press-event", self.on_key_press_event)
        # make the drawing area able to receive keyboard events
        self.panda_drawing_area.set_can_focus(True)
        self.panda_drawing_area.grab_focus()
        # specify the types of events that the drawing area will receive
        self.panda_drawing_area.set_events(
            Gdk.EventMask.POINTER_MOTION_MASK | 
            Gdk.EventMask.BUTTON_PRESS_MASK | 
            Gdk.EventMask.BUTTON_RELEASE_MASK)
        self.hpaned.add2(self.panda_drawing_area)
        
        
        
        self.addNotebookTab(
            self.drawing_notebook, 
            "3D", 
            self.panda_drawing_area, tab_index=None)
        
        # make label for the scenario
        self.scenario_label = Gtk.Label("Scenario:")
        self.scenario_label.set_size_request(50, 20)
        self.make_scenario_combobox()
        self.mobility_label = Gtk.Label("Mobility mode:")
        self.scenario_label.set_size_request(50, 20)
        self.make_mobility_combobox()
        
        # add to the top_hbox
        self.top_hbox.pack_start(self.scenario_label, False, False, 0)
        self.top_hbox.pack_start(self.scenario_combobox, False, False, 0)
        self.top_hbox.pack_start(self.mobility_label, False, False, 0)
        self.top_hbox.pack_start(self.mobility_combobox, False, False, 0)
        
        # add toggle button
        self.add_mobility_toggle()
        #
        self.set_status_bar("Ready")
        
    def setWindowSize(self, width, height):
        # width = 1280
        # height = 960
        self.middle_height = height - self.top_vbox_size - self.bottom_vbox_size
        self.right_panel_size = 360
        self.set_default_size(width, height)
        self.set_size_request(width, height)
    
    def on_size_allocate(self, widget, allocation):
        height = self.get_allocated_height()
        self.middle_height = height - self.top_vbox_size - self.bottom_vbox_size
        self.vpaned.set_position(self.top_vbox_size)
        self.vpaned2.set_position(self.middle_height)
        
    def on_scenario_changed(self, combo):
        iter = combo.get_active_iter()
        if iter is not None:
            model = combo.get_model()
            print(f"Selected: {model[iter][0]}")
            #
            self.scenario = model[iter][0]
            # send message
            messenger.send("scenario_changed")
            
    def on_mobility_changed(self, combo):
        iter = combo.get_active_iter()
        if iter is not None:
            model = combo.get_model()
            print(f"Selected: {model[iter][0]}")
            # 
            self.mobility_pattern = model[iter][0]
    
    def create_list_store(self, data_types, data):
        """
            Create a ListStore
        """
        list_store = Gtk.ListStore()
        # check if data_types is a tuple or list
        if isinstance(data_types, tuple):
            list_store.set_column_types(data_types)
        elif isinstance(data_types, list):
            data_types = tuple(data_types)
            list_store.set_column_types(data_types)
        else:
            raise TypeError("data_types must be a tuple or list")
        # check if data is a list of tuples and tuples have the same length as data_types
        if isinstance(data, list):
            if len(data) == 0:
                return list_store
            else:
                if isinstance(data[0], tuple) and len(data[0]) == len(data_types):
                    for i in data:
                        list_store.append([j for j in i])
                    return list_store
                else:
                    raise TypeError("data must be a list of tuples")

    def create_combobox(self, list_store, entry_index_to_show = 0, column = 0):
        """
            Create a ComboBox
        """
        # check if list_store is a Gtk.ListStore
        if not isinstance(list_store, Gtk.ListStore):
            raise TypeError("list_store must be a Gtk.ListStore")
        #
        combobox = Gtk.ComboBox.new_with_model_and_entry(list_store)
        #self.scenario_combobox.connect("changed", self.onScenarioChanged)
        combobox.set_entry_text_column(entry_index_to_show)
        combobox.set_active(column)
        return combobox
    
        
    def set_status_bar(self, text):
        """
            Set the text of the status bar
        """
        context_id = self.status_bar.get_context_id("status")
        self.status_bar.push(context_id, text)
    
    def toggle_scenario_combobox_sensitivity(self):
        """
            Toggle the sensitivity of the scenario combobox
        """
        if self.scenario_combobox.get_sensitive():
            self.scenario_combobox.set_sensitive(False)
        else:
            self.scenario_combobox.set_sensitive(True)
    
    def make_scenario_combobox(self):
        self.scenario_list = [
            'Suburban',
            'Urban',
            'Dense Urban',
            'High-rise Urban',
        ] 
        self.scenarioDict = {
            'Suburban': 'UMa',
            'Urban': 'UMa',
            'Dense Urban': 'UMi',
            'High-rise Urban': 'UMi',
        }
        data_types = (str,)
        self.scenario_list_store = Gtk.ListStore()
        self.scenario_list_store.set_column_types(data_types)
        for i in self.scenario_list:
            self.scenario_list_store.append([i])
        # Create a ComboBox
        self.scenario_combobox = \
            Gtk.ComboBox.new_with_model_and_entry(self.scenario_list_store)
        self.scenario_combobox.connect("changed", self.on_scenario_changed)
        active_index = 0
        self.scenario_combobox.set_entry_text_column(0)
        self.scenario_combobox.set_active(active_index)
        self.scenario_combobox.set_size_request(50, 20)
        self.scenario_combobox.set_can_focus(False)
        self.scenario = self.scenario_list_store[active_index][0]
        
    def setScenario(self, scenario):
        """
            Set the scenario
        """
        self.scenario = scenario
        self.scenario_combobox.set_active(self.scenario_list.index(scenario))
        
    def make_mobility_combobox(self):
        # create combo box for mobility patterns
        self.mobility_patterns = [
            'Random PoIs', 
            'Random Walk', 
            'Random Waypoints', 
            'Tidal'
        ]
        data_types = (str,)
        self.mobility_list_store = Gtk.ListStore()
        self.mobility_list_store.set_column_types(data_types)
        for i in self.mobility_patterns:
            self.mobility_list_store.append([i])
        self.mobility_combobox = Gtk.ComboBox.new_with_model_and_entry(self.mobility_list_store)
        self.mobility_combobox.connect("changed", self.on_mobility_changed)
        self.mobility_combobox.set_entry_text_column(0)
        self.mobility_combobox.set_active(0)
        # self.grid.attach(self.mobility_combobox, 1, 0, 1, 1)
        self.mobility_pattern = self.mobility_list_store[0][0]
        # set size of the combo box
        self.mobility_combobox.set_size_request(70, 30)
    
    def add_notebook_tab_from_dict(self, tab_name, params):
        """
            Make a tab in the notebook for the given tab name and a dictionary of parameters.
        """
        self.info_controls = ParamTreeView(ParamTreestore(params))
        self.add_notebook_tab(tab_name, self.info_controls)
    
    def add_notebook_tab_for_mobility(self, tab_name, params, mobility_generator):
        """
            Make a tab in the notebook for the given tab name and a dictionary of parameters.
        """
        mobility_TreeView = MobilityParamTreeView(ParamTreestore(params), mobility_generator)
        self.add_notebook_tab(tab_name, mobility_TreeView)
    
    def update_info_notebook(self, scenario_parameters):
        """
            Update the info notebook
        """
        self.info_controls = ParamTreeView(ParamTreestore(scenario_parameters))
        #self.info_notebook.add_scrollable_tab('AiWorld', self.info_controls, 0)
        self.add_notebook_tab('Parameters', self.info_controls)
        
        # mobility panel
        mobility_dict = {
            'Outdoor Percentage': 0.2,
            'Walking Speed (m/s)': 0.83,
            'Car Speed (m/s)': 19.4,
            'Bullet Train Speed (m/s)': 83.3,
            
        }
        self.mobility_param = ParamTreestore(mobility_dict)
        self.mobility_panel = ParamTreeView(self.mobility_param)
        #self.info_notebook.add_scrollable_tab('Mobility', self.mobility_panel, 1)
        self.add_notebook_tab('Mobility', self.mobility_panel)
        
    def addNotebookTab(self, notebook, tab_name, tab_widget, tab_index=None):
        """
            Add a tab to the info notebook by giving a Gtk.TreeStore
        """
        #get the length of the notebook
        if tab_index == None:
            tab_index = notebook.get_n_pages()
        notebook.add_scrollable_tab(tab_name, tab_widget, tab_index)
        
        
    def add_notebook_tab(self, tab_name, tab_widget, tab_index=None):
        """
            Add a tab to the info notebook by giving a Gtk.TreeStore
        """
        #get the length of the notebook
        if tab_index == None:
            tab_index = self.info_notebook.get_n_pages()
        self.info_notebook.add_scrollable_tab(tab_name, tab_widget, tab_index)
        
    def show_panel(self, tab_name):
        """
            Show a panel in the info notebook
        """
        self.info_notebook.set_current_page(self.info_notebook.get_tab_position(tab_name))
        
        
    def add_mobility_toggle(self):
        """
            Add a toggle button to the info notebook
        """
        # add toggle button
        self.mobility_toggle_button = MobilityToggleButton('Start', 'Stop')
        self.mobility_toggle_button.set_size_request(50, 20)
        self.top_hbox.pack_start(self.mobility_toggle_button, False, False, 0)
        self.mobility_toggle_button.show()
        
    def on_key_press_event(self, widget, event):
        # check if the key pressed is an arrow key
        if event.keyval == Gdk.KEY_Left:
            print("Left arrow key pressed")
            messenger.send("arrow_left")
        elif event.keyval == Gdk.KEY_Right:
            print("Right arrow key pressed")
            messenger.send("arrow_right")
        elif event.keyval == Gdk.KEY_Up:
            print("Up arrow key pressed")
            messenger.send("arrow_up")
        elif event.keyval == Gdk.KEY_Down:
            print("Down arrow key pressed")
            messenger.send("arrow_down")
        # c
        elif event.keyval == Gdk.KEY_c:
            print("c key pressed")
            messenger.send("c")
        
        

