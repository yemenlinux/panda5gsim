import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
from gi.repository import GObject

            
class ParamTreestore(Gtk.TreeStore):
    def __init__(self, data_dict):
        super().__init__(str, str)
        self.data_dict = data_dict
        parent = None
        self.add_dict(parent, self.data_dict)
    
    def add_dict(self, parent, data_dict):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                child_iter = self.append(parent, [key, None])
                self.add_dict(child_iter, value)
            else:
                if not isinstance(value, str):
                    value = str(value)
                self.append(parent, [key, value])
    

class ParamTreeView(Gtk.TreeView):
    def __init__(self, model):
        self.model = model
        Gtk.TreeView.__init__(self)
        self.set_model(self.model)
        self.set_show_expanders(True)
        #self.set_sensitive(True)

        # Make the value cells editable
        for i in range(model.get_n_columns()):
            renderer = Gtk.CellRendererText()
            if i == 0:
                renderer.set_property("editable", False)
                #renderer.connect("edited", self.on_cell_in_column_edited, i)
                #renderer.connect("edited", self.on_cell_edited)
                column = Gtk.TreeViewColumn("Parameter", renderer, text=i)
                #column.set_cell_data_func(renderer, self.cell_data_func)
            else:
                renderer.set_property("editable", True)
                #renderer.connect("edited", self.on_cell_in_column_edited, i)
                renderer.connect("edited", self.on_cell_edited)
                # on renderer cell selection 
                #renderer.connect("editing-started", self.on_editing_started)
                column = Gtk.TreeViewColumn("Value", renderer, text=i)
                #column.set_cell_data_func(renderer, self.cell_data_func)
                
            
            #
            self.append_column(column)
            self.set_expander_column()
    
    def on_cell_edited(self, widget, path, new_text):
        _iter = self.get_model().get_iter_from_string(path)
        self.get_model().set(_iter, 1, new_text)
        # update the model data_dict
        self.get_model().data_dict = self.get_data_dict()
        #data_dict = self.get_data_dict()
        # print(data_dict)
        
    def on_editing_started(self, widget, editable, path):
        # set the cursor to the widget
        print("on_editing_started")
        print(f'type widget: {widget}, editable: {editable}, path: {path}')
        widget.set_sensitive(True)
        widget.set_property("editable", True)
        # enable parent Gtk.TreeView
        #self.set_sensitive(True)
        

    def on_cell_in_column_edited(self, renderer, path, new_text, column):
        _iter = self.get_model().get_iter(path)
        key = list(self.get_model().data_dict.keys())[column]
        _iter.user_data[key] = new_text
        self.get_model().row_changed(self.get_model().get_path(iter), iter)
        
    def get_data_dict(self):
        #self.get_model().foreach(self.print_row, None)
        treestore = self.get_model()
        # Export the TreeStore to a dictionary
        data_dict = {}
        iter = treestore.get_iter_first()
        while iter is not None:
            key = treestore.get_value(iter, 0)
            value = treestore.get_value(iter, 1)
            if treestore.iter_has_child(iter):
                data_dict[key] = self.export_tree_store_to_dict(treestore, iter, data_dict)
            else:
                data_dict[key] = value
            iter = treestore.iter_next(iter)
        return data_dict
        
            
    # Define a function to get the key, value of each row
    def print_row(self, model, path, iter, data):
        #usage:
        # Iterate over all rows of the TreeStore and get the key, value of each row
        #treestore.foreach(self.print_row, None)
        key = model.get_value(iter, 0)
        value = model.get_value(iter, 1)
        print(path, iter, key, value)
        

    # Define a function to export the TreeStore to a dictionary
    def export_tree_store_to_dict(self, model, iter, data):
        if model.iter_has_child(iter):
            child_dict = {}
            child_iter = model.iter_children(iter)
            while child_iter is not None:
                child_key = model.get_value(child_iter, 0)
                child_value = model.get_value(child_iter, 1)
                if model.iter_has_child(child_iter):
                    child_dict[child_key] = self.export_tree_store_to_dict(model, child_iter, data)
                else:
                    child_dict[child_key] = child_value
                child_iter = model.iter_next(child_iter)
            return child_dict
        else:
            key = model.get_value(iter, 0)
            value = model.get_value(iter, 1)
            return {key: value}
        
    # Set a function to be called for each cell in the first column
    def cell_data_func(self, column, cell, model, _iter, data):
        if model.iter_has_child(_iter):
            cell.set_property('text', 'Parent')
            cell.set_property('weight', 700)
            cell.set_property('foreground', 'blue')
            cell.set_property('underline', Pango.Underline.SINGLE)
        else:
            cell.set_property('text', model.get_value(iter, 0))
            cell.set_property('weight', 400)
            cell.set_property('foreground', 'black')
            cell.set_property('underline', Pango.Underline.NONE)

        
        
class MobilityParamTreeView(ParamTreeView):
    def __init__(self, model, mobility_manager):
        super().__init__(model)
        self.mobility_manager = mobility_manager
        
    def on_cell_edited(self, widget, path, new_text):
        super().on_cell_edited(widget, path, new_text)
        # update the mobility_manager
        self.mobility_manager.param = self.get_data_dict()
    