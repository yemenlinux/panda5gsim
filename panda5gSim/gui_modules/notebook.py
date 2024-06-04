import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

def construct_tab(notebook, tab_label, child_widget, tab_position):
    # (self, child:Gtk.Widget, tab_label:Gtk.Widget=None, position:int) -> int
    if not isinstance(child_widget, Gtk.Widget):
        raise TypeError("child_widget must be a Gtk.Widget")
    if isinstance(tab_label, str):
        tab_label = Gtk.Label(label=tab_label)
    notebook.insert_page(child_widget, tab_label, tab_position)
    notebook.set_tab_reorderable(child_widget, True)
    # notebook.set_tab_detachable(tab_title, True)
    tab_label.show()


def make_parameter_notebooks(notebook, parameters_dict):
    """ read a dictionary of dictionaries and show them in Gui

    Args:
        notebook (Gtk.Notebook): _description_
        parameters_dict (dict): _description_
    """
    paramlen = len(parameters_dict)
    if 'General' not in parameters_dict:
        param = {'general': {}}
    else:
        param = {}
    param.update(parameters_dict)
    for i, (key, value) in enumerate(param.items()):
        if isinstance(value, dict):
            make_expandable(tab_label, key, value)
            

class NoteBook(Gtk.Notebook):
    def __init__(self):
        Gtk.Notebook.__init__(self)
        self.set_scrollable(True)
        self.set_show_tabs(True)
        self.set_show_border(True)
        self.set_tab_pos(Gtk.PositionType.TOP)
        # self.set_tab_detachable(True)
        # self.set_tab_reorderable(True)
        self.set_scrollable(True)
        self.set_sensitive(True)
        
    def add_tab(self, tab_label, child_widget, tab_position):
        # (self, child:Gtk.Widget, tab_label:Gtk.Widget=None, position:int) -> int
        if not isinstance(child_widget, Gtk.Widget):
            raise TypeError("child_widget must be a Gtk.Widget")
        if isinstance(tab_label, str):
            tab_label = Gtk.Label(label=tab_label)
        self.insert_page(child_widget, tab_label, tab_position)
        # self.insert_page(tab_label, child_widget, tab_position)
        self.set_tab_reorderable(child_widget, True)
        self.show_all()
        
    def remove_tab(self, tab_position):
        self.remove_page(tab_position)
        
    def add_scrollable_tab(self, tab_label, child_widget, tab_position):
        # (self, child:Gtk.Widget, tab_label:Gtk.Widget=None, position:int) -> int
        if self.is_exist(tab_label):
            pos = self.get_tab_position(tab_label)
            # print(f'{tab_label} is already exist at {pos}')
            self.remove_tab(pos)
            
        if not isinstance(child_widget, Gtk.Widget):
            raise TypeError("child_widget must be a Gtk.Widget")
        if isinstance(tab_label, str):
            tab_label = Gtk.Label(label=tab_label)
        scrollable = Gtk.ScrolledWindow()
        scrollable.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrollable.add(child_widget)
        
        self.insert_page(scrollable, tab_label, tab_position)
        # self.insert_page(tab_label, child_widget, tab_position)
        self.set_tab_reorderable(scrollable, True)
        self.show_all()
        
    def is_exist(self, tab_label):
        # return True if tab_label is in the notebook
        for i in range(self.get_n_pages()):
            if self.get_tab_label_text(self.get_nth_page(i)) == tab_label:
                return True
        return False
    
    def get_tab_position(self, tab_label):
        # return the tab position of tab_label
        for i in range(self.get_n_pages()):
            if self.get_tab_label_text(self.get_nth_page(i)) == tab_label:
                return i
        return None

    