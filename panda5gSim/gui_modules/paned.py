import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk


class Hpaned(Gtk.Paned):
    def __init__(self, breakpoint):
        """ Add a horizontal Gtk.Paned to the parent window

        Args:
            breakpoint (int): a point to set the paned position
            
        Usages:
            # add a horizontal paned
            hpaned = Hpaned(360)
            # add a widget to the left pane
            hpaned.add1(widget)
            # add a widget to the right pane
            hpaned.add2(widget)
        """
        Gtk.Paned.__init__(self)
        self.set_position(breakpoint)
        self.set_orientation(Gtk.Orientation.HORIZONTAL)
        
    def add1(self, widget):
        self.add1(widget)
    
    def add2(self, widget):
        self.add2(widget)

def make_HPaned(parent_window, breakpoint):
    hpaned = Gtk.Paned()
    hpaned.set_position(360)