import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk


# source: https://www.blakerain.com/blog/overlays-with-custom-widgets-in-gtk
class ControlPanelGroup(Gtk.Expander):
    def __init__(self, title: str):
        super().__init__(label=title)
        self._inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL,
                            spacing=5)

        # Set the size request to 200 pixels wide
        self.set_size_request(200, -1)

        self.add(self._inner)

    def add_row(self, widget):
        self._inner.pack_start(widget, False, False, 0)
