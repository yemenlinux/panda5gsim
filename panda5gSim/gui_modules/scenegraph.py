import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk

class SceneGraphReader(Gtk.VBox):
    def __init__(self, render):
        Gtk.VBox.__init__(self)
        self.render = render
        #
        #self.box = Gtk.VBox()
        # self.set_size_request(200, 200)
        # self.set_border_width(10)
        self.set_spacing(6)
        # self.set_homogeneous(False)
        self.set_valign(Gtk.Align.START)
        self.set_halign(Gtk.Align.START)
        
        # button
        self.read = Gtk.Button(label="Read Graph")
        self.read.connect("clicked", self.on_read_clicked)
        self.pack_start(self.read, False, False, 0)
        
        #
        self.text_view = Gtk.TextView()
        self.text_view.set_editable(False)
        self.text_view.set_cursor_visible(False)
        self.text_view.set_wrap_mode(Gtk.WrapMode.WORD)
        self.text_view.set_justification(Gtk.Justification.LEFT)
        self.text_view.set_left_margin(10)
        self.text_view.set_right_margin(10)
        self.text_view.set_pixels_above_lines(4)
        self.pack_start(self.text_view, True, True, 0)
        
        
        
    def on_read_clicked(self, widget):
        print("Read clicked")
        buffer = self.text_view.get_buffer()
        _iter = buffer.get_end_iter()
        for node_path in self.render.find_all_matches("**/*"):
            buffer.insert(_iter, node_path.get_name() + "\n")
            
        
        
