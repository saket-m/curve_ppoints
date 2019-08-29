from myeNutils import *

toolbar = nuke.menu( 'Nodes' )
m = toolbar.addMenu('Myelin Spline')
m.addCommand( 'Myelin/Roto', "draw_bezier()", 'a')
