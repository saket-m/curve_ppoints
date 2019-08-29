import nuke
import _curveknob

def create_node(node_name):
    '''
    Create a node in a global graph given the node name.
    If the name of the node is wrong, it will not create anything.

    Argument ::
        node_name -- str | name of the node
    '''
    nuke.createNode(node_name)

def get_RotoPaint(node_name = None ,connect_to_selected = False):
    '''
    Creates a RotoPaint node and returns its corresponding object
    to the console

    Argument ::
        node_name -- str | default value is '' | if specified assigns a name to
                     the node
        connect_to_selected -- boolean | deault value is False |
                               if set Ture connects the created node to the
                               selected node in the global graph area.
    Return ::
        rp -- RotoPaint node object
    '''
    if node_name == None:
        rp = nuke.nodes.RotoPaint()
    else:
        rp = nuke.nodes.RotoPaint(name=node_name)

    if connect_to_selected == True :
        rp.setInput(0, nuke.selectedNode())

    return rp
            
def get_Roto(node_name = None, connect_to_selected = False):
    '''
    Creates a Roto node and returns its corresponding object
    to the console

    Argument ::
        node_name -- str | default value is '' | if specified assigns a name to
                     the node
        connect_to_selected -- boolean | deault value is False |
                               if set Ture connects the created node to the
                               selected node in the global graph area.
    Return ::
        r -- RotoPaint node object
    '''
    if node_name == None:
        r = nuke.nodes.Roto()
    else:
        r = nuke.nodes.Roto(name=node_name)

    if connect_to_selected == True :
        r.setInput(0, nuke.selectedNode())

    return r

def get_Read(file_path, node_name = None, connect_to_selected = False):
    '''
    Creates a Rot node and returns its corresponding object
    to the console

    Argument ::
        file-path -- str | file name to be read
        node_name -- str | default value is None | if specified assigns a name to
                     the node
        connect_to_selected -- boolean | deault value is False |
                               if set Ture connects the created node to the
                               selected node in the global graph area.
    Return ::
        r -- RotoPaint node object
    '''
    if node_name == None:
        r = nuke.nodes.Read(file=file_path)
    else:
        r = nuke.nodes.Roto(file=file_path, name=node_name)

    if connect_to_selected == True :
        r.setInput(0, nuke.selectedNode())

    return r

def get_Node_by_name(node_name):
    '''
    Returns the object of the node by the name specified.

    Arguments ::
        node_name -- str | name of a node e.g. Read1
    Return ::
        n -- object of the node
    '''
    n = nuke.toNode(node_name)
    return n

def show_proporties(node=None):
    '''
    Displays all the proporties of the node.

    Argument ::
        node -- node object
    '''
    for i in range (node.getNumKnobs()):
        print node.knob (i).name()

def get_Roto(node_name = '', connect_to_selected = False):
    '''
    Creates a Rot node and returns its corresponding object
    to the console

    Argument ::
        node_name -- str | default value is '' | if specified assigns a name to
                     the node
        connect_to_selected -- boolean | deault value is False |
                               if set Ture connects the created node to the
                               selected node in the global graph area.
    Return ::
        r -- RotoPaint node object
    '''
    if node_name == '':
        r = nuke.nodes.Roto()
    else:
        r = nuke.nodes.Roto(node_name)

    if connect_to_selected == True :
        r.setInput(0, nuke.selectedNode())

    return r

def get_spline(roto_node_name, mask_dir):
    '''
    Generates a spline given a mask number.

    Arguments ::
        roto_node_name -- str | name of the roto node
        mask_dir -- str | directory where the mask is stored
    '''
    roto = get_Node_by_name('Roto1')

    shapeName = 'Bezier1'
    cv = 0

    curveKnob = roto['curves']
    
    strokes = mask_dir

    vertices = []
    with open(strokes	, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.split('\t')
            vertices.append((int(l[0]), int(l[1][:-1])))

    emptyShape = _curveknob.Shape( curveKnob )
    maskShape = _curveknob.Shape( curveKnob, *vertices, type='bspline' )
    nuke.message('Spline Drawn')

def draw_bezier():
    '''
    '''
    roto_node_name = nuke.getInput('Roto node name', 'Roto1')
    mask_dir = nuke.getFilename('Choose Bezier file', '*.bezier')
    get_spline(roto_node_name, mask_dir)
    
