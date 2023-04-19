import bpy

def clear_compositor():
    scene = bpy.context.scene

    # reset nodes
    scene.use_nodes = True
    scene.node_tree.nodes.clear()
    scene.node_tree.links.clear()

    # reset view layer
    view_layer = scene.view_layers['ViewLayer']
    view_layer.use_pass_uv = False
    view_layer.use_pass_object_index = False
    view_layer.use_pass_emit = False

def setup_trivial_compositor():
    scene = bpy.context.scene

    # clear state
    clear_compositor()

    # setup nodes
    nodes = scene.node_tree.nodes
    output_node = nodes.new(type='CompositorNodeComposite')
    render_layers_node = nodes.new(type='CompositorNodeRLayers')

    # add linkes
    links = scene.node_tree.links
    links.new(render_layers_node.outputs['Image'], output_node.inputs['Image'])


def setup_uviz_compositor():
    scene = bpy.context.scene

    # clear state
    clear_compositor()

    # setup layers
    view_layer = scene.view_layers['ViewLayer']
    view_layer.use_pass_uv = True
    view_layer.use_pass_object_index = True
    view_layer.use_pass_emit = True
    # print("Dylan added to get z values as alpha channel!")
    view_layer.use_pass_z = True

    # create nodes
    nodes = scene.node_tree.nodes
    output_node = nodes.new(type='CompositorNodeComposite')
    render_layers_node = nodes.new(type='CompositorNodeRLayers')
    separate_rgba_node = nodes.new(type='CompositorNodeSepRGBA')
    combine_rgba_node = nodes.new(type='CompositorNodeCombRGBA')

    # add links
    links = scene.node_tree.links
    links.new(render_layers_node.outputs['UV'], separate_rgba_node.inputs[0])
    links.new(render_layers_node.outputs['Depth'], output_node.inputs['Alpha'])
    links.new(separate_rgba_node.outputs['R'], combine_rgba_node.inputs['R'])
    links.new(separate_rgba_node.outputs['G'], combine_rgba_node.inputs['G'])
    links.new(render_layers_node.outputs['IndexOB'], combine_rgba_node.inputs['B'])
    links.new(combine_rgba_node.outputs[0], output_node.inputs['Image'])

    # configure nodes
    output_node.use_alpha = True
