import bpy
import pathlib
import numpy as np

def get_world_material(name='World'):
    world_materail = bpy.data.worlds[name]
    return world_materail

def require_material(name):
    if name not in bpy.data.materials:
        bpy.data.materials.new(name)
    material = bpy.data.materials[name]
    return material

def clear_materials():
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)

def clear_nodes(material):
    material.node_tree.nodes.clear()
    material.node_tree.links.clear()

def setup_black_world_material(material):
    material.use_nodes = True
    clear_nodes(material)

    # create nodes
    nodes = material.node_tree.nodes
    output_node = nodes.new(type='ShaderNodeOutputWorld')
    background_node = nodes.new(type='ShaderNodeBackground')
    # connect nodes
    links = material.node_tree.links
    links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
    # modify values
    background_node.inputs['Strength'].default_value = 0

def setup_hdri_world_material(material, hdri_path="data/hdri/studio.exr"):
    # setup texture image
    image_path = pathlib.Path(hdri_path)
    assert(image_path.exists())
    if image_path.name not in bpy.data.images:
        bpy.ops.image.open(filepath=str(image_path.absolute()))
    hdri_iamge = bpy.data.images[image_path.name]

    material.use_nodes = True
    clear_nodes(material)

    # create nodes
    nodes = material.node_tree.nodes
    output_node = nodes.new(type='ShaderNodeOutputWorld')
    background_node = nodes.new(type='ShaderNodeBackground')
    texture_node = material.node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    # connect nodes
    links = material.node_tree.links
    links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
    links.new(texture_node.outputs['Color'], background_node.inputs['Color'])
    # modify values
    texture_node.image = hdri_iamge
    texture_node.interpolation = 'Cubic'
    texture_node.projection = 'EQUIRECTANGULAR'


def setup_uv_material(material):
    material.use_nodes = True
    clear_nodes(material)

    # create nodes
    nodes = material.node_tree.nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    emission_node = nodes.new(type='ShaderNodeEmission')
    uv_coord_node = nodes.new(type='ShaderNodeTexCoord')
    vector_math_node = nodes.new(type='ShaderNodeVectorMath')
    # connect nodes
    links = material.node_tree.links
    links.new(uv_coord_node.outputs['UV'], vector_math_node.inputs[0])
    links.new(vector_math_node.outputs[0], emission_node.inputs['Color'])
    links.new(emission_node.outputs[0], output_node.inputs['Surface'])
    # set values
    vector_math_node.operation = 'MULTIPLY'
    vector_math_node.inputs[1].default_value = (1, 1, 0)

def setup_white_material(material, roughness=1.0):
    material.use_nodes = True
    clear_nodes(material)

    # create nodes
    nodes = material.node_tree.nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    # connect nodes
    links = material.node_tree.links
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    # set values
    principled_node.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    principled_node.inputs['Roughness'].default_value = roughness

def setup_metal_materail(material, roughness=0.2, base_color=(0.6, 0.6, 0.1, 1.0)):
    material.use_nodes = True
    clear_nodes(material)

    # create nodes
    nodes = material.node_tree.nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    # connect nodes
    links = material.node_tree.links
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    # set values
    principled_node.inputs['Base Color'].default_value = base_color
    principled_node.inputs['Metallic'].default_value = 1.0
    principled_node.inputs['Roughness'].default_value = roughness

def setup_plastic_material(material, roughness=0.6, base_color=(0.07, 0.03, 0.8, 1.0)):
    material.use_nodes = True
    clear_nodes(material)

    # create nodes
    nodes = material.node_tree.nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    # connect nodes
    links = material.node_tree.links
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    # set values
    principled_node.inputs['Base Color'].default_value = base_color
    principled_node.inputs['Metallic'].default_value = 0.0
    principled_node.inputs['Roughness'].default_value = roughness

def get_principled_bsdf_config(material):
    node = material.node_tree.nodes['Principled BSDF']
    config = dict()
    for key, value in node.inputs.items():
        input_value = value.default_value
        if isinstance(input_value, bpy.types.bpy_prop_array):
            input_value = list(input_value)
        config[key] = input_value
    return config

def get_default_principled_bsdf_config():
    config = {
        'Base Color': [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0],
        'Subsurface': 0.0,
        'Subsurface Radius': [1.0, 0.20000000298023224, 0.10000000149011612],
        'Subsurface Color': [0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0],
        'Metallic': 0.0,
        'Specular': 0.5,
        'Specular Tint': 0.0,
        'Roughness': 0.5,
        'Anisotropic': 0.0,
        'Anisotropic Rotation': 0.0,
        'Sheen': 0.0,
        'Sheen Tint': 0.5,
        'Clearcoat': 0.0,
        'Clearcoat Roughness': 0.029999999329447746,
        'IOR': 1.4500000476837158,
        'Transmission': 0.0,
        'Transmission Roughness': 0.0,
        'Emission': [0.0, 0.0, 0.0, 1.0],
        'Alpha': 1.0,
        'Normal': [0.0, 0.0, 0.0],
        'Clearcoat Normal': [0.0, 0.0, 0.0],
        'Tangent': [0.0, 0.0, 0.0]
    }
    return config

def setup_principled_bsdf_material(
    material, 
    config=get_default_principled_bsdf_config()):
    material.use_nodes = True
    clear_nodes(material)

    # create nodes
    nodes = material.node_tree.nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    # connect nodes
    links = material.node_tree.links
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    # set values
    inputs = principled_node.inputs
    for key, value in config.items():
        inputs[key].default_value = value

def setup_textured_bsdf_material(
    material,
    image):
    material.use_nodes = True
    clear_nodes(material)

    # create nodes
    nodes = material.node_tree.nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    texture_node = nodes.new(type="ShaderNodeTexImage")
    # connect nodes
    links = material.node_tree.links
    links.new(texture_node.outputs['Color'], principled_node.inputs['Base Color'])
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    # set values
    texture_node.image = image
    principled_node.inputs['Specular'].default_value = 0.2

def require_image(image_name, img):
    options = {
        'width': img.shape[1],
        'height': img.shape[0],
        'alpha': True,
        'float_buffer': True
    }
    float_img = img
    if img.dtype == np.uint8:
        float_img = img.astype(np.float32) / 255

    alpha_img = float_img
    if float_img.shape[2] == 3:
        alpha_img = np.concatenate(
            [float_img, np.ones(float_img.shape[:-1] + (1,), float_img.dtype)], axis=2)

    if image_name not in bpy.data.images:
        bpy.data.images.new(image_name, **options)
    else:
        bpy.data.images.remove(bpy.data.images[image_name])
        bpy.data.images.new(image_name, **options)
    image = bpy.data.images[image_name]
    image.pixels = alpha_img.ravel().tolist()
    return bpy.data.images[image_name]
