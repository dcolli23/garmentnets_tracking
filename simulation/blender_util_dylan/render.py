from pathlib import Path
import os

import bpy
import numpy as np
import pickle

from simulation.blender_util.render import get_renderer_config, get_default_cycles_config
from simulation.blender_util.camera import (
    require_camera, 
    get_camera_extrinsic,
    set_camera_extrinsic,
    get_camera_intrinsic, 
    set_camera_intrinsic,
    set_camera_focus_point,
    generate_intrinsic
)
from simulation.blender_util.physics import require_subdiv_modifier
from simulation.blender_util.material import (
    require_material,
    clear_materials,
    get_world_material,
    setup_black_world_material,
    setup_hdri_world_material,
    setup_metal_materail, 
    setup_plastic_material, 
    setup_uv_material, 
    setup_white_material,
    setup_textured_bsdf_material,
    require_image
)
from simulation.blender_util.render import (
    setup_color_management_raw,
    setup_color_management_srgb,
    setup_cycles,
    setup_eevee,
    get_cycles_uviz_config,
    get_eevee_rgb_config,
    get_cycles_rgb_config,
    setup_png_output,
    setup_exr_output,
    render_animation
)
from simulation.cloth_3d_util.accessors.access_functions import (
    get_info, get_garment_texture, get_garment_metadata
)
from simulation.blender_util.mesh import (NumpyBMeshManager, require_mesh, set_material)
from simulation.blender_util.compositor import (
    setup_trivial_compositor, 
    setup_uviz_compositor
)
from simulation.blender_util.collection import remove_all_collections

ROOT = os.path.dirname(__file__)
HDRI_PATH = os.path.join(ROOT, "..", "data", "hdri", "studio.exr")

def enable_gpu_renders():
    # enable GPU
    cprefs = bpy.context.preferences.addons['cycles'].preferences
    cprefs.compute_device_type = 'CUDA'
    cprefs.get_devices()
    for device in cprefs.devices:
        if device.type == 'CUDA':
            device.use = True
        else:
            device.use = False
        print(device.id, device.use, device.type)

def render_dylan(output_path, sample_id, garment_name, gender, fabric, garment_verts, garment_faces,
                 garment_uv_verts, garment_uv_faces, garment_texture, num_camera_angles,
                 camera_intrinsic, render_animation=False, z_offset=-0.8):
    # NOTE: Assuming we're starting from a saved checkpoint instead of using Cheng's BMesh 
    # checkpointing.

    # Grab the cloth object
    cloth_obj = bpy.data.objects['cloth']

    # Setup the camera.
    camera_obj = require_camera()
    set_camera_intrinsic(camera_obj, camera_intrinsic)

    z_offset = z_offset
    radius = 4.0
    focus_point = np.array([0,0,z_offset])
    camera_rads = np.linspace(0, 2*np.pi, num=num_camera_angles+1)[:-1]
    # TODO: I thought Cheng said he used 4 angles, not 3?
    camera_locations = np.zeros((num_camera_angles, 3))
    camera_locations[:, 0] = np.cos(camera_rads)
    camera_locations[:, 1] = np.sin(camera_rads)
    camera_locations *= radius
    camera_locations[:, 2] = z_offset

    camera_extrinsic_list = list()
    for i in range(len(camera_locations)):
        camera_obj.location = camera_locations[i]
        set_camera_focus_point(camera_obj, focus_point)
        extrinsic = get_camera_extrinsic(camera_obj)
        camera_extrinsic_list.append(extrinsic)

    # generate output filenames
    output_dir = Path(output_path)
    pickle_path = str(output_dir.joinpath('meta.pk').absolute())
    uviz_paths = list()
    rgb_paths = list()
    for i in range(num_camera_angles):
        uviz_paths.append(str(output_dir.joinpath(
            'uviz_{}.exr'.format(i)).absolute()))
        rgb_paths.append(str(output_dir.joinpath(
            'rgb_{}.png'.format(i)).absolute()))
    # curr_s = time.perf_counter()
    # print("Setup: {}".format(curr_s - s))
    # s = curr_s

    # setup modifiers for rendering
    require_subdiv_modifier(cloth_obj)

    clear_materials()

    uv_material = require_material('uv')
    setup_uv_material(uv_material)

    image = require_image('texture.png', garment_texture)
    cloth_material = require_material('rgb')
    setup_textured_bsdf_material(cloth_material, image)

    # checkpointer.save_checkpoint_if_desired()

    ## Render Images

    

    # print(bpy.context.scene.view_layers.keys())

    # # The active scene is the only scene available so this shouldn't be the problem.
    # print(bpy.context.scene)
    # print(bpy.data.scenes.keys())

    # print("Render FPS:", bpy.context.scene.render.fps)
    # print("Render FPS base:", bpy.context.scene.render.fps_base)

    # print("Resolution x:,", bpy.context.scene.render.resolution_x)
    # print("Resolution y:,", bpy.context.scene.render.resolution_y)

    # default_render_config_blender = get_renderer_config(bpy.context.scene.render)
    # print("Default Blender Render Config:")
    # for k, v in default_render_config_blender.items():
    #     print(f"\t{k} = {v}")



    # default_render_config_cheng = get_default_cycles_config()
    # print("Default render config:")
    # for k, v in default_render_config_blender.items():
    #     print(f"\t{k} = {v}")

    # settings_missing_from_cheng_config = set(default_render_config_blender) - set(default_render_config_cheng)
    # settings_in_cheng_config_not_in_blender = set(default_render_config_cheng) - set(default_render_config_blender)

    # print(settings_missing_from_cheng_config)

    # print("Settings missing from Cheng's config:")
    # for k, v in default_render_config_blender.items():
    #     if k not in default_render_config_cheng:
    #         print(f"\t{k} = {v}")
    # print()

    # print("Settings in Cheng's config but NOT Blender:")
    # for k, v in settings_in_cheng_config_not_in_blender:
    #     print(f"\t{k} = {v}")

    # uviz
    world_material = get_world_material()
    setup_black_world_material(world_material)
    set_material(cloth_obj, uv_material)
    # setup compositor
    setup_uviz_compositor()

    # setup pass index for object index channel
    cloth_obj.pass_index = 1

    render_cycles = True



    if render_cycles:
        # setup output
        setup_cycles(get_cycles_uviz_config(), use_light_tree=False)
        setup_color_management_raw()
        # curr_s = time.perf_counter()
        # print("Setup UVIZ: {}".format(curr_s - s))
        # s = curr_s

        # render
        for i in range(num_camera_angles):
            setup_exr_output(uviz_paths[i])
            set_camera_extrinsic(camera_obj, camera_extrinsic_list[i])

            if render_animation:
                scene = bpy.context.scene
                scene.frame_start = 1
                scene.frame_end = 200
            bpy.ops.render.render(animation=render_animation, write_still=True, use_viewport=False)
        # curr_s = time.perf_counter()
        # print("Render UVIZ: {}".format(curr_s - s))
        # s = curr_s

    # rgb
    # assign materials for rgb
    world_material = get_world_material()

    setup_hdri_world_material(world_material, hdri_path=HDRI_PATH)

    set_material(cloth_obj, cloth_material)

    # setup compositor
    setup_trivial_compositor()

    # setup output
    setup_eevee(get_eevee_rgb_config())
    setup_color_management_srgb()
    # curr_s = time.perf_counter()
    # print("Setup RGB: {}".format(curr_s - s))
    # s = curr_s

    # render
    for i in range(num_camera_angles):
        setup_png_output(rgb_paths[i])
        set_camera_extrinsic(camera_obj, camera_extrinsic_list[i])

        if render_animation:
            scene = bpy.context.scene
            scene.frame_start = 1
            scene.frame_end = 200
        bpy.ops.render.render(animation=render_animation, write_still=True, use_viewport=False)
    # curr_s = time.perf_counter()
    # print("Render RGB: {}".format(curr_s - s))
    # s = curr_s

    # checkpointer.save_checkpoint_if_desired()

    # pickle non-image data
    result_data = {
        'camera': {
            'intrinsic': camera_intrinsic,
            'extrinsic_list': camera_extrinsic_list
        },
        'images': {
            'uviz': [str(Path(x).name) for x in uviz_paths],
            'rgb': [str(Path(x).name) for x in rgb_paths]
        },
        'meta': {
            'sample_id': sample_id,
            'garment_name': garment_name,
            'gender': gender,
            'fabric': fabric
        }
    }
    pickle.dump(result_data, open(pickle_path, 'wb'))
    # curr_s = time.perf_counter()
    # print("Dump pickle: {}".format(curr_s - s))
    # s = curr_s