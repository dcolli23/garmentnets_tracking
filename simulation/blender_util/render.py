import bpy

# config dict helpers
# ===================
# can't find where this is defined
COLOR_TYPE = type(bpy.context.scene.eevee.bloom_color)

def get_renderer_config(renderer):
    config = dict()
    for attr_name in renderer.bl_rna.properties.keys():
        attr = getattr(renderer, attr_name)
        if attr_name.startswith('__') or 'rna' in attr_name:
            continue
        if renderer.bl_rna.properties[attr_name].is_readonly:
            continue
        if callable(attr):
            continue
        if isinstance(attr, COLOR_TYPE):
            attr = list(attr)
        config[attr_name] = attr
    return config

def set_renderer_config(renderer, config):
    print("Setting render configuration and printing failed sets.")
    for key, value in config.items():
        # print(f"\t{key} = {value}")
        try:
            setattr(renderer, key, value)
        except:
            print(f"\t{key}")

def get_default_eevee_config():
    # These parameters might change with blender major version.
    config = {'gi_diffuse_bounces': 3,
        'gi_cubemap_resolution': '512',
        'gi_visibility_resolution': '32',
        'gi_irradiance_smoothing': 0.10000000149011612,
        'gi_glossy_clamp': 0.0,
        'gi_filter_quality': 3.0,
        'gi_show_irradiance': False,
        'gi_show_cubemaps': False,
        'gi_irradiance_display_size': 0.10000000149011612,
        'gi_cubemap_display_size': 0.30000001192092896,
        'gi_auto_bake': False,
        'taa_samples': 16,
        'taa_render_samples': 64,
        'use_taa_reprojection': True,
        'sss_samples': 7,
        'sss_jitter_threshold': 0.30000001192092896,
        'use_ssr': False,
        'use_ssr_refraction': False,
        'use_ssr_halfres': True,
        'ssr_quality': 0.25,
        'ssr_max_roughness': 0.5,
        'ssr_thickness': 0.20000000298023224,
        'ssr_border_fade': 0.07500000298023224,
        'ssr_firefly_fac': 10.0,
        'volumetric_start': 0.10000000149011612,
        'volumetric_end': 100.0,
        'volumetric_tile_size': '8',
        'volumetric_samples': 64,
        'volumetric_sample_distribution': 0.800000011920929,
        'use_volumetric_lights': True,
        'volumetric_light_clamp': 0.0,
        'use_volumetric_shadows': False,
        'volumetric_shadow_samples': 16,
        'use_gtao': False,
        'use_gtao_bent_normals': True,
        'use_gtao_bounce': True,
        'gtao_factor': 1.0,
        'gtao_quality': 0.25,
        'gtao_distance': 0.20000000298023224,
        'bokeh_max_size': 100.0,
        'bokeh_threshold': 1.0,
        'use_bloom': False,
        'bloom_threshold': 0.800000011920929,
        'bloom_color': [1.0, 1.0, 1.0],
        'bloom_knee': 0.5,
        'bloom_radius': 6.5,
        'bloom_clamp': 0.0,
        'bloom_intensity': 0.05000000074505806,
        'use_motion_blur': False,
        'motion_blur_shutter': 0.5,
        'motion_blur_depth_scale': 100.0,
        'motion_blur_max': 32,
        'motion_blur_steps': 1,
        'shadow_cube_size': '512',
        'shadow_cascade_size': '1024',
        'use_shadow_high_bitdepth': False,
        'use_soft_shadows': True,
        'light_threshold': 0.009999999776482582,
        'use_overscan': False,
        'overscan_size': 3.0}
    return config

def get_default_cycles_config():
    config = {'name': '',
        'device': 'GPU',
        'feature_set': 'SUPPORTED',
        'shading_system': False,
        'progressive': 'PATH',
        'preview_pause': False,
        'use_denoising': False,
        'use_preview_denoising': False,
        'denoiser': 'NLM',
        'preview_denoiser': 'AUTO',
        'use_square_samples': False,
        'samples': 1,
        'preview_samples': 32,
        'aa_samples': 128,
        'preview_aa_samples': 32,
        'diffuse_samples': 1,
        'glossy_samples': 1,
        'transmission_samples': 1,
        'ao_samples': 1,
        'mesh_light_samples': 1,
        'subsurface_samples': 1,
        'volume_samples': 1,
        'sampling_pattern': 'SOBOL',
        'use_layer_samples': 'USE',
        'sample_all_lights_direct': True,
        'sample_all_lights_indirect': True,
        'light_sampling_threshold': 0.009999999776482582,
        'use_adaptive_sampling': False,
        'adaptive_threshold': 0.0,
        'adaptive_min_samples': 0,
        'min_light_bounces': 0,
        'min_transparent_bounces': 0,
        'caustics_reflective': True,
        'caustics_refractive': True,
        'blur_glossy': 1.0,
        'max_bounces': 0,
        'diffuse_bounces': 4,
        'glossy_bounces': 4,
        'transmission_bounces': 12,
        'volume_bounces': 0,
        'transparent_max_bounces': 8,
        'volume_step_rate': 1.0,
        'volume_preview_step_rate': 1.0,
        'volume_max_steps': 1024,
        'dicing_rate': 1.0,
        'preview_dicing_rate': 8.0,
        'max_subdivisions': 12,
        'dicing_camera': None,
        'offscreen_dicing_scale': 4.0,
        'film_exposure': 1.0,
        'film_transparent_glass': False,
        'film_transparent_roughness': 0.10000000149011612,
        'filter_type': 'BLACKMAN_HARRIS',
        'pixel_filter_type': 'BLACKMAN_HARRIS',
        'filter_width': 1.5,
        'seed': 0,
        'use_animated_seed': False,
        'sample_clamp_direct': 0.0,
        'sample_clamp_indirect': 10.0,
        'debug_tile_size': 1024,
        'preview_start_resolution': 64,
        'preview_denoising_start_sample': 1,
        'debug_reset_timeout': 0.10000000149011612,
        'debug_cancel_timeout': 0.10000000149011612,
        'debug_text_timeout': 1.0,
        'debug_bvh_type': 'DYNAMIC_BVH',
        'debug_use_spatial_splits': False,
        'debug_use_hair_bvh': True,
        'debug_bvh_time_steps': 0,
        'tile_order': 'HILBERT_SPIRAL',
        'use_progressive_refine': False,
        'bake_type': 'COMBINED',
        'use_camera_cull': False,
        'camera_cull_margin': 0.10000000149011612,
        'use_distance_cull': False,
        'distance_cull_margin': 50.0,
        'motion_blur_position': 'CENTER',
        'rolling_shutter_type': 'NONE',
        'rolling_shutter_duration': 0.10000000149011612,
        'texture_limit': 'OFF',
        'texture_limit_render': 'OFF',
        'ao_bounces': 0,
        'ao_bounces_render': 0,
        'debug_use_cpu_avx2': True,
        'debug_use_cpu_avx': True,
        'debug_use_cpu_sse41': True,
        'debug_use_cpu_sse3': True,
        'debug_use_cpu_sse2': True,
        'debug_bvh_layout': 'EMBREE',
        'debug_use_cpu_split_kernel': False,
        'debug_use_cuda_adaptive_compile': False,
        'debug_use_cuda_split_kernel': False,
        'debug_optix_cuda_streams': 1,
        'debug_optix_curves_api': False,
        'debug_opencl_kernel_type': 'DEFAULT',
        'debug_opencl_device_type': 'ALL',
        'debug_use_opencl_debug': False,
        'debug_opencl_mem_limit': 0}
    return config

def get_eevee_rgb_config():
    config = get_default_eevee_config()
    config['use_gtao'] = True  # ambient occlusion
    config['use_ssr'] = True  # screen space reflection
    return config

def get_cycles_uviz_config():
    config = get_default_cycles_config()
    # maximize framerate
    config['device'] = 'GPU'
    config['progressive'] = 'PATH'
    config['samples'] = 1
    config['max_bounces'] = 0
    return config

def get_cycles_rgb_config():
    config = get_default_cycles_config()
    config['device'] = 'GPU'
    config['progressive'] = 'PATH'
    config['samples'] = 16
    config['max_bounces'] = 2
    return config

# render setup functions
# ======================

def setup_eevee(config=get_default_eevee_config()):
    scene = bpy.context.scene

    scene.render.engine = 'BLENDER_EEVEE'
    # disable AA
    scene.render.filter_size = 0.0

    set_renderer_config(scene.eevee, config)

def setup_cycles(config=get_default_cycles_config(), use_light_tree=True):
    scene = bpy.context.scene

    scene.render.engine = 'CYCLES'
    # NOTE Dylan: This doesn't appear to be supported anymore.
    # maximize GPU throughput by setting one tile per image
    # scene.render.tile_x = scene.render.resolution_x
    # scene.render.tile_y = scene.render.resolution_y
    # disable AA
    scene.render.filter_size = 0.0

    # Dylan: Trying to stop the light tree from building since that takes the majority of the time.
    scene.cycles.use_light_tree = use_light_tree

    set_renderer_config(scene.cycles, config)


# color management settings
# =========================

def setup_color_management_srgb():
    scene = bpy.context.scene

    # display
    scene.display_settings.display_device = 'sRGB'
    # view
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    # sequencer
    scene.sequencer_colorspace_settings.name = 'sRGB'

def setup_color_management_raw():
    scene = bpy.context.scene

    # display
    scene.display_settings.display_device = 'None'
    # view
    scene.view_settings.view_transform = 'Standard'
    # sequencer
    scene.sequencer_colorspace_settings.name = 'Raw'

# output setup functions
# ======================
def setup_png_output(output_path='data/output/###.png'):
    scene = bpy.context.scene
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.use_zbuffer = False
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'

def setup_exr_output(output_path='data/output/###.exr'):
    scene = bpy.context.scene
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.exr_codec = 'ZIP'
    print("Dylan need to change use zbuffer to True??")
    scene.render.image_settings.use_zbuffer = False
    # scene.render.image_settings.use_zbuffer = True
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '16'

# render
# ======
def render_animation(start, end):
    assert(start <= end)

    scene = bpy.context.scene
    scene.frame_start = start
    scene.frame_end = end

    bpy.ops.render.render(animation=True, write_still=True, use_viewport=False)
