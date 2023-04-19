# %% [markdown]
# # Rendering Prep
#
# This notebook is for prototyping/prepping the driver for rendering the freespace dynamics dataset
#
# NOTE: This was converted to a script and is run via command line.

# %%
import os
import sys
from pathlib import Path
import pickle
import time

import bpy
import mathutils
import numpy as np

# Imports from this repository
sys.path.append("../../")
sys.path.append("../")
from simulation.cloth_3d_util.accessor import Cloth3DCanonicalAccessor

from simulation.blender_util_dylan.render import enable_gpu_renders, render_dylan
from simulation.blender_util_dylan.debug import print_nested_dict_types
from simulation.blender_util.camera import generate_intrinsic
from simulation.cloth_3d_util.accessors.access_functions import (
    get_info, get_garment_texture, get_garment_metadata
)

# %load_ext autoreload
# %autoreload 2

# %%
FILE_ROOT = Path(os.getcwd())
CLOTH3D_PATH = Path(os.path.expanduser("~/DataLocker/datasets/CLOTH3D/training/"))
OUTPUT_ROOT = FILE_ROOT / ".." / "script_output" / "full_dataset_attempt_2"
GARMENTNETS_SAMPLE_DATASET_ZARR = (FILE_ROOT / ".." / ".." / "data" /
                                   "garmentnets_simulation_dataset_sample.zarr")
assert (GARMENTNETS_SAMPLE_DATASET_ZARR.exists())
GARMENTNETS_SAMPLE_PATH = ["Tshirt", "samples"]

NUM_ACTION_SEQUENCES = 5
DYNAMICS_ANIMATION_DIR_TEMPLATE = "dynamics_seq_{idx}"
DYNAMICS_ANIMATION_FILENAME = "dynamics_animation.blend"

PLANE_OFFSET = 0.025  # [m]

CAMERA_INTRINSIC = generate_intrinsic(1024, 1024, 2048)
CAMERA_Z_OFFSET = -0.4

# %% [markdown]
# ## First, Render The Hanging Resting State (GarmentNets Input)

# %%
def render_resting_states(render_eevee=True):
    num_camera_angles = 4
    for sample_dir in OUTPUT_ROOT.iterdir():
        start_time = time.perf_counter()
        sample_key = sample_dir.name

        render_output_example_1 = sample_dir / "rgb_0.png"
        render_output_example_2 = sample_dir / "uviz_3.exr"
        if render_output_example_1.exists() and render_output_example_2.exists():
            print(f"Already rendered hanging state for '{sample_key}', skipping render.", flush=True)
            continue

        print(f"Animating {sample_key} resting state (GarmentNets input)", flush=True)

        sample_id, garment_name, grip_vertex_idx_key = sample_key.split('_')

        # Load the initial sim results
        results_dict_filepath = sample_dir / "hanging_rest_state_results.pkl"
        assert(results_dict_filepath.exists())
        results_dict_smpl = pickle.load(results_dict_filepath.open('rb'))
        # print(results)
        cloth_state = results_dict_smpl["cloth_state"]

        assert (int(grip_vertex_idx_key) == results_dict_smpl["grip_vertex_idx"])

        # Load the garment information dictionary.
        accessor = Cloth3DCanonicalAccessor(CLOTH3D_PATH)
        sample_data = accessor.get_sample_data(sample_id, garment_name)
        # print_nested_dict_types(sample_data)


        # Load the hanging rest state blend file.
        blend_path = sample_dir / "hanging_rest_state.blend"
        assert(blend_path.exists())
        bpy.ops.wm.open_mainfile(filepath=blend_path.as_posix())

        # Add the world material I accidentally deleted when simulating hanging rest state
        # bpy.ops.world.new()
        # print(bpy.data.worlds.keys())


        info = get_info(CLOTH3D_PATH, sample_id)
        garment_texture = get_garment_texture(CLOTH3D_PATH, sample_id, garment_name, info=info)
        garment_meta = get_garment_metadata(CLOTH3D_PATH, sample_id, garment_name, info=info)
        gender = garment_meta['gender']
        fabric = garment_meta['garment_fabric']

        enable_gpu_renders()
        render_dylan(
            output_path=sample_dir,
            sample_id=sample_id,
            garment_name=garment_name,
            gender=gender,
            fabric=fabric,
            garment_verts=cloth_state["verts"],
            garment_faces=cloth_state["faces"],
            garment_uv_verts=cloth_state["uv_verts"],
            garment_uv_faces=cloth_state["uv_faces"],
            garment_texture=garment_texture,
            num_camera_angles=num_camera_angles,
            camera_intrinsic=CAMERA_INTRINSIC,
            z_offset=CAMERA_Z_OFFSET,
            render_eevee=render_eevee
        )
        end_time = time.perf_counter()
        print(f"Took {end_time - start_time}",flush=True)

# %%
def render_dynamics_animations(render_eevee=True, views_to_render=None):
    # Iterate through the action sequences first so we still get good sample coverage even if we have to
    # stop the rendering early to train
    for seq_idx in range(NUM_ACTION_SEQUENCES):
        for sample_dir in OUTPUT_ROOT.iterdir():
            start_time = time.perf_counter()
            sample_key = sample_dir.name
            print(f"Animating {sample_key}, action sequence: {seq_idx}", flush=True)

            action_seq_path = sample_dir / DYNAMICS_ANIMATION_DIR_TEMPLATE.format(idx=seq_idx)
            full_animation_blend_path = action_seq_path / DYNAMICS_ANIMATION_FILENAME

            assert(full_animation_blend_path.exists())

            sample_id, garment_name, grip_vertex_idx_key = sample_key.split('_')

            # Load the initial sim results
            results_dict_filepath = sample_dir / "hanging_rest_state_results.pkl"
            assert(results_dict_filepath.exists())
            results_dict_smpl = pickle.load(results_dict_filepath.open('rb'))
            # print(results)
            cloth_state = results_dict_smpl["cloth_state"]

            # Load the garment information dictionary.
            accessor = Cloth3DCanonicalAccessor(CLOTH3D_PATH)
            sample_data = accessor.get_sample_data(sample_id, garment_name)

            # Load the blend file.
            bpy.ops.wm.open_mainfile(filepath=full_animation_blend_path.as_posix())

            info = get_info(CLOTH3D_PATH, sample_id)
            garment_texture = get_garment_texture(CLOTH3D_PATH, sample_id, garment_name, info=info)
            garment_meta = get_garment_metadata(CLOTH3D_PATH, sample_id, garment_name, info=info)
            gender = garment_meta['gender']
            fabric = garment_meta['garment_fabric']

            print("Manually setting number of camera angles to 4")
            enable_gpu_renders()
            render_dylan(
                output_path=action_seq_path,
                sample_id=sample_id,
                garment_name=garment_name,
                gender=gender,
                fabric=fabric,
                garment_verts=cloth_state["verts"],
                garment_faces=cloth_state["faces"],
                garment_uv_verts=cloth_state["uv_verts"],
                garment_uv_faces=cloth_state["uv_faces"],
                garment_texture=garment_texture,
                num_camera_angles=4,
                camera_intrinsic=CAMERA_INTRINSIC,
                render_animation=True,
                z_offset=CAMERA_Z_OFFSET,
                views_to_render=views_to_render,
                render_eevee=render_eevee
            )
            end_time = time.perf_counter()
            print(f"Full animation render took: {end_time - start_time}", flush=True)


if __name__ == "__main__":
    # print("Only rendering resting states!")
    print("Only rendering Cycles!")
    render_eevee = False
    # render_resting_states(render_eevee=render_eevee)
    print("Only rendering dynamics animations!", flush=True)
    print("Not rendering view 0! Rendering view 1, 2, 3")
    render_dynamics_animations(render_eevee=render_eevee, views_to_render=[1, 2, 3])
# %%



