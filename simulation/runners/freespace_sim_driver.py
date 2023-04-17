import os
import sys
from pathlib import Path
import pickle

import bpy
import mathutils
import numpy as np

# Imports from this repository
sys.path.append("../../")
sys.path.append("../")
from simulation.cloth_3d_util.accessor import Cloth3DCanonicalAccessor
from simulation.cloth_3d_util.util import loadInfo
from simulation.pipeline.simulate_garment_hanging_rest_state import \
    simulate_garment_hanging_rest_state
from simulation.blender_util_dylan.physics import set_sim_output_as_default_mesh_shape
from simulation.blender_util_dylan.checkpointer import BlendFileCheckpointer
from simulation.blender_util_dylan.gripper import GripperAnimation
from simulation.blender_util_dylan.debug import print_obj_keyframe_coordinates


FILE_ROOT = Path(os.path.dirname(__file__))
CLOTH3D_PATH = Path(os.path.expanduser("~/DataLocker/datasets/CLOTH3D/training/"))
OUTPUT_ROOT = FILE_ROOT / ".." / "script_output" / "full_dataset_attempt_2"
GARMENTNETS_SAMPLE_DATASET_ZARR = (FILE_ROOT / ".." / ".." / "data" / 
                                   "garmentnets_simulation_dataset_sample.zarr")
assert (GARMENTNETS_SAMPLE_DATASET_ZARR.exists())
GARMENTNETS_SAMPLE_PATH = ["Tshirt", "samples"]
NUM_ACTION_SEQS_PER_SAMPLE = 5

# Make the output directory if it doesn't exist.
OUTPUT_ROOT.mkdir(exist_ok=True)

def simulate_all_sequences(sample_configs, overwrite=False):
    for config in sample_configs:
        ## Setup the simulation.
        sample_key = f"{config['sample_id']}_{config['garment_name']}_{config['grip_vertex_idx']}"
        # control_sequences = dynamics_control_sequences[sample_key]
        print("Simulating Dynamics for Sample:", sample_key)

        
        # accessor = Cloth3DCanonicalAccessor(CLOTH3D_PATH)
        sample_dir = OUTPUT_ROOT / sample_key
        sample_dir.mkdir(exist_ok=True)

        # Get a dictionary containing the data for this sample garment.
        # sample_data = accessor.get_sample_data(**config)

        # garment_info_mat_filename = CLOTH3D_PATH / config["sample_id"] / "info.mat"
        # garment_info = loadInfo(garment_info_mat_filename)

        checkpointer = BlendFileCheckpointer(sample_dir, save_new_checkpoints=False)

        for seq_idx in range(NUM_ACTION_SEQS_PER_SAMPLE):
            print("Simulating Control Sequence", seq_idx)

            # Reset the environment to the simulated resting state
            checkpointer.load_hanging_rest_state()
            
            seq_dir = sample_dir / f"dynamics_seq_{seq_idx}"
            assert (seq_dir.exists())

            control_sequence_path = seq_dir / "control_sequence.pkl"
            control_sequence = pickle.load(control_sequence_path.open('rb'))

            final_blend_path = seq_dir / "dynamics_animation.blend"

            if final_blend_path.exists():
                if overwrite:
                    print("Overwriting existing dynamics simulation")
                else:
                    print("Dynamics simulation already exists. Skipping")
                    continue


            # Create the gripper animation controller
            cloth_obj = bpy.data.objects["cloth"]
            gripper_obj = bpy.data.objects["Empty"]
            gripper_animation = GripperAnimation(gripper_obj, cloth_obj)

            # for vec in direction_vecs:
            for action in control_sequence:
                gripper_animation.add_movement(action["direction_vec"], action["velocity"], 
                                            action["frame_duration"])
            
            # Now bake the sim.
            bpy.ops.ptcache.bake_all()

            # Get the ground truth mesh for each frame in the animation.
            # Actually, this isn't necessary if we just save the final blend file state.
            # bpy.context.scene.frame_set()

            # Finally, save the final simulation blend file state so we can easily render later on.
            bpy.ops.wm.save_as_mainfile(filepath=final_blend_path.as_posix())

def main():
    sample_configs = []
    for sample_output_dir in OUTPUT_ROOT.iterdir():
        sample_id, garment_name, grasped_idx = sample_output_dir.name.split('_')
        sample_configs.append({
            "sample_id": sample_id,
            "garment_name": garment_name,
            "grip_vertex_idx": int(grasped_idx)
        })

    # for config in sample_configs:
    #     print(config)
    simulate_all_sequences(sample_configs, overwrite=False)
            
if __name__ == "__main__":
    main()