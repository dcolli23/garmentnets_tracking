from pathlib import Path
from typing import Tuple

import bpy
import numpy as np

from simulation.pipeline.simulate_garment_hanging_rest_state import \
    simulate_garment_hanging_rest_state
from simulation.blender_util.physics import run_simulation
from simulation.blender_util_dylan.checkpointer import BlendFileCheckpointer
from simulation.blender_util_dylan.physics import set_sim_output_as_default_mesh_shape
from simulation.blender_util_dylan.mesh import add_collision_plane
from simulation.blender_util_dylan.gripper import require_virtual_gripper

PLANE_OFFSET = 0.025  # [m]
LINE_SEP = 80 * '-'

def simulate_lowering_cloth_onto_table_full(sample_config: dict, sample_data: dict,
                                            smpl_simulation_duration_pair: Tuple[int, int],
                                            grip_lowering_args: dict,
                                            blend_checkpoint_filepath_root: str=None):
    result_data_smpl = simulate_garment_hanging_rest_state(sample_config, sample_data)

    # TODO: Figure out what I want to return from the cloth lowering simulation.
    result_data_lowering = simulate_lowering_cloth_onto_table_after_smpl_sim(grip_lowering_args,
        smpl_simulation_duration_pair, result_data_smpl, blend_checkpoint_filepath_root)

def simulate_lowering_cloth_onto_table_after_smpl_sim(grip_lowering_args: dict,
                                                      smpl_simulation_duration_pair: Tuple[int, int],
                                                      smpl_sim_results: dict,
                                                      sample_dir: Path,
                                                      save_new_checkpoints=True):
    print("Running Simulation Of Lowering Cloth Onto Table")
    print(LINE_SEP)
    checkpointer = BlendFileCheckpointer(sample_dir, save_new_checkpoints=save_new_checkpoints)

    cloth_obj = bpy.data.objects["cloth"]

    set_sim_output_as_default_mesh_shape(cloth_obj,
                                         initial_sim_end_frame=smpl_simulation_duration_pair[1])

    checkpointer.save_hanging_rest_state_if_desired()

    # Reset to the 1st frame of the simulation.
    bpy.context.scene.frame_set(1)

    # Delete the physics cache.
    bpy.ops.ptcache.free_bake_all()

    # Add a hook to a new empty so that we can control the cloth and lower it onto the "table"
    require_virtual_gripper(cloth_obj)

    ## Add animation of the virtual gripper. We'll lower the cloth onto the table over some period of
    # time, then stay stationary for a brief period of time to let the cloth settle.

    # Get the minimum Z value of all vertices.
    min_z_val = smpl_sim_results['cloth_state']['verts'][:, 2].min()
    min_z_val_abs = np.abs(min_z_val)

    # The amount we move the "gripper" (empty object) downward should be <fraction_moved> * abs(min_z_val)
    gripper_dz = min_z_val_abs * grip_lowering_args["fraction_lowered"]

    # Select the Empty object.
    bpy.context.view_layer.objects.active = bpy.data.objects["Empty"]

    # Add keyframe for empty at start frame.
    bpy.context.scene.frame_set(grip_lowering_args["start_frame"])
    empty_obj = bpy.data.objects["Empty"]
    empty_obj.keyframe_insert(data_path="location", frame=grip_lowering_args["start_frame"])

    # Move the gripper to the final Z value at the final frame.
    empty_obj_z_init = empty_obj.location[2]
    bpy.context.scene.frame_set(grip_lowering_args["end_frame"])
    empty_obj.location[2] = empty_obj_z_init - gripper_dz

    # Now insert another keyframe at the final frame so that Blender will animate the movement for us.
    empty_obj.keyframe_insert(data_path="location", frame=grip_lowering_args["end_frame"])

    checkpointer.save_checkpoint_if_desired()

    # Add a plane below the lowest vertex value from the initial simulation. This is what we'll
    # lower the mesh onto to deform it.
    # NOTE: The plane must be in the same collection as the cloth.
    #       Additionally, the plane must have collision physics turned on.
    add_collision_plane(min_z_val, PLANE_OFFSET)

    # Clear the bake cache for the physics.
    bpy.ops.ptcache.free_bake_all()

    # Select the cloth object.
    bpy.context.view_layer.objects.active = bpy.data.objects["cloth"]

    # Change the physics simulation to last as long as we specify above.
    cloth_modifier = bpy.context.object.modifiers["CLOTH"]
    cloth_modifier.point_cache.frame_start = 0
    cloth_modifier.point_cache.frame_end = grip_lowering_args["end_frame"]
    cloth_modifier.collision_settings.use_collision = True
    cloth_modifier.collision_settings.distance_min = 0.005

    # Bake the physics
    run_simulation(cloth_obj, cloth_modifier.point_cache)

    checkpointer.save_checkpoint_if_desired()