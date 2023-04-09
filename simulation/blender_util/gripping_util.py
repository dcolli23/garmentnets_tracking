import bpy
from mathutils import Matrix, Vector
import numpy as np
from typing import Sequence

from blender_util.common import get_attr_data_path
from blender_util.animation import (
    build_animation_dataframe,
    set_animiation
)
from blender_util.mesh import (
    require_vertex_group
)
from blender_util.modifier import (
    VertexWeigthMixModifier
)


kPinnedGroupName = 'pinned'
kZeroGroupName = 'zero'


def setup_gripper_vertex_groups(
        armature_obj, mesh_obj, bone_name, vertex_idx):
    """
    vertex_idx is the index of vertex to be gripped.
    """
    bone_vg = require_vertex_group(mesh_obj, bone_name)
    pinned_vg = require_vertex_group(mesh_obj, kPinnedGroupName)
    _ = require_vertex_group(mesh_obj, kZeroGroupName)

    bone_vg.add([vertex_idx], 1, 'ADD')
    pinned_vg.add([vertex_idx], 1, 'ADD')


def setup_gripper_movement(
        armature_obj,
        bone,
        frames: Sequence[float],
        gripper_positions: Sequence[tuple]):

    tx_bone_world = armature_obj.matrix_world.inverted() @ bone.bone.matrix_local.inverted()
    gripper_positions = [tuple(tx_bone_world @ Vector(x)) for x in gripper_positions]
    
    animation_df = build_animation_dataframe(frames, gripper_positions)
    data_path = get_attr_data_path(bone, 'location')
    set_animiation(armature_obj, data_path, animation_df)


def setup_gripper_effect(
        mesh_obj,
        frames: Sequence[float], 
        effect: Sequence[float]):
    """
    Effect is a float in [0, 1] where 1 is fully gripped and 0 is fully loosed.
    """
    # modifiers' name is same as type
    weigth_mix_modifier = mesh_obj.modifiers[VertexWeigthMixModifier.get_type_string()]

    subtracted_weight = 1 - np.array(effect)
    animation_df = build_animation_dataframe(frames, subtracted_weight)
    data_path = get_attr_data_path(weigth_mix_modifier, 'default_weight_b')
    set_animiation(mesh_obj, data_path, animation_df)

