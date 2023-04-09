import bpy

from blender_util.collection import get_armature_collection
from blender_util.mesh import require_vertex_group


def require_armature(name):
    if name not in bpy.data.armatures:
        bpy.data.armatures.new(name)
    armature = bpy.data.armatures[name]

    if name not in bpy.data.objects:
        bpy.data.objects.new(name, armature)
    armature_obj = bpy.data.objects[name]

    armature_collection = get_armature_collection()
    if name not in armature_collection.objects:
        armature_collection.objects.link(armature_obj)
        # fix the bug that armature is not immediately avaliable
        # after creation
        bpy.context.view_layer.update()
    return armature_obj


def require_bone(armature_obj, bone_name):
    assert(hasattr(armature_obj.pose, 'bones'))
    if bone_name not in armature_obj.pose.bones:
        # must be in edit mode to add bones
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        edit_bones = armature_obj.data.edit_bones

        bone = edit_bones.new(bone_name)
        bone.head = (0, 0, 0)
        bone.tail = (0, 0, 0.1)

        # exit edit mode to save bones so they can be used in pose mode
        bpy.ops.object.mode_set(mode='OBJECT')

    # pickup bone in object mode
    bone = armature_obj.pose.bones[bone_name]
    return bone


def bind_armature_to_mesh(armature_obj, mesh_obj):
    armature_obj.matrix_world = mesh_obj.matrix_world
    mesh_obj.location = (0, 0, 0)
    mesh_obj.rotation_euler = (0, 0, 0)
    mesh_obj.parent = armature_obj


def require_bone_vertex_groups(armature_obj, mesh_obj):
    bone_names = list(armature_obj.pose.bones.keys())
    vertex_groups = dict()
    for bone_name in bone_names:
        vg = require_vertex_group(mesh_obj, bone_name)
        vertex_groups[bone_name] = vg
    return vertex_groups
