import bpy
from typing import Sequence


def remove_all_collections() -> None:
    # avoid holding references to any objects, which might cause memory error
    # the list conversion is required to prevent crashing
    for collection in list(bpy.data.collections):
        for obj in list(collection.all_objects):
            bpy.data.objects.remove(obj)
        bpy.data.collections.remove(collection)


def require_collection(name: str) -> bpy.types.Collection:
    """
    Return colleciton with name, create if not exist.
    """
    if name not in bpy.data.collections:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return bpy.data.collections[name]


def get_camera_collection() -> bpy.types.Collection:
    return require_collection('camera_collection')


def get_light_collection() -> bpy.types.Collection:
    return require_collection('light_collection')


def get_mesh_collection() -> bpy.types.Collection:
    return require_collection('mesh_collection')


def get_armature_collection() -> bpy.types.Collection:
    return require_collection('armature_collection')

def get_empty_collection() -> bpy.types.Collection:
    return require_collection('empty_collection')
