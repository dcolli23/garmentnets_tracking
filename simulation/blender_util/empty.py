import bpy

from blender_util.collection import get_empty_collection

def require_empty(name):
    if name not in bpy.data.objects:
        bpy.data.objects.new(name, None)
    empty_obj = bpy.data.objects[name]

    empty_collection = get_empty_collection()
    if name not in empty_collection.objects:
        empty_collection.objects.link(empty_obj)
    return empty_obj
