import bpy

def get_attr_data_path(obj, attr):
    return obj.path_from_id(attr)

def get_data_path_attr(obj, datapath):
    return obj.path_resolve(datapath)
