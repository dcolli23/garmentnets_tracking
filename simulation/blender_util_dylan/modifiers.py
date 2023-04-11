import bpy

def make_modifier_highest_priority(modifier_name: str):
    bpy.ops.object.modifier_set_active(modifier=modifier_name)
    bpy.ops.object.modifier_move_to_index(modifier=modifier_name, index=0)