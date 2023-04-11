import bpy

def add_collision_plane(z_val: float, plane_offset: float):
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', 
                                     location=(0, 0, z_val - plane_offset), scale=(1, 1, 1))
    bpy.ops.object.modifier_add(type='COLLISION')