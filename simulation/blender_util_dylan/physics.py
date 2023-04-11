import bpy

from simulation.blender_util_dylan.shape_keys import set_obj_default_shape_from_shape_key

def set_sim_output_as_default_mesh_shape(obj, initial_sim_end_frame: int,
                                         modifier_name: str="CLOTH"):
    """Sets the output of a cloth simulation as the new resting state for the mesh

    This is helpful when we need to simulate a static resting state for a grasped cloth then do more
    simulations with the hanging configuration.
    """
    # Set the object as the active object for Blender so that we can modify it.
    bpy.context.view_layer.objects.active = obj

    # Set the active frame to one that is after the initial simulation.
    bpy.context.scene.frame_set(initial_sim_end_frame + 1)

    # Apply the cloth modifier as a shapekey.
    bpy.ops.object.modifier_apply_as_shapekey(keep_modifier=True, modifier=modifier_name)

    # Set the mesh's default shape to the shape of the shape key we just created from the sim output.
    set_obj_default_shape_from_shape_key(obj, modifier_name, verbose=True)
