import bpy
# from blender_util import modifier
from blender_util.modifier import (
    append_modifier,
    require_modifier_stack,
    ClothModifier,
    VertexWeigthMixModifier,
    ArmatureModifier,
    CollisionModifier,
    SubdivModifier
)
from blender_util.gripping_util import kPinnedGroupName


# setup helpers
# =============
def setup_simple_cloth_modifiers(
        obj, 
        cloth_config=ClothModifier.get_default_config()):
    stack = require_modifier_stack(obj, [ClothModifier])
    cloth_modifier = stack[0]
    cloth_modifier.set_config(cloth_config)
    return cloth_modifier


def setup_cloth_gripping_modifiers(
        obj,
        armature_object,
        wight_mix_config=VertexWeigthMixModifier.get_default_config(),
        cloth_config=ClothModifier.get_default_config()):
    class_stack = [
        VertexWeigthMixModifier, 
        ArmatureModifier, 
        ClothModifier]
    stack = require_modifier_stack(obj, class_stack)
    vwm, arm, cloth = stack

    vwm.set_config(wight_mix_config)
    arm_config = ArmatureModifier.get_default_config()
    arm_config['object'] = armature_object
    arm.set_config(arm_config)
    cloth_config['settings.vertex_group_mass'] = kPinnedGroupName
    cloth.set_config(cloth_config)
    return vwm, arm, cloth


def setup_collision_modifiers(
        obj,
        collision_config=CollisionModifier.get_default_config()):
    stack = require_modifier_stack(obj, [CollisionModifier])
    collision = stack[0]
    collision.set_config(collision_config)
    return collision


def require_subdiv_modifier(
        mesh_obj, 
        subdiv_config=SubdivModifier.get_default_config()):
    """
    Ensures the top of the modifier stack is a subsurf modifier.
    """
    need_modifier = False
    if len(mesh_obj.modifiers) > 0:
        last_modifier = mesh_obj.modifiers[-1]
        if last_modifier.type != 'SUBSURF':
            need_modifier = True
    else:
        need_modifier = True
    if need_modifier:
        last_modifier = append_modifier(mesh_obj, SubdivModifier).modifier
    subdiv_modifier = SubdivModifier(last_modifier)
    subdiv_modifier.set_config(subdiv_config)
    return subdiv_modifier

def remove_subdiv_modifier(mesh_obj):
    last_modifier = mesh_obj.modifiers[-1]
    if last_modifier.type == 'SUBSURF':
        mesh_obj.modifier.remove(last_modifier)


# bake physics
# ============
def run_simulation(obj, point_cache):
    context = {
        'scene': bpy.data.scenes[0],
        'active_object': obj,
        'point_cache': point_cache
    }
    if point_cache.is_baked:
        bpy.ops.ptcache.free_bake(context)
    assert(not point_cache.is_baked)
    # setting active object is requried to prevent crashing
    bpy.context.view_layer.objects.active = obj
    assert(bpy.ops.ptcache.bake.poll(context))
    bpy.ops.ptcache.bake(context, bake=True)
