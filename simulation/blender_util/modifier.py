import bpy
from typing import Sequence

# object modifier manipulation
# ============================
class BaseModifier:
    def __init__(self, modifier):
        self.modifier = modifier
    
    @staticmethod
    def get_type_string():
        return 'NONE'
    
    @staticmethod
    def get_default_config(self):
        return dict()
    
    def get_config(self):
        return dict()
    
    def set_config(self, config):
        obj = self.modifier
        for key, value in config.items():
            sub_keys = key.split('.')
            last_level = obj
            for sub_key in sub_keys[:-1]:
                last_level = getattr(last_level, sub_key)
            setattr(last_level, sub_keys[-1], value)


def clear_modifiers(obj):
    obj.modifiers.clear()


def append_modifier(obj, modifier_class: type):
    assert(issubclass(modifier_class, BaseModifier))
    type_str = modifier_class.get_type_string()
    modifier = obj.modifiers.new(name=type_str, type=type_str)
    wrapper = modifier_class(modifier)
    return wrapper


def require_modifier_stack(obj, modifier_class_list: Sequence[type]):
    clear_modifiers(obj)
    modifiers = [append_modifier(obj, x) for x in modifier_class_list]
    return modifiers


# modifier wrappers
# =================
class ClothModifier(BaseModifier):
    def __init__(self, modifier):
        super().__init__(modifier)
    
    @staticmethod
    def get_type_string():
        return 'CLOTH'
    
    @staticmethod
    def get_default_config():
        config = {
            "settings.quality": 10,
            "settings.mass": 0.3,
            "collision_settings.use_collision": True,
            "collision_settings.use_self_collision": True,
            "collision_settings.distance_min": 0.005,
            "collision_settings.self_distance_min": 0.005
        }
        return config
    
    def get_config(self):
        modifier = self.modifier
        config = dict()
        setting_names = ["settings", "collision_settings"]
        for setting_name in setting_names:
            setting_attr = getattr(modifier, setting_name)
            for attr_name in dir(setting_attr):
                if attr_name.startswith("__") or "rna" in attr_name:
                    continue
                key = '.'.join([setting_name, attr_name])
                config[key] = getattr(setting_attr, attr_name)
        return config
    
    def get_point_cache(self):
        return self.modifier.point_cache


class CollisionModifier(BaseModifier):
    def __init__(self, modifier):
        super().__init__(modifier)
    
    @staticmethod
    def get_type_string():
        return 'COLLISION'
    
    @staticmethod
    def get_default_config():
        config = {
            "settings.cloth_friction": 5.0
        }
        return config

    def get_config(self):
        modifier = self.modifier
        config = dict()
        setting_names = ["settings"]
        for setting_name in setting_names:
            setting_attr = getattr(modifier, setting_name)
            for attr_name in dir(setting_attr):
                if attr_name.startswith("__") or "rna" in attr_name:
                    continue
                key = '.'.join([setting_name, attr_name])
                config[key] = getattr(setting_attr, attr_name)


class SubdivModifier(BaseModifier):
    def __init__(self, modifier):
        super().__init__(modifier)
    
    @staticmethod
    def get_type_string():
        return 'SUBSURF'

    @staticmethod
    def get_default_config():
        config = {
            "levels": 3,
            "render_levels": 3
        }
        return config


class VertexWeigthMixModifier(BaseModifier):
    def __init__(self, modifier):
        super().__init__(modifier)

    @staticmethod
    def get_type_string():
        return 'VERTEX_WEIGHT_MIX'

    @staticmethod
    def get_default_config():
        config = {
            'vertex_group_a': 'pinned',
            'vertex_group_b': 'zero',
            'mix_mode': 'SUB',
            'mix_set': 'ALL',
            'default_weight_a': 0,
            'default_weight_b': 0
        }
        return config


class ArmatureModifier(BaseModifier):
    def __init__(self, modifier):
        super().__init__(modifier)

    @staticmethod
    def get_type_string():
        return 'ARMATURE'

    @staticmethod
    def get_default_config():
        config = {
            'object': None,
            'use_vertex_groups': True
        }
        return config
