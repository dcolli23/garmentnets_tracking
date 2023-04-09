"""
The animation system of Blender...
What a mess!
"""

import bpy
import numpy as np
import pandas as pd

from blender_util.common import get_attr_data_path, get_data_path_attr

# low level API
# =============
def require_action(obj):
    assert(obj.name in bpy.data.objects)

    if not obj.animation_data:
        obj.animation_data_create()

    if not obj.animation_data.action:
        obj.animation_data.action = bpy.data.actions.new(name=obj.name + "_action")

    return obj.animation_data.action


def get_fcurves_dataframe(obj, data_path=None):
    action = require_action(obj)
    fcurves = list(action.fcurves)

    data = list()
    keys = ['data_path', 'array_index']
    for fcurve in fcurves:
        row_dict = dict()
        for key in keys:
            row_dict[key] = getattr(fcurve, key)
        row_dict['fcurve'] = fcurve
        data.append(row_dict)

    df = pd.DataFrame(data=data)
    if len(df) > 0 and data_path is not None:
        df = df.loc[df.data_path == data_path]
    return df


def get_dimention(value):
    dim = 1
    try:
        dim = len(value)
    except TypeError:
        pass
    return dim


def requrie_fcurves(obj, data_path):
    curr_data = get_data_path_attr(obj, data_path)
    # string can not be animated
    assert(not isinstance(curr_data, str))
    num_curves = get_dimention(curr_data)

    exisitng_df = get_fcurves_dataframe(obj, data_path)

    if len(exisitng_df) == num_curves:
        return exisitng_df
    elif len(exisitng_df) == 0:
        # create curves
        action = require_action(obj)
        _ = [action.fcurves.new(data_path=data_path, index=i) for i in range(num_curves)]
        new_df = get_fcurves_dataframe(obj, data_path)
        assert(len(new_df) == num_curves)
        return new_df
    else:
        # corrupted fcurve!
        raise RuntimeError("Corrupted fcurve: {}".format(data_path))


def remove_fcurves(obj, data_path=None):
    """
    If data_path is not provided, remove all fcurves
    """
    action = require_action(obj)
    df = get_fcurves_dataframe(obj, data_path=data_path)
    if len(df) > 0:
        for fcurve in df.fcurve:
            action.fcurves.remove(fcurve)


# high level API
# ==============
def get_keyframes_dataframe(fcurve):
    """
    each value is always a scalar
    """
    data = {
        'frame': list(),
        'value': list(),
        'keyframe': list()
    }
    for keyframe in fcurve.keyframe_points:
        frame, value = keyframe.co
        data['frame'].append(int(frame))
        data['value'].append(value)
        data['keyframe'].append(keyframe)
    df = pd.DataFrame(data)
    return df


def build_animation_dataframe(frames, values):
    """
    each value can be a vector
    """
    assert(len(frames) == len(values))
    data = {
        'frame': frames,
        'value': values
    }
    df = pd.DataFrame(data=data)
    return df


def get_animation_dataframe(obj, data_path):
    fcurves_df = requrie_fcurves(obj, data_path=data_path)
    keyframe_dfs = [get_keyframes_dataframe(x) for x in fcurves_df.fcurve]
    frames = keyframe_dfs[0].frame
    for df in keyframe_dfs:
        if not (df.frame == frames).all():
            raise RuntimeError('Corrupted fcurve: {}'.format(data_path))
    
    values = np.vstack([np.array(df.value) for df in keyframe_dfs]).T
    data = {
        'frame': frames,
        'value': values.tolist()
    }
    result_df = pd.DataFrame(data=data, index=frames.index)
    return result_df


def set_animiation(obj, data_path, animation_df):
    remove_fcurves(obj, data_path=data_path)
    fcurves_df = requrie_fcurves(obj, data_path=data_path)
    
    # add keyframes
    for fcurve in fcurves_df.fcurve:
        fcurve.keyframe_points.add(len(animation_df))
    # set values
    for idx, row in animation_df.iterrows():
        frame = int(row.frame)
        value = row.value
        dim = 1
        try:
            dim = len(value)
        except TypeError:
            value = [value]
        for i in range(dim):
            fcurve = fcurves_df.loc[fcurves_df.array_index == i, 'fcurve'].iloc[0]
            fcurve.keyframe_points[idx].co = (float(frame), value[i])
