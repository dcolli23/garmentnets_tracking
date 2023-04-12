import bpy
import numpy as np

def print_obj_keyframe_coordinates(obj: bpy.types.Object):
    loc_idx_map = {0: 'x', 1: 'y', 2: 'z'}
    locs = {'x': [], 'y': [], 'z': []}
    keyframe_frames = {'x': [], 'y': [], 'z': []}

    for fcurve in obj.animation_data.action.fcurves:
        axis_key = loc_idx_map[fcurve.array_index]
        coordinate_locations = locs[axis_key]
        coordinate_frames = keyframe_frames[axis_key]

        for pt in fcurve.keyframe_points:
            # TODO: The pt.co is two values instead of one? Maybe the first value is the frame.
            # The second value is definitely the cartesian coordinate.
            coordinate_frames.append(pt.co[0])
            coordinate_locations.append(pt.co[1])
            
    keyframe_frames = {k: np.array(v) for k, v in keyframe_frames.items()}
    xy_same_frames = np.allclose(keyframe_frames['x'], keyframe_frames['y'])
    yz_same_frames = np.allclose(keyframe_frames['y'], keyframe_frames['z'])
    if (not xy_same_frames) or (not yz_same_frames):
        print("Detected that X, Y, and Z keyframe frames are not all the same. Is this on purpose?")
        print("If trying to animate a curve, then this makes sense.")
    
    for axis_key in ['x', 'y', 'z']:
        print("Axis:", axis_key)
        print("\tFrames:", keyframe_frames[axis_key])
        print("\tCoordinates:", locs[axis_key])

def print_obj_fcurve_information(obj: bpy.types.Object):
    for i, fcurve in enumerate(obj.animation_data.action.fcurves):
        print(f"Fcurve #{i}")
        print("\tGroup name:", fcurve.group.name)
        print("\tarray_index:", fcurve.array_index)
        print("\tData the FCurve is animating:", fcurve.data_path)
