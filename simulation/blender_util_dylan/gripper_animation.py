
import bpy
import mathutils
import numpy as np

class GripperAnimation:
    """Class for easily defining piecewise linear gripper control with Bezier interpolation"""
    obj: bpy.types.Object
    animation_frame_prev: int
    animation_loc_prev: mathutils.Vector
    blender_fps: float
    # Adding this as actions should have distinct names.
    __fcurve_group_num: int = 1

    def __init__(self, obj: bpy.types.Object):
        self.obj = obj
        self.blender_fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base

        self.__fcurve_group = f"gripper_action_{GripperAnimation.__fcurve_group_num}"
        GripperAnimation.__fcurve_group_num += 1

        # These are updated in self.__add_init_keyframe_if_none_exist()
        self.animation_frame_prev = None
        self.animation_loc_prev = None

        self.__add_init_keyframe_if_none_exist()

    def add_movement(self, direction_vec: np.ndarray, velocity_mps: float, end_frame: int):
        """Adds a movement to the gripper animation by adding a KeyFrame at the ending Frame
        
        TODO: This may get slow for gripper animations involving lots of keyframes as we update the
        scene every time we insert a movement. This can be refactored to only insert keyframes and
        change the locations of the keyframes without updating the scene for speedup.

        Parameters
        ----------
        direction_vec : np.ndarray
            Normalized vector indicating direction of linear movement.
        velocity_mps : float
            Velocity of motion in meters / second.
        end_frame : int
            The frame with which to insert the keyframe.
        """
        # Assert that the given end_frame is after the last keyframe in the action sequence.
        # I don't believe this is actually necessary but it is a code smell since I can't think of a 
        # good reason to insert actions out of order.
        assert (end_frame > self.animation_frame_prev), "Must insert keyframe at frame *after* the current last keyframe"

        # Find where the object will be at end_frame given the unit direction vector and velocity
        frame_range = end_frame - self.animation_frame_prev
        
        # Convert the given velocity from [meters / second] to [meters / frame] and find final 
        # location.
        velocity_mpf = velocity_mps / self.blender_fps
        loc_delta = mathutils.Vector(direction_vec * velocity_mpf * frame_range)
        loc_end = self.animation_loc_prev + loc_delta

        # Set the current frame to `end_frame` and move the object since keyframe_insert pulls 
        # location data from the object's current location.
        bpy.context.scene.frame_set(end_frame)
        self.animation_loc_prev = self.obj.location = loc_end

        self.obj.keyframe_insert(data_path="location", group=self.__fcurve_group, frame=end_frame)

        self.animation_frame_prev = end_frame

    def __add_init_keyframe_if_none_exist(self, frame_start=0):
        # if self.obj. # no keyframes present:
        if self.obj.animation_data is None:
            # This means that there is no animation associated with the object so we need to add an 
            # initial keyframe at `frame_start` for the object's current location.
            self.obj.keyframe_insert(data_path="location", frame=frame_start, group=self.__fcurve_group)
            self.animation_frame_prev = frame_start
        else:
            # Grab the frame of the last keyframe. NOTE: For some reason this is a float but Blender
            # expects input to bpy.context.frame_set to be an int.
            self.animation_frame_prev = int(self.obj.animation_data.action.frame_end)

        # Set the active frame to the animation start frame so we can easily grab object location.
        print("Starting gripper keyframe frame:", self.animation_frame_prev)
        bpy.context.scene.frame_set(self.animation_frame_prev)
        self.animation_loc_prev = self.obj.location
        
        