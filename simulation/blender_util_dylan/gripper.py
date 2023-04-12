
import bpy
import mathutils
import numpy as np

from simulation.blender_util_dylan.modifiers import make_modifier_highest_priority

def require_virtual_gripper(obj: bpy.types.Object):
    """Adds a virtual gripper ('empty' object) if the given object doesn't already have one"""
    all_obj_names = [o.name for o in bpy.data.objects]
    if "Empty" not in all_obj_names:
        print(f"Virtual gripper not found in given object, '{obj.name}'. Creating one.")
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.editmode_toggle()
        bpy.ops.object.vertex_group_set_active(group="pin")
        bpy.ops.object.hook_add_newob()
        bpy.ops.object.editmode_toggle()

        # Check that the new "Empty" object was created.
        all_obj_names = [o.name for o in bpy.data.objects]
        assert ("Empty" in all_obj_names)

        # Move the newly created hook modifier to be above the CLOTH modifier so that it takes
        # precedent. This is required to be able to move the cloth like its held by a gripper.
        make_modifier_highest_priority("Hook-Empty")
    else:
        print(f"Virtual gripper found in given object, '{obj.name}'. Not creating one.")

class GripperAnimation:
    """Class for easily defining piecewise linear gripper control with Bezier interpolation"""
    gripper_obj: bpy.types.Object
    grasped_obj: bpy.types.Object
    animation_frame_prev: int
    animation_loc_prev: mathutils.Vector
    blender_fps: float
    # Adding this as actions should have distinct names.
    __fcurve_group_num: int = 1

    def __init__(self, gripper_obj: bpy.types.Object, grasped_obj: bpy.types.Object):
        self.gripper_obj = gripper_obj
        self.grasped_obj = grasped_obj
        self.blender_fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base

        self.__fcurve_group = f"gripper_action_{GripperAnimation.__fcurve_group_num}"
        GripperAnimation.__fcurve_group_num += 1

        # These are updated in self.__add_init_keyframe_if_none_exist()
        self.animation_frame_prev = None
        self.animation_loc_prev = None

        self.__add_init_keyframe_if_none_exist()

    def add_movement(self, direction_vec: np.ndarray, velocity_mps: float, frame_duration: int):
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
        frame_duration : int
            The number of frames this movement should take.
        """
        # Find where the object will be at end_frame given the unit direction vector and velocity
        end_frame = frame_duration + self.animation_frame_prev
        velocity_mpf = velocity_mps / self.blender_fps  # [meters / frame]
        loc_delta = mathutils.Vector(direction_vec * velocity_mpf * frame_duration)
        loc_end = self.animation_loc_prev + loc_delta

        # Set the current frame to `end_frame` and move the object since keyframe_insert pulls 
        # location data from the object's current location.
        bpy.context.scene.frame_set(end_frame)
        self.animation_loc_prev = self.gripper_obj.location = loc_end

        self.gripper_obj.keyframe_insert(data_path="location", group=self.__fcurve_group,
                                         frame=end_frame)

        self.animation_frame_prev = end_frame

        # Update the physics sim end frame so that we always simulate up to the last keyframe.
        self.grasped_obj.modifiers['CLOTH'].point_cache.frame_end = end_frame


    def __add_init_keyframe_if_none_exist(self, frame_start=0):
        # if self.obj. # no keyframes present:
        if self.gripper_obj.animation_data is None:
            # This means that there is no animation associated with the object so we need to add an 
            # initial keyframe at `frame_start` for the object's current location.
            self.gripper_obj.keyframe_insert(data_path="location", frame=frame_start,
                                             group=self.__fcurve_group)
            self.animation_frame_prev = frame_start
        else:
            # Grab the frame of the last keyframe. NOTE: For some reason this is a float but Blender
            # expects input to bpy.context.frame_set to be an int.
            self.animation_frame_prev = int(self.gripper_obj.animation_data.action.frame_end)

        # Set the active frame to the animation start frame so we can easily grab object location.
        print("Starting gripper keyframe frame:", self.animation_frame_prev)
        bpy.context.scene.frame_set(self.animation_frame_prev)
        self.animation_loc_prev = self.gripper_obj.location
        
        