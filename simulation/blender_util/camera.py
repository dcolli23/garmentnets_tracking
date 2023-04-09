import bpy
from mathutils import Matrix, Vector
import numpy as np

from blender_util.collection import get_camera_collection
from blender_util.camera_matrix import (
    get_3x4_RT_matrix_from_blender, 
    get_calibration_matrix_K_from_blender, 
    get_3x4_P_matrix_from_blender)


def _R_cv_bcam():
    R_cv_bcam = np.array(
        ((1, 0,  0),
            (0, -1, 0),
            (0, 0, -1)))
    return R_cv_bcam


def require_camera(name='main_camera'):
    # create camera
    if name not in bpy.data.cameras:
        bpy.data.cameras.new(name)
    camera = bpy.data.cameras[name]

    # create camera object
    if name not in bpy.data.objects:
        bpy.data.objects.new(name, camera)
    camera_obj = bpy.data.objects[name]

    # link to camera collection
    camera_collection = get_camera_collection()
    if name not in camera_collection.objects:
        camera_collection.objects.link(camera_obj)
    
    bpy.context.scene.camera = camera_obj
    return camera_obj

def get_camera_extrinsic(camera_obj):
    """
    CV extrinsic: camera frame <- world frame
    Blender matrix_world: world frame <- camera frame
    """
    # blender's camera points to -z axis
    # with y up and x rigth
    # this function converts blender convention
    # to computer vision convention
    bmat = get_3x4_RT_matrix_from_blender(camera_obj)
    mat = np.concatenate(
        [np.array(bmat), np.array([[0, 0, 0, 1]])],
        axis=0)
    return mat

def set_camera_extrinsic(camera_obj, extrinsic):
    mat = np.array(extrinsic)
    assert(mat.shape == (4, 4))

    R_cv_bcam = _R_cv_bcam()

    Tx_cv_world = extrinsic
    Tx_world_cv = np.linalg.inv(Tx_cv_world)

    R_world_cv = Tx_world_cv[:3, :3]
    T_world_cv = Tx_world_cv[:3, 3]

    R_world_bcam = R_world_cv @ R_cv_bcam
    T_world_bcam = T_world_cv

    blender_mat = np.eye(4)
    # inverse of unitary matrix
    blender_mat[:3, :3] = R_world_bcam
    blender_mat[:3, 3] = T_world_bcam

    bmat = Matrix(blender_mat)
    camera_obj.matrix_world = bmat

def generate_intrinsic(resolution_x, resolution_y, f_pixel):
    intrinsic = np.array([
        [f_pixel, 0, resolution_x/2],
        [0, f_pixel, resolution_y/2],
        [0, 0, 1]
    ], dtype=np.float64)
    return intrinsic

def get_camera_intrinsic(camera_obj):
    K = get_calibration_matrix_K_from_blender(camera_obj.data)
    intrinsic = np.array(K)
    return intrinsic

def set_camera_intrinsic(camera_obj, intrinsic):
    assert(np.allclose(intrinsic[0][0], intrinsic[1][1]))
    focal_length_pixel = intrinsic[0][0]

    # assume optical center is the center of frame
    resolution_pixel = np.rint(intrinsic[:2, 2] * 2).astype(np.int64)
    assert(np.all(resolution_pixel > 0))

    # set resolution
    scene = bpy.context.scene
    scene.render.resolution_x = resolution_pixel[0]
    scene.render.resolution_y = resolution_pixel[1]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0

    # set lens focal length
    camera_obj.data.sensor_fit = 'HORIZONTAL'
    width_scale = camera_obj.data.sensor_width / resolution_pixel[0]
    camera_obj.data.lens = focal_length_pixel * width_scale

def get_camera_projection_matrix(camera_obj):
    P, _, _ = get_3x4_P_matrix_from_blender(camera_obj)
    projection_matrix = P
    return projection_matrix

def get_camera_focus_point(camera_obj):
    focus_distance = camera_obj.data.dof.focus_distance
    focus_point_cam_space = np.array([[0, 0, focus_distance, 1]]).T
    extrinsic = get_camera_extrinsic(camera_obj)
    focus_point_world_space = extrinsic @ focus_point_cam_space
    focus_point = focus_point_world_space[:3] / focus_point_world_space[3]
    return focus_point

def set_camera_focus_point(camera_obj, focus_point, horizontal=True):
    if horizontal is False:
        raise NotImplementedError()

    focus_point = np.array(focus_point)
    assert(focus_point.shape == (3,))

    location = np.array(camera_obj.location)

    assert(not np.allclose(focus_point, location))
    focus_point_local = focus_point - location
    focus_distance = np.linalg.norm(focus_point_local)

    z_local = focus_point_local / focus_distance
    z_global = np.array([0, 0, 1])

    x_local = np.array([1, 0, 0])
    if not np.allclose(np.abs(z_local[2]), 1):
        x_local = np.cross(z_local, z_global)
        x_local /= np.linalg.norm(x_local)

    y_local = np.cross(z_local, x_local)

    Tx_world_cv = np.eye(4)
    Tx_world_cv[:3, 0] = x_local
    Tx_world_cv[:3, 1] = y_local
    Tx_world_cv[:3, 2] = z_local
    Tx_world_cv[:3, 3] = location

    Tx_cv_world = np.linalg.inv(Tx_world_cv)
    # set
    camera_obj.data.dof.focus_distance = focus_distance
    set_camera_extrinsic(camera_obj, Tx_cv_world)
