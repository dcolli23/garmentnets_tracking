# %%
# import
import bpy
import pathlib
import pickle
import os
from common.geometry_util import build_rectangle, faces_to_edges
from blender_util.camera import (
    require_camera, 
    get_camera_extrinsic,
    set_camera_extrinsic,
    get_camera_intrinsic, 
    set_camera_intrinsic,
    set_camera_focus_point,
    generate_intrinsic)
from blender_util.collection import remove_all_collections, get_mesh_collection
from blender_util.mesh import (
    NumpyBMeshManager, DepsGraphBMeshManager, 
    mesh_animation_to_numpy, mesh_to_numpy,
    require_mesh, create_cube, set_material,
    get_soft_weight, set_vertex_group_weight)
from blender_util.physics import (
    setup_cloth_gripping_modifiers,
    setup_collision_modifiers,
    require_subdiv_modifier,
    run_simulation, ClothModifier)
from blender_util.modifier import (
    append_modifier,
    require_modifier_stack,
    ClothModifier,
    VertexWeigthMixModifier,
    ArmatureModifier,
    CollisionModifier,
    SubdivModifier
)
from blender_util.material import (
    require_material,
    clear_materials,
    get_world_material,
    setup_black_world_material,
    setup_hdri_world_material,
    setup_metal_materail, 
    setup_plastic_material, 
    setup_uv_material, 
    setup_white_material,
    setup_textured_bsdf_material,
    require_image)
from blender_util.compositor import (
    setup_trivial_compositor, 
    setup_uviz_compositor)
from blender_util.render import (
    setup_color_management_raw,
    setup_color_management_srgb,
    setup_cycles,
    setup_eevee,
    get_cycles_uviz_config,
    get_eevee_rgb_config,
    setup_png_output,
    setup_exr_output,
    render_animation
)
from blender_util.armature import (
    require_armature, require_bone, require_vertex_group
)
from blender_util.animation import (
    build_animation_dataframe, set_animiation
)
from blender_util.gripping_util import (
    setup_gripper_movement, setup_gripper_effect, setup_gripper_vertex_groups
)
from cloth_3d_util.read import DataReader
from cloth_3d_util.util import quads2tris, axis_angle_to_matrix
from cloth_3d_util.IO import readOBJ
import skimage.io
import numpy as np
from scipy.spatial.ckdtree import cKDTree
from scipy.spatial.transform import Rotation
import networkx as nx
import time
from mathutils import Quaternion

# %%
# helper functions
class Cloth3DCanonicalAccessor:
    def __init__(self, dataset_path="/home/cchi/dev/blender_experiment/data/CLOTH3D/train"):
        self.reader = DataReader(dataset_path)
        self.default_frame = 0
    
    def read_human(self, sample):
        reader = self.reader
        frame = self.default_frame
        _, human_faces = reader.read_human(sample, frame, absolute=False)

        gender, pose, shape, trans = reader.read_smpl_params(sample, frame)
        rest_pose = np.zeros((24, 3), dtype=np.float32)
        rest_pose[0, 0] = np.pi / 2  # y-up to z-up
        rest_pose[1, 2] = 0.15
        rest_pose[2, 2] = -0.15

        V, J = reader.smpl[gender].set_params(pose=rest_pose, beta=shape, trans=None)
        human_verts = V - J[0:1]
        data = {
            'verts': human_verts,
            'faces': human_faces
        }
        return data
    
    def get_garment_names(self, sample):
        reader = self.reader
        info = reader.read_info(sample)
        garment_names = list(info['outfit'].keys())
        return garment_names

    def read_cloths(self, sample, garment_name):
        reader = self.reader
        info = reader.read_info(sample)
        zrot = info['zrot']
        rx_zrot = axis_angle_to_matrix(np.array([0, 0, -zrot]))

        obj_path = os.path.join(reader.SRC, sample, garment_name + '.obj')
        garment_verts_rotated, garment_faces, uv_verts, uv_faces = readOBJ(obj_path)
        garment_verts = garment_verts_rotated @ rx_zrot.T
        data = {
            'verts': garment_verts,
            'faces': garment_faces,
            'uv_verts': uv_verts,
            'uv_faces': uv_faces
        }
        return data
    
    def read_cloth_texture(self, sample, garment_name):
        reader = self.reader
        texture_path = os.path.join(reader.SRC, sample, garment_name + '.png')
        texture = skimage.io.imread(texture_path)
        return texture

    def get_sample_data(self, sample_id, garment_name, **kwargs):
        accessor = self
        reader = accessor.reader
        # load cloth and human from dataset
        cloth_data = accessor.read_cloths(sample_id, garment_name)
        human_data = accessor.read_human(sample_id)
        # load cloth texture etc
        info = reader.read_info(sample_id)
        garment_info = info['outfit'][garment_name]
        texture_type = garment_info['texture']['type']
        texture = None
        if texture_type == 'color':
            color = garment_info['texture']['data']
            color_int = (color * 255).astype(np.uint8)
            texture = np.empty((2048, 2048, 3), dtype=np.uint8)
            texture[:,:,:] = color_int
        else:
            texture = accessor.read_cloth_texture(sample_id, garment_name)

        data = {
            'human_verts': human_data['verts'],
            'human_faces': human_data['faces'],
            'human_gender': info['gender'],
            'garment_verts': cloth_data['verts'],
            'garment_faces': cloth_data['faces'],
            'garment_uv_verts': cloth_data['uv_verts'],
            'garment_uv_faces': cloth_data['uv_faces'],
            'garment_fabric': garment_info['fabric'],
            'garment_texture': texture
        }
        return data


def get_garments_df(
    path='/home/cchi/dev/blender_experiment/data/cloth_3d_index/garments_df_v2.pk'):
    return pickle.load(open(path, 'rb'))


# %%
# setup parameters for debugging
def _setup():
    # %%
    import os
    os.chdir(os.path.expanduser("~/dev/blender_experiment"))

    # %%
    accessor = Cloth3DCanonicalAccessor()
    garments_df = get_garments_df()

    selected_row = garments_df.iloc[0]
    sample_config = {
        'sample_id': selected_row.sample_id,
        'garment_name': selected_row.garment_name,
        'grip_vertex_idx': 0
    }

    sample_id = sample_config['sample_id']
    garment_name = sample_config['garment_name']
    grip_vertex_idx = sample_config['grip_vertex_idx']

    sample_data = accessor.get_sample_data(**sample_config)

    output_path = '/home/cchi/dev/blender_experiment/data/cloth_3d_output/test'
    garment_verts = sample_data['garment_verts']
    garment_faces = sample_data['garment_faces']
    garment_uv_verts = sample_data['garment_uv_verts']
    garment_uv_faces = sample_data['garment_uv_faces']
    garment_texture = sample_data['garment_texture']

    simulation_duration_pair=(0, 120)


# %%
# pipeline function
def smpl_simulation_pipeline(
        # garment data
        garment_verts, garment_faces, 
        garment_uv_verts, garment_uv_faces, 
        garment_texture,
        # simulation parameters
        grip_vertex_idx,
        simulation_duration_pair,
        **kwargs
    ):

    # %%
    # clear existing state
    remove_all_collections()

    # %%
    # load mesh
    cloth_obj = require_mesh('cloth')
    with NumpyBMeshManager(
        verts=garment_verts, faces=garment_faces, 
        uv=garment_uv_verts, uv_faces=garment_uv_faces) as bm:
        bm.to_mesh(cloth_obj.data)
    
    # %%
    # move mesh such that gripping point is at origin
    bpy.context.scene.frame_set(0)
    grip_normal = None
    deps_graph = deps_graph = bpy.context.evaluated_depsgraph_get()
    with DepsGraphBMeshManager(cloth_obj, deps_graph) as bm:
        bm.normal_update()
        grip_normal = np.array(bm.verts[grip_vertex_idx].normal)
    assert(np.linalg.norm(grip_normal) > 0.1)

    cross_vec = np.cross(grip_normal, np.array([0,0,1]))
    magnitude = np.clip(np.linalg.norm(cross_vec), 0, 1)
    rot_rad = np.arcsin(magnitude)
    rot_vec = cross_vec * (-rot_rad / magnitude)
    rotation = Rotation.from_rotvec(rot_vec)
    tx = np.eye(4)
    tx[:3, :3] = rotation.as_matrix()
    cloth_obj.matrix_world = tx

    data = mesh_to_numpy(cloth_obj, global_coordinate=True)
    verts = data['verts']
    faces = data['faces']
    
    grip_location = verts[grip_vertex_idx]
    cloth_obj.location = -grip_location

    # %%
    # setup gripping simulation
    pin_group_name = 'pin'
    pin_weights = get_soft_weight(
        verts, faces, 
        vert_idxs=[grip_vertex_idx], radius=0.01)
    pin_group = set_vertex_group_weight(cloth_obj, pin_group_name, pin_weights)

    # setup simulation modifier
    class_stack = [ClothModifier]
    stack = require_modifier_stack(cloth_obj, class_stack)
    cloth_modifier_wrapper = stack[0]
    cloth_modifier = cloth_modifier_wrapper.modifier
    
    # setup simulation parameters
    cloth_modifier.settings.quality = 10
    cloth_modifier.settings.mass = 0.05
    cloth_modifier.settings.air_damping = 2.0
    cloth_modifier.collision_settings.collision_quality = 5
    cloth_modifier.collision_settings.use_collision = False
    cloth_modifier.collision_settings.use_self_collision = True
    cloth_modifier.collision_settings.self_distance_min = 0.001

    cloth_modifier.settings.vertex_group_mass = pin_group_name

    # %% 
    # run simulation
    cloth_point_cache = cloth_modifier_wrapper.get_point_cache()
    cloth_point_cache.frame_start = simulation_duration_pair[0]
    cloth_point_cache.frame_end = simulation_duration_pair[1]
    cloth_point_cache.use_disk_cache = False
    run_simulation(cloth_obj, cloth_point_cache)

    # %%
    # get ground truth mesh
    bpy.context.scene.frame_set(simulation_duration_pair[1])
    simulated_mesh_data = mesh_to_numpy(cloth_obj, global_coordinate=True)

    # %%
    result_data = {
        'cloth_state': simulated_mesh_data,
        'grip_vertex_idx': grip_vertex_idx
    }
    return result_data
