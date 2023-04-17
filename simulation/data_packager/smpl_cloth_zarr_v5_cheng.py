# %%
# Change CWD (Cheng's workflow)
# Instead, use sys to append paths the the system path to find all modules.
import os
import sys
FILE_ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(FILE_ROOT, ".."))
# os.chdir(os.path.expanduser("~/dev/blender_experiment"))
# os.chdir(os.path.expanduser("~/code/external_packages/"))


# %%
# set numpy threads
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np

# %%
# import
import dask
import dask.delayed
import dask.bag as db
from dask.diagnostics import ProgressBar
import pathlib
import pyexr
import skimage.io
import numpy as np
import pandas as pd
import time
import zarr
from numcodecs import Blosc
import pickle
import skimage.io
from scipy.spatial.ckdtree import cKDTree
import traceback
import functools
import itertools

from common.projection import ray_length_to_zbuffer, zbuffer_to_pcloud
from common.geometry_util import (barycentric_interpolation, get_aabb, get_union_aabb)
from common.igl_util import query_uv_barycentric
from io_util.image_io_util import read_uviz
from common.parallel_util import parallel_map


# %%
# helper functions
import os
import numpy as np
from scipy.spatial.ckdtree import cKDTree
import skimage.io

from cloth_3d_util.read import DataReader
from cloth_3d_util.util import quads2tris, axis_angle_to_matrix
from cloth_3d_util.IO import readOBJ

from data_packager.cloth_3d_canonical_accessor import Cloth3DCanonicalAccessor

# class Cloth3DCanonicalAccessor:
#     def __init__(self, dataset_path="/home/cchi/dev/blender_experiment/data/CLOTH3D/train"):
#         self.reader = DataReader(dataset_path)
#         self.default_frame = 0

#     def read_human(self, sample):
#         reader = self.reader
#         frame = self.default_frame
#         _, human_faces = reader.read_human(sample, frame, absolute=False)

#         gender, pose, shape, trans = reader.read_smpl_params(sample, frame)
#         rest_pose = np.zeros((24, 3), dtype=np.float32)
#         rest_pose[0, 0] = np.pi / 2  # y-up to z-up
#         rest_pose[1, 2] = 0.15
#         rest_pose[2, 2] = -0.15

#         V, J = reader.smpl[gender].set_params(pose=rest_pose, beta=shape, trans=None)
#         human_verts = V - J[0:1]
#         data = {
#             'verts': human_verts,
#             'faces': human_faces
#         }
#         return data

#     def get_garment_names(self, sample):
#         reader = self.reader
#         info = reader.read_info(sample)
#         garment_names = list(info['outfit'].keys())
#         return garment_names

#     def read_cloths(self, sample, garment_name):
#         reader = self.reader
#         info = reader.read_info(sample)
#         zrot = info['zrot']
#         rx_zrot = axis_angle_to_matrix(np.array([0, 0, -zrot]))

#         obj_path = os.path.join(reader.SRC, sample, garment_name + '.obj')
#         garment_verts_rotated, garment_faces, uv_verts, uv_faces = readOBJ(obj_path)
#         garment_verts = garment_verts_rotated @ rx_zrot.T
#         data = {
#             'verts': garment_verts,
#             'faces': garment_faces,
#             'uv_verts': uv_verts,
#             'uv_faces': uv_faces
#         }
#         return data

#     def read_cloth_texture(self, sample, garment_name):
#         reader = self.reader
#         texture_path = os.path.join(reader.SRC, sample, garment_name + '.png')
#         texture = skimage.io.imread(texture_path)
#         return texture

#     @functools.lru_cache(maxsize=32)
#     def get_sample_data(self, sample_id, garment_name, **kwargs):
#         accessor = self
#         reader = accessor.reader
#         # load cloth and human from dataset
#         cloth_data = accessor.read_cloths(sample_id, garment_name)
#         human_data = accessor.read_human(sample_id)
#         # load cloth texture etc
#         info = reader.read_info(sample_id)
#         garment_info = info['outfit'][garment_name]
#         texture_type = garment_info['texture']['type']
#         texture = None
#         if texture_type == 'color':
#             color = garment_info['texture']['data']
#             color_int = (color * 255).astype(np.uint8)
#             texture = np.empty((2048, 2048, 3), dtype=np.uint8)
#             texture[:,:,:] = color_int
#         else:
#             texture = accessor.read_cloth_texture(sample_id, garment_name)

#         data = {
#             'human_verts': human_data['verts'],
#             'human_faces': human_data['faces'],
#             'human_gender': info['gender'],
#             'garment_verts': cloth_data['verts'],
#             'garment_faces': cloth_data['faces'],
#             'garment_uv_verts': cloth_data['uv_verts'],
#             'garment_uv_faces': cloth_data['uv_faces'],
#             'garment_fabric': garment_info['fabric'],
#             'garment_texture': texture
#         }
#         return data


def get_pixel_keypoint_idx_from_uv(
    uv_img_tensor, mask_tensor,
    uv_verts, uv_faces, faces,
    vert_idx_keypoint_idx_map):
    assert(uv_faces.shape == faces.shape)
    uv_vert_to_vert_idx = np.zeros((uv_verts.shape[0],), dtype=np.int32)
    uv_vert_to_vert_idx[uv_faces.flatten()] = faces.flatten()
    # a bug in data pipeline generates incorrect uv_verts
    # use the data from accessor instead
    uv_verts_kdtree = cKDTree(uv_verts.astype(uv_img_tensor.dtype), copy_data=True)

    # for each pixel, find cooresponding uv vert index
    assert(mask_tensor.shape == uv_img_tensor.shape[:-1])
    query_uv = uv_img_tensor[mask_tensor]
    query_dist, uv_nn_idx = uv_verts_kdtree.query(query_uv, k=1, n_jobs=1)
    vert_nn_idx = uv_vert_to_vert_idx[uv_nn_idx]
    keypoint_nn_idx = vert_idx_keypoint_idx_map[vert_nn_idx]
    keypoint_idx_img_arr = np.ones(mask_tensor.shape, dtype=np.int32)
    keypoint_idx_img_arr *= -1
    keypoint_idx_img_arr[mask_tensor] = keypoint_nn_idx
    return keypoint_idx_img_arr

def get_pixel_nn_idx_from_uv(
    uv_img_tensor, mask_tensor,
    uv_verts, uv_faces, faces,
    vert_idx_nn_idx_map):
    assert(uv_faces.shape == faces.shape)
    uv_vert_to_vert_idx = np.zeros((uv_verts.shape[0],), dtype=np.int32)
    uv_vert_to_vert_idx[uv_faces.flatten()] = faces.flatten()
    # a bug in data pipeline generates incorrect uv_verts
    # use the data from accessor instead
    uv_verts_kdtree = cKDTree(uv_verts.astype(uv_img_tensor.dtype), copy_data=True)

    # for each pixel, find cooresponding uv vert index
    assert(mask_tensor.shape == uv_img_tensor.shape[:-1])
    query_uv = uv_img_tensor[mask_tensor]
    query_dist, uv_nn_idx = uv_verts_kdtree.query(query_uv, k=1, n_jobs=1)
    vert_nn_idx = uv_vert_to_vert_idx[uv_nn_idx]
    nn_idx = vert_idx_nn_idx_map[vert_nn_idx]
    nn_idx_img_arr = np.ones(mask_tensor.shape, dtype=np.int32)
    nn_idx_img_arr *= -1
    nn_idx_img_arr[mask_tensor] = nn_idx
    return nn_idx_img_arr


# %%
# Dylan: Probably ran on a cluster or something.
# specify input
workspace_dir = "/local/crv/cchi/data/cloth_3d_workspace"
cloth_3d_path = os.path.join(workspace_dir, 'CLOTH3D', 'train')
render_packets_dir = os.path.join(workspace_dir, 'render_packet_jsons')
render_output_root = os.path.join(workspace_dir, 'simulation_output')

# %%
# orgnize data
import json
from tqdm import tqdm

# check completion status
json_packet_paths = sorted(pathlib.Path(render_packets_dir).glob("*.json"), key=lambda x: int(x.stem))
rows = list()
for json_path in tqdm(json_packet_paths):
    packet_id = int(json_path.stem)
    packet_dict = json.load(json_path.open('r'))
    for sample_config in packet_dict['sample_configs']:
        sample_dir_name = "_".join(str(sample_config[x]) for x in ['sample_id', 'garment_name', 'grip_vertex_idx'])
        sample_dir = pathlib.Path(render_output_root).joinpath(str(packet_id), sample_dir_name)
        sim_result_file = sample_dir.joinpath('simulation_result.pk')
        render_result_file = sample_dir.joinpath('meta.pk')

        row = sample_config
        row['packet_id'] = packet_id
        row['simulation_done'] = sim_result_file.exists()
        row['rendering_done'] = render_result_file.exists()
        rows.append(row)

sample_config_df = pd.DataFrame(rows, index=pd.RangeIndex(len(rows)))
sample_config_df['idx'] = sample_config_df.index

# group by instance
finished_sample_df = sample_config_df.loc[sample_config_df.rendering_done]
sample_agg_df = finished_sample_df.groupby(['sample_id', 'garment_name']).agg({
    'idx': list
})
sample_agg_df['num_samples'] = sample_agg_df.idx.apply(lambda x: len(x))
print(sample_agg_df.num_samples.min())

# select grips
grip_per_cloth = sample_agg_df.num_samples.min()
selected_sample_agg_df = sample_agg_df.copy()
selected_sample_agg_df['idx'] = selected_sample_agg_df.idx.apply(lambda x: x[:grip_per_cloth])
selected_sample_agg_df['num_samples'] = selected_sample_agg_df.idx.apply(lambda x: len(x))
assert (selected_sample_agg_df.num_samples == grip_per_cloth).all()
selected_samples_df = selected_sample_agg_df.explode('idx')
selected_samples_df['idx'] = selected_samples_df['idx'].astype('int64')

# group by type
category_agg_df = selected_samples_df.reset_index(
    drop=False).sort_values(
        ['garment_name', 'sample_id', 'idx']).drop('num_samples', axis=1)
category_agg_df.set_index('idx', drop=True, inplace=True)
all_samples_df = pd.merge(
    category_agg_df, sample_config_df[['grip_vertex_idx', 'packet_id']],
    how='left', left_index=True, right_index=True)
all_samples_df.reset_index(drop=False, inplace=True)
all_samples_df.set_index(['garment_name', 'sample_id', 'idx'],
    drop=True, inplace=True, verify_integrity=True)
all_samples_df['category'] = all_samples_df.index.get_level_values('garment_name')
all_samples_df['sample_id'] = all_samples_df.index.get_level_values('sample_id')

# add dir
all_samples_df['sample_dir'] = all_samples_df.apply(
    lambda row: pathlib.Path(render_output_root).joinpath(
        str(row.packet_id), '_'.join(str(row[x]) for x in ['sample_id', 'category', 'grip_vertex_idx'])),
    axis=1)

# %%
# setup datasets
cloth_3d_path = '/local/crv/cchi/data/cloth_3d_workspace/CLOTH3D/train'
dataset_root_dir = "/local/crv/cchi/data/cloth_3d_workspace/cloth_3d_grip_dataset"
compressor = Blosc(cname='zstd', clevel=6, shuffle=Blosc.BITSHUFFLE)

# categories = sorted(all_samples_df.index.get_level_values('garment_name').unique())
categories = ['Dress','Jumpsuit','Skirt','Top','Trousers','Tshirt']
accessor = Cloth3DCanonicalAccessor(cloth_3d_path)

rows = dict()
for category in categories:
    store = zarr.DirectoryStore(os.path.join(dataset_root_dir, category))
    root = zarr.group(store=store, overwrite=False)
    sample_root = root.require_group('samples', overwrite=False)
    summary_root = root.require_group('summary', overwrite=False)
    rows[category] = {
        'compressor': compressor,
        'store': store,
        'root': root,
        'sample_root': sample_root,
        'summary_root': summary_root
    }
datasets_df = pd.DataFrame(list(rows.values()), index=rows.keys())

# generate input dicts
generation_input_df = all_samples_df.copy()
generation_input_df['sample_root'] = generation_input_df.category.apply(
    lambda x: datasets_df.sample_root.loc[x])
generation_input_df['compressor'] = compressor
generation_input_df['accessor'] = accessor

# %%
# example data for experiment
args_dict = dict(generation_input_df.iloc[0])
sample_dir = args_dict['sample_dir']
sample_root = args_dict['sample_root']
compressor = args_dict['compressor']
accessor = args_dict['accessor']

# %%
# conversion function
def convert_experiment(
    sample_dir,
    # zarr config
    sample_root, compressor, accessor,
    # catch all
    **kwargs):
    # load metadata
    meta_path = sample_dir.joinpath('meta.pk')
    sim_result_path = sample_dir.joinpath('simulation_result.pk')
    meta = pickle.load(meta_path.open('rb'))
    sim_result = pickle.load(sim_result_path.open('rb'))

    # check and read metadata
    cloth_verts = sim_result['cloth_state']['verts']
    cloth_faces = np.array(sim_result['cloth_state']['faces'])
    # CLOTH3D data has all quad faces
    assert(cloth_faces.shape[1] == 4)
    cloth_uv_verts = sim_result['cloth_state']['uv_verts']
    cloth_uv_faces = np.array(sim_result['cloth_state']['uv_faces'])
    assert(cloth_uv_faces.shape[1] == 4)
    grip_vertex_idx = sim_result['grip_vertex_idx']
    assert(0 <= grip_vertex_idx < len(cloth_verts))

    # load canonical data
    canonical_data = accessor.get_sample_data(
        sample_id=meta['meta']['sample_id'],
        garment_name=meta['meta']['garment_name'])

    # compute per-cloth-vertex nearest human vertex
    human_verts = canonical_data['human_verts']
    human_faces = canonical_data['human_faces']
    cloth_canonical_verts = canonical_data['garment_verts']
    cloth_texture = canonical_data['garment_texture']

    # load images
    uviz_fnames = meta['images']['uviz']
    rgb_fnames = meta['images']['rgb']
    assert(len(uviz_fnames) == len(rgb_fnames))
    rows = list()
    for uviz_fname, rgb_fname in zip(uviz_fnames, rgb_fnames):
        uviz_path = sample_dir.joinpath(uviz_fname)
        rgb_path = sample_dir.joinpath(rgb_fname)
        uviz_dict = read_uviz(str(uviz_path.absolute()), index_dtype=np.uint8)
        rgb = skimage.io.imread(str(rgb_path.absolute()))

        row = uviz_dict
        row['rgb'] = rgb
        rows.append(row)
    images_df = pd.DataFrame(rows)

    # reformat images
    intrinsic = meta['camera']['intrinsic']
    extrinsic_arr = np.array(list(meta['camera']['extrinsic_list']))
    rgb_arr = np.array(list(images_df.rgb))
    uv_arr = np.array(list(images_df.uv))
    index_arr = np.array(list(images_df.object_index))
    # should specify index in meta
    mask_arr = (index_arr == 1).squeeze()
    # convert Cycles ray length to CV depth
    # depth_arr = np.array(list(images_df.depth.apply(
    #     lambda x: ray_length_to_zbuffer(x, intrinsic))))
    # in Blender 2.90, Cycle's definition of depth changed to CV depth
    depth_arr = np.array(list(images_df.depth))

    # generate point cloud in global frame
    point_cloud_arr = np.empty(rgb_arr.shape, dtype=np.float16)
    assert(len(depth_arr) == len(extrinsic_arr))
    for i in range(len(depth_arr)):
        depth = depth_arr[i]
        extrinsic = extrinsic_arr[i]
        pc_local = zbuffer_to_pcloud(depth, intrinsic)
        tx_world_camera = np.linalg.inv(extrinsic)
        pc_global = pc_local @ tx_world_camera[:3,:3].T + tx_world_camera[:3, 3]
        point_cloud_arr[i] = pc_global

    # extract cloth point cloud
    pc_points = point_cloud_arr[mask_arr]
    pc_uv = uv_arr[mask_arr]
    pc_rgb = rgb_arr[mask_arr]
    pc_sizes = np.sum(mask_arr, (1,2))
    assert(np.sum(pc_sizes) == len(pc_points))

    # compute canonical coordinate for point cloud
    cloth_uv_faces_tri = quads2tris(cloth_uv_faces)
    cloth_faces_tri = quads2tris(cloth_faces)

    query_uv = pc_uv
    target_uv_verts = cloth_uv_verts
    target_uv_faces = cloth_uv_faces_tri

    barycentric, proj_face_idx = query_uv_barycentric(pc_uv, cloth_uv_verts, cloth_uv_faces_tri)
    pc_canonical = barycentric_interpolation(
        barycentric, cloth_canonical_verts, cloth_faces_tri[proj_face_idx])

    # compute human/cloth aabb
    human_aabb = get_aabb(human_verts)
    cloth_aabb = get_aabb(cloth_canonical_verts)
    aabb = get_union_aabb(human_aabb, cloth_aabb)

    # write to zarr
    experiment_group = sample_root.require_group(sample_dir.stem, overwrite=False)
    image_group = experiment_group.require_group('image', overwrite=True)
    point_cloud_group = experiment_group.require_group('point_cloud', overwrite=True)
    misc_group = experiment_group.require_group('misc', overwrite=True)

    # write misc arrays
    misc_data = {
        'cloth_verts': cloth_verts.astype(np.float32),
        'cloth_faces': cloth_faces.astype(np.uint16),
        'cloth_uv_verts': cloth_uv_verts.astype(np.float32),
        'cloth_uv_faces': cloth_uv_faces.astype(np.uint16),
        'cloth_canonical_verts': cloth_canonical_verts.astype(np.float32),
        'human_verts': human_verts.astype(np.float32),
        'human_faces': human_faces.astype(np.uint16),
        'cloth_aabb': cloth_aabb.astype(np.float32),
        'human_aabb': human_aabb.astype(np.float32),
        'intrinsic': intrinsic.astype(np.float32),
        'extrinsic_list': extrinsic_arr.astype(np.float32),
        'cloth_texture': cloth_texture.astype(np.uint8)
    }
    for key, data in misc_data.items():
        misc_group.array(
            name=key, data=data, chunks=data.shape,
            compressor=compressor, overwrite=True)

    # write image arrays
    image_data = {
        'rgb': rgb_arr.astype(np.uint8),
        'uv': uv_arr.astype(np.float16),
        'depth': depth_arr.astype(np.float16),
        'mask': np.expand_dims(mask_arr, axis=-1)
    }
    for key, data in image_data.items():
        image_group.array(
            name=key, data=data, chunks=(1,) + data.shape[1:],
            compressor=compressor, overwrite=True)

    # write point cloud arrays
    pc_data = {
        'point': pc_points.astype(np.float16),
        'uv': pc_uv.astype(np.float16),
        'rgb': pc_rgb.astype(np.uint8),
        'canonical_point': pc_canonical.astype(np.float16),
        'sizes': pc_sizes.astype(np.int64)
    }
    for key, data in pc_data.items():
        point_cloud_group.array(
            name=key, data=data, chunks=data.shape,
            compressor=compressor, overwrite=True)

    # set attrs
    meta_attr = meta['meta']
    attrs = {
        'sample_id': meta_attr['sample_id'],
        'garment_name': meta_attr['garment_name'],
        'gender': meta_attr['gender'],
        'fabric': meta_attr['fabric'],
        'grip_vertex_idx': grip_vertex_idx
    }
    experiment_group.attrs.put(attrs)

# %%
# test threads
# for i in range(10):
#     convert_experiment(**args_dict)

# %%
# do it
# num_workers = 40
# Dylan: Changing to 1 so as to not explode my computer.
num_workers = 1
input_df = generation_input_df.copy()
# input_df = redo_input_df.copy()
input_sequence = list()
for _, row in tqdm(input_df.iterrows()):
    input_sequence.append(dict(row))

import numcodecs
numcodecs.blosc.set_nthreads(1)
result_df = parallel_map(
    lambda kwargs: convert_experiment(**kwargs),
    sequence=input_sequence,
    num_workers=num_workers)

result_output_df = result_df[['result', 'error', 'stack_trace']]
pickle.dump(result_output_df, open('/home/cchi/dev/blender_experiment/data/logs/result_output_df.pk', 'wb'))

# %%
# handle failure
result_output_df = pickle.load(open('/home/cchi/dev/blender_experiment/data/logs/result_output_df.pk', 'rb'))
error_df = result_output_df.loc[result_output_df.error.notnull()]

rows = dict()
for idx, row in error_df.iterrows():
    error_file_path = pathlib.Path(str(row.error).split("'")[1])
    sample_key = error_file_path.parent.stem
    packet_id = int(error_file_path.parent.parent.stem)
    sample_id, category, grip_vertex_idx = sample_key.split('_')
    rows[idx] = {
        'error_file_path': error_file_path,
        'sample_key': sample_key,
        'packet_id': packet_id,
        'sample_id': sample_id,
        'category': category,
        'grip_vertex_idx': grip_vertex_idx
    }
redo_info_df = pd.DataFrame(list(rows.values()), index=rows.keys())

import shutil
for _, row in redo_info_df.iterrows():
    sample_dir = row.error_file_path.parent
    old_meta_path = str(sample_dir.joinpath('meta.pk').absolute())
    new_meta_path = str(sample_dir.joinpath('meta.pk.old').absolute())
    shutil.move(old_meta_path, new_meta_path)


# setup new input df
rows = dict()
for idx, row in redo_info_df.iterrows():
    sample_dir = pathlib.Path(render_output_root).joinpath(
        str(row.packet_id), '_'.join(str(row[x]) for x in
            ['sample_id', 'category', 'grip_vertex_idx']))
    rows[idx] = {
        'sample_dir': sample_dir,
        'category': row.category,
        'sample_id': row.sample_id,
        'sample_root': datasets_df.sample_root.loc[row.category],
        'compressor': compressor,
        'accessor': accessor
    }
redo_input_df = pd.DataFrame(list(rows.values()), index=rows.keys())


# %%
import functools

def compute_summary(dataset_dir):
    compressor = Blosc(cname='zstd', clevel=6, shuffle=Blosc.BITSHUFFLE)
    store = zarr.DirectoryStore(str(dataset_dir.absolute()))
    root = zarr.group(store=store, overwrite=False)
    sample_root = root.require_group('samples', overwrite=False)
    summary_root = root.require_group('summary', overwrite=False)

    cloth_aabb_union = functools.reduce(
        get_union_aabb,
        (group['misc/cloth_aabb'][:]
            for key, group in sample_root.groups()))

    human_aabb_union = functools.reduce(
        get_union_aabb,
        (group['misc/human_aabb'][:]
            for key, group in sample_root.groups()))

    summary_data = {
        'cloth_aabb_union': cloth_aabb_union,
        'human_aabb_union': human_aabb_union
    }
    for key, data in summary_data.items():
        summary_root.array(
            name=key, data=data, chunks=data.shape,
            compressor=compressor, overwrite=True)

dataset_root_dir = "/local/crv/cchi/data/cloth_3d_workspace/cloth_3d_grip_dataset"
dataset_dirs = list(pathlib.Path(dataset_root_dir).glob("*"))
result_df = parallel_map(compute_summary, dataset_dirs, num_workers=10)


# %%
