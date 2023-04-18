import os
import sys
from pathlib import Path
import pickle

import bpy
import zarr
import numpy as np
import skimage.io
import pandas as pd
from numcodecs import Blosc

# Do some ugly path manipulation to find all packages
sys.path.append("../")
sys.path.append("../../")
from simulation.io_util.image_io_util import read_uviz
from simulation.common.projection import ray_length_to_zbuffer, zbuffer_to_pcloud
from simulation.common.geometry_util import (barycentric_interpolation, get_aabb, get_union_aabb)
from simulation.cloth_3d_util.util import quads2tris, axis_angle_to_matrix
from simulation.common.igl_util import query_uv_barycentric
from simulation.data_packager.cloth_3d_canonical_accessor import Cloth3DCanonicalAccessor
from simulation.blender_util_dylan.gripper import GripperData

# Trying to avoid this import as it will execute the Zarr packaging code as is.
# from simulation.data_packager.smpl_cloth_zarr_v5_cheng import Cloth3DCanonicalAccessor

FILE_ROOT = Path(os.getcwd())
print("File root:", FILE_ROOT)
GARMENTNETS_ROOT = (FILE_ROOT / ".." / "..").absolute().resolve()
print("GarmentNets root:", GARMENTNETS_ROOT)
SIM_DATASET_DIR = (FILE_ROOT / ".." / "script_output" / "full_dataset_attempt_2").absolute()
CLOTH3D_PATH = Path(os.path.expanduser("~/DataLocker/datasets/CLOTH3D/training/"))

ZARR_FILEPATH = GARMENTNETS_ROOT / "data" / "garmentnets_tracking_simulation_dataset.zarr"

DYNAMICS_SEQUENCE_FILENAME_TEMPLATE = "dynamics_seq_{idx}"
CAMERA_IDX_USED_FOR_ANIMATION = 0

def main():
    compressor = Blosc(cname='zstd', clevel=6, shuffle=Blosc.BITSHUFFLE)
    categories = ["Tshirt"]
    rows = dict()
    for category in categories:
        store = zarr.DirectoryStore(ZARR_FILEPATH / category)
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
    # datasets_df = pd.DataFrame(list(rows.values()), index=rows.keys())

    accessor = Cloth3DCanonicalAccessor(CLOTH3D_PATH)

    for sample_dir in SIM_DATASET_DIR.iterdir():
        sample_root_zarr = rows["Tshirt"]["sample_root"]
        compressor_zarr = rows["Tshirt"]["compressor"]
        print(f"Zarrifying '{sample_dir.name}'")
        zarrify_rest_state(sample_dir, sample_root_zarr, compressor_zarr, accessor)

        print(f"Appending dynamics to '{sample_dir.name}' zarr file.")
        append_dynamics_to_zarr(sample_dir, sample_root_zarr, compressor_zarr, accessor)

def zarrify_rest_state(sample_dir,
                       # Zarr configuration
                       sample_root, compressor, accessor):
    # load metadata
    meta_path = sample_dir.joinpath('meta.pk')
    sim_result_path = sample_dir.joinpath('hanging_rest_state_results.pkl')
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
    # TODO: This will have to be heavily modified to work with the dynamics simulations.
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

def append_dynamics_to_zarr(sample_dir: Path, sample_root: zarr.hierarchy.Group, compressor,
                            accessor):
    dynamics_group_zarr = sample_root.require_group('dynamics', overwrite=False)
    i = 0
    dynamics_seq_dir = sample_dir / DYNAMICS_SEQUENCE_FILENAME_TEMPLATE.format(idx=i)
    while dynamics_seq_dir.exists():
        print(f"Appending '{dynamics_seq_dir.name}' to '{sample_dir.name}' zarr file.")
        # Create the group for this sequence in the zarr file.
        seq_group = sample_root.require_group(str(i), overwrite=False)

        append_single_dynamics_sequence_to_zarr(dynamics_seq_dir, seq_group, compressor, accessor)

        i += 1
        dynamics_seq_dir = sample_dir / DYNAMICS_SEQUENCE_FILENAME_TEMPLATE.format(idx=i)

def append_single_dynamics_sequence_to_zarr(dynamics_seq_dir: Path,
                                            dynamics_seq_zarr: zarr.hierarchy.Group,
                                            compressor, accessor):
    # Load the blend file associated with this dynamics simulation.
    blend_file_path = dynamics_seq_dir / "dynamics_animation.blend"
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)

    # Get the start and end frame of the animation so we know which images to load. This shouldn't
    # be necessary but I accidentally hardcoded the number of images to render (200) even though
    # most simulations are far less.
    animation_frame_start = bpy.context.scene.animation_data.action.frame_start
    animation_frame_end = bpy.context.scene.animation_data.action.frame_end

    # Load the metadata
    meta_path = dynamics_seq_dir.joinpath('meta.pk')
    meta = pickle.load(meta_path.open('rb'))

    # Tentative: load the canonical data? Or maybe pass it in if we need it to get canonical
    # coordinates corresponding to the points in the point cloud of the dynamics sequence.

    # Load in the images.
    uviz_fnames = dynamics_seq_dir.glob("*.exr")
    rgb_fnames = dynamics_seq_dir.glob("*.png")
    assert (len(uviz_fnames) == len(rgb_fnames))
    rows = list()
    for uviz_fname, rgb_fname in zip(uviz_fnames, rgb_fnames):
        uviz_path = dynamics_seq_dir / uviz_fname
        rgb_path = dynamics_seq_dir / rgb_fname
        uviz_dict = read_uviz(str(uviz_path.absolute()), index_dtype=np.uint8)
        rgb = skimage.io.imread(str(rgb_path.absolute()))

        row = uviz_dict
        row['rgb'] = rgb
        rows.append(row)
    images_df = pd.DataFrame(rows)

    # Reformat the images.
    intrinsic = meta['camera']['intrinsic']
    extrinsic_arr = np.array(list(meta['camera']['extrinsic_list']))[CAMERA_IDX_USED_FOR_ANIMATION]
    rgb_arr = np.stack(list(images_df.rgb), axis=0)
    uv_arr = np.stack(list(images_df.uv), axis=0)
    index_arr = np.stack(list(images_df.object_index), axis=0)
    # should specify index in meta
    mask_arr = (index_arr == 1).squeeze()
    # convert Cycles ray length to CV depth
    # depth_arr = np.array(list(images_df.depth.apply(
    #     lambda x: ray_length_to_zbuffer(x, intrinsic))))
    # in Blender 2.90, Cycle's definition of depth changed to CV depth
    depth_arr = np.stack(list(images_df.depth), axis=0)

    # Generate point cloud in the global frame.
    # NOTE: Have to translate by the negative of the amount the camera was translated for dynamics
    # sequence recording.
    point_cloud_arr = np.empty(rgb_arr.shape, dtype=np.float16)
    assert(depth_arr.shape[0] == rgb_arr.shape[0])
    for i in range(len(depth_arr)):
        depth = depth_arr[i]
        extrinsic = extrinsic_arr[i]
        pc_local = zbuffer_to_pcloud(depth, intrinsic)
        tx_world_camera = np.linalg.inv(extrinsic)
        pc_global = pc_local @ tx_world_camera[:3,:3].T + tx_world_camera[:3, 3]
        point_cloud_arr[i] = pc_global

    # Compute (or pass in) the canonical coordinates for the point cloud.

    # Tenative: Compute human/cloth aabb.
    # NOTE: Probably not necessary, right?

    # Compute/find the gripper velocity at each timestep of the simulation.
    # NOTE: Should we also include acceleration?

    gripper_data = GripperData(blend_file_path)

    # Write to Zarr

