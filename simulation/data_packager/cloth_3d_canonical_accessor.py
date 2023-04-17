import os
import functools

import numpy as np
import skimage.io

from cloth_3d_util.read import DataReader
from cloth_3d_util.util import quads2tris, axis_angle_to_matrix
from cloth_3d_util.IO import readOBJ

class Cloth3DCanonicalAccessor:
    def __init__(self, dataset_path="/home/dcolli23/DataLocker/datasets/CLOTH3D/"):
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

    @functools.lru_cache(maxsize=32)
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