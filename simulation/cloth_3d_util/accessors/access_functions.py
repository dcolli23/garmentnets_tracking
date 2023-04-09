from typing import Optional

import os
import numpy as np
import skimage.io
from cloth_3d_util.util import loadInfo, quads2tris, axis_angle_to_matrix
from cloth_3d_util.IO import readOBJ

def get_info(dataset_path: str, sample_id: str):
    info_path = os.path.join(dataset_path, sample_id, 'info')
    info = loadInfo(info_path)
    return info

def get_garment_texture(
    dataset_path: str, sample_id: str, 
    garment_name: str, info: Optional[dict]=None) -> np.ndarray:
    if info is None:
        info = get_info(dataset_path, sample_id)
    garment_info = info['outfit'][garment_name]
    texture_type = garment_info['texture']['type']
    texture = None
    if texture_type == 'color':
        color = garment_info['texture']['data']
        color_int = (color * 255).astype(np.uint8)
        texture = np.empty((2048, 2048, 3), dtype=np.uint8)
        texture[:,:,:] = color_int
    else:
        texture_path = os.path.join(dataset_path, sample_id, garment_name + '.png')
        texture = skimage.io.imread(texture_path)
    return texture

def get_garment_metadata(
    dataset_path: str, sample_id: str, 
    garment_name: str, info: Optional[dict]=None) -> np.ndarray:
    if info is None:
        info = get_info(dataset_path, sample_id)
    garment_info = info['outfit'][garment_name]
    data = {
        'gender': info['gender'],
        'garment_fabric': garment_info['fabric']
    }
    return data
