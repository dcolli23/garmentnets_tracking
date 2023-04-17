import pyexr
import pathlib
import numpy as np

def read_uviz(fname, index_dtype=np.uint16):
    """
    UVIZ is a openexr file convension. 
    (R, G) channels represent uv coordinates (0, 1)
    B channel represent object index (0, inf)
    A channel represent depth (0, inf)
    All channels in 16 bit float (HALF)
    """
    exr_file = pyexr.open(fname)
    channels = dict()
    for key, dtype in exr_file.channel_precision.items():
        data = exr_file.get(key, precision=dtype)
        channels[key] = data
    uv = np.dstack((channels['R'], channels['G']))
    object_index = np.rint(channels['B']).astype(np.uint16)
    z = channels['A']

    data = {
        'uv': uv,
        'object_index': object_index,
        'depth': z,
    }
    return data

