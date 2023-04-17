import numpy as np
import numexpr as ne


def ray_length_to_zbuffer(ray_length, intrinsic, dtype=None):
    if dtype is None:
        dtype = ray_length.dtype

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    assert(np.allclose(fx, fy))
    f_pixel = fx

    # remove redendent dimention
    ray_length_flat = ray_length.squeeze().astype(dtype)
    img_x, img_y = np.meshgrid(
        np.arange(ray_length_flat.shape[1], dtype=dtype),
        np.arange(ray_length_flat.shape[0], dtype=dtype))
    x_pixel = img_x - cx
    y_pixel = img_y - cy

    hypot_pixel = ne.evaluate("sqrt(x_pixel*x_pixel + y_pixel*y_pixel + f_pixel*f_pixel)")
    z_flat = ray_length_flat / hypot_pixel * f_pixel
    zbuffer = z_flat.reshape(ray_length.shape)
    return zbuffer


def ray_length_to_pcloud(ray_length, intrinsic, dtype=None, fill_invalid=np.nan):
    if dtype is None:
        dtype = np.float32
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    assert(np.allclose(fx, fy))
    f_pixel = fx
    
    # remove redendent dimention
    ray_length_flat = ray_length.squeeze().astype(dtype)
    img_x, img_y = np.meshgrid(
        np.arange(ray_length_flat.shape[1], dtype=dtype), 
        np.arange(ray_length_flat.shape[0], dtype=dtype))
    x_pixel = img_x - intrinsic[0, 2]
    y_pixel = img_y - intrinsic[1, 2]
    
    pcloud = None
    with np.errstate(invalid='ignore'):
        hypot_pixel = ne.evaluate("sqrt(x_pixel*x_pixel + y_pixel*y_pixel + f_pixel * f_pixel)")
        z = ray_length_flat / hypot_pixel * f_pixel
        x = x_pixel * z / f_pixel
        y = y_pixel * z / f_pixel
        pcloud = np.stack((x, y, z), axis=-1)

    if fill_invalid is not None:
        pcloud[~np.isfinite(pcloud)] = fill_invalid
    return pcloud


def zbuffer_to_pcloud(zbuffer, intrinsic, dtype=None, fill_invalid=np.nan):
    if dtype is None:
        dtype = np.float32

    # remove redendent dimention
    zbuffer_flat = zbuffer.squeeze().astype(dtype)
    img_x, img_y = np.meshgrid(
        np.arange(zbuffer_flat.shape[1], dtype=dtype), 
        np.arange(zbuffer_flat.shape[0], dtype=dtype))
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    
    pcloud = None
    with np.errstate(invalid='ignore'):
        x = (img_x - intrinsic[0, 2]) * zbuffer_flat / fx
        y = (img_y - intrinsic[1, 2]) * zbuffer_flat / fy
        pcloud = np.stack((x, y, zbuffer_flat), axis=-1)
    
    if fill_invalid is not None:
        pcloud[~np.isfinite(pcloud)] = fill_invalid
    return pcloud


def camera_to_world(pcloud, extrinsic, dtype=None):
    if dtype is None:
        dtype = pcloud.dtype
    extrinsic = extrinsic.astype(dtype)
    out_cloud = (pcloud - extrinsic[:3, -1]) @ extrinsic[:3, :-1]
    return out_cloud


def world_to_camera(pcloud, extrinsic):
    if dtype is None:
        dtype = pcloud.dtype
    extrinsic = extrinsic.astype(dtype)
    out_cloud = pcloud @ extrinsic[:3, :-1].T + extrinsic[:3, -1]
    return out_cloud