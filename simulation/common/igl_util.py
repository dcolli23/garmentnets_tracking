from typing import Optional, Tuple

import numpy as np
from scipy.spatial.ckdtree import cKDTree
import igl
# import pyembree
import trimesh


def query_uv_barycentric_igl(query_uv, target_uv_verts, target_uv_faces):
    """
    Input:
    query_uv: (V, 2) float array for query uv points
    target_uv_verts: (V, 2) float array for target uv vertex
    target_uv_faces: (F, 3) int array for uv face vertecies

    Output:
    barycentric: (V, 3) float array for projected barycentric coordinate
    proj_face_idx: (V,) int array for cooresponding face index
    """
    assert(target_uv_faces.shape[1] == 3)
    query_uv_3d = np.zeros((len(query_uv), 3), dtype=np.float32)
    query_uv_3d[:,:2] = query_uv

    target_uv_3d = np.zeros((len(target_uv_verts), 3), dtype=np.float32)
    target_uv_3d[:,:2] = target_uv_verts

    tg_uv_faces = target_uv_faces.astype(np.int32)

    # hack to find face projectionin 2d
    # slow (1 sec for a regular point cloud)
    # can be accelerated with a BVH + triangle intersection test (byembree?)
    signed_distance, proj_face_idx, proj_verts = igl.signed_distance(
        query_uv_3d, target_uv_3d, tg_uv_faces, return_normals=False)

    proj_faces = tg_uv_faces[proj_face_idx]
    corners = list()
    for i in range(3):
        corners.append(target_uv_3d[proj_faces[:, i]])

    barycentric = igl.barycentric_coordinates_tri(proj_verts, *corners)
    return barycentric, proj_face_idx


def query_uv_barycentric(query_uv, target_uv_verts, target_uv_faces):
    """
    Input:
    query_uv: (V, 2) float array for query uv points
    target_uv_verts: (V, 2) float array for target uv vertex
    target_uv_faces: (F, 3) int array for uv face vertecies

    Output:
    barycentric: (V, 3) float array for projected barycentric coordinate
    proj_face_idx: (V,) int array for cooresponding face index
    """
    assert(target_uv_faces.shape[1] == 3)
    uv_verts_3d = np.zeros((len(target_uv_verts), 3), dtype=target_uv_verts.dtype)
    uv_verts_3d[:,:2] = target_uv_verts

    ray_origins = np.zeros((len(query_uv), 3), dtype=query_uv.dtype)
    ray_origins[:,:2] = query_uv
    ray_origins[:,2] = 1
    ray_directions = np.zeros((len(query_uv), 3), dtype=query_uv.dtype)
    ray_directions[:,2] = -1

    mesh = trimesh.Trimesh(vertices=uv_verts_3d, faces=target_uv_faces, use_embree=True)
    intersector = mesh.ray
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins, ray_directions, multiple_hits=False)

    # convert to query_uv index
    proj_face_idx = np.full(len(query_uv), fill_value=-1, dtype=np.int64)
    proj_face_idx[index_ray] = index_tri

    # get barycentric coordinates
    corners = list()
    for i in range(3):
        corners.append(uv_verts_3d[target_uv_faces[index_tri, i]].astype(np.float64))
    barycentric = np.full((len(query_uv), 3), fill_value=-1, dtype=target_uv_verts.dtype)
    barycentric[index_ray] = igl.barycentric_coordinates_tri(locations.astype(np.float64), *corners)

    # in case of miss
    if len(index_ray) < len(query_uv):
        miss_idxs = np.nonzero(proj_face_idx == -1)[0]
        kdtree = cKDTree(target_uv_verts)
        _, nn_vert_idxs = kdtree.query(query_uv[miss_idxs], k=1)

        vert_face_idx_map = np.zeros(len(target_uv_verts), dtype=np.int64)
        face_idxs = np.tile(np.arange(len(target_uv_faces)), (3,1)).T
        vert_face_idx_map[target_uv_faces] = face_idxs

        miss_face_idxs = vert_face_idx_map[nn_vert_idxs]
        proj_face_idx[miss_idxs] = miss_face_idxs
        for i in range(len(miss_idxs)):
            miss_face = target_uv_faces[miss_face_idxs[i]]
            is_vertex = (miss_face == nn_vert_idxs[i])
            assert(is_vertex.sum() == 1)
            this_bc = is_vertex.astype(barycentric.dtype)
            barycentric[miss_idxs[i]] = this_bc

    return barycentric, proj_face_idx


def test_query_uv_barycentric():
    query_uv = np.array([[0.25,0.25],[-1,-1],[2,0]], dtype=np.float32)
    target_uv_verts = np.array([[0,0],[1,0],[0,1]], dtype=np.float32)
    target_uv_faces = np.array([[0,1,2]], dtype=np.int64)

    barycentric, proj_face_idx = query_uv_barycentric(query_uv, target_uv_verts, target_uv_faces)
    assert(np.allclose(np.sum(barycentric, axis=1), 1))
    assert(np.all(proj_face_idx >= 0))
    assert(np.all(proj_face_idx == 0))

    p_idx = 0
    p0_proj = target_uv_verts[target_uv_faces[proj_face_idx[p_idx]]].T @ barycentric[p_idx].T
    assert(np.allclose(p0_proj, query_uv[p_idx]))

    assert(np.allclose(barycentric[1], [1,0,0]))
    assert(np.allclose(barycentric[2], [0,1,0]))


def mesh_sample_barycentric(
        verts: np.ndarray, faces: np.ndarray,
        num_samples: int, seed: Optional[int] = None,
        face_areas: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    # generate face area
    if face_areas is None:
        face_areas = igl.doublearea(verts, faces)
    face_areas = face_areas / np.sum(face_areas)
    assert(len(face_areas) == len(faces))

    rs = np.random.RandomState(seed=seed)
    # select faces
    selected_face_idx = rs.choice(
        len(faces), size=num_samples,
        replace=True, p=face_areas).astype(faces.dtype)

    # generate random barycentric coordinate
    barycentric_uv = rs.uniform(0, 1, size=(num_samples, 2))
    not_triangle = (np.sum(barycentric_uv, axis=1) >= 1)
    barycentric_uv[not_triangle] = 1 - barycentric_uv[not_triangle]

    barycentric_all = np.zeros((num_samples, 3), dtype=barycentric_uv.dtype)
    barycentric_all[:, :2] = barycentric_uv
    barycentric_all[:, 2] = 1 - np.sum(barycentric_uv, axis=1)

    return barycentric_all, selected_face_idx