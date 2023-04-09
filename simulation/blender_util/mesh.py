import bpy
import bmesh
import numpy as np
import networkx as nx
import collections

from blender_util.collection import get_mesh_collection
from cloth_3d_util.util import quads2tris
from common.geometry_util import faces_to_edges

# bmesh utils
# ===========
class BMeshManager:
    """
    Generic BMesh Manager.
    Usually not direclty used.
    """
    def __init__(self):
        self.bm = bmesh.new()
    
    def __enter__(self):
        return self.bm
    
    def __exit__(self, type, value, traceback):
        self.bm.free()


class NumpyBMeshManager(BMeshManager):
    def __init__(self, verts, edges=None, faces=None, uv=None, uv_faces=None):
        super().__init__()

        if faces is not None and uv_faces is not None:
            # assume faces and uv_faces have 1:1 coorespondence
            assert(len(faces) == len(uv_faces))
        
        if uv is not None and uv_faces is None:
            # assume verts and uv coords have 1:1 coorespondence
            assert(len(uv) == len(verts))
            # uv_faces always exist if uv exists
            uv_faces = faces

        bm = self.bm
        # add vertices
        for v in verts:
            bm.verts.new(tuple(v))
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        # add edges
        if edges is not None:
            for e in edges:
                assert(len(e) == 2)
                e_verts = [bm.verts[i] for i in e]
                bm.edges.new(e_verts)
            bm.edges.ensure_lookup_table()
            bm.edges.index_update()
        # add faces
        if faces is not None:
            for f in faces:
                assert(len(f) >= 3)
                f_verts = [bm.verts[i] for i in f]
                bm.faces.new(f_verts)
            bm.faces.ensure_lookup_table()
            bm.faces.index_update()
        # add uv
        if uv is not None:
            uv_layer = bm.loops.layers.uv.new()
            for f in bm.faces:
                face = faces[f.index]
                uv_face = uv_faces[f.index]
                local_v_idx_to_uv = dict(
                    (v_idx, tuple(uv[uv_idx]))
                    for v_idx, uv_idx in zip(face, uv_face))
                for l in f.loops:
                    # each loop is an edge (two verts)
                    v_idx = l.vert.index
                    l[uv_layer].uv = local_v_idx_to_uv[v_idx]


class DepsGraphBMeshManager(BMeshManager):
    def __init__(self, mesh_obj, deps_graph):
        super().__init__()
        bm = self.bm
        bm.from_object(mesh_obj, deps_graph)
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        bm.edges.ensure_lookup_table()
        bm.edges.index_update()
        bm.faces.ensure_lookup_table()
        bm.faces.index_update()


def get_bmesh_verts(bm):
    verts = np.empty((len(bm.verts), 3), dtype=np.float32)
    for i in range(len(bm.verts)):
        verts[i, :] = bm.verts[i].co
    return verts

def get_bmesh_edges(bm):
    edges = np.empty((len(bm.edges), 2), dtype=np.uint32)
    for i in range(len(bm.edges)):
        verts = bm.edges[i].verts
        idxs = [v.index for v in verts]
        edges[i] = idxs
    return edges

def get_bmesh_faces(bm):
    faces = list()
    for f in bm.faces:
        idxs = [v.index for v in f.verts]
        faces.append(idxs)
    return faces

def bmesh_to_numpy(
        bm, 
        record_verts=True, record_edges=False, record_faces=True, 
        record_uv_verts=True, record_uv_faces=True, uv_layer_id=0):
    data = dict()
    if record_verts:
        data['verts'] = get_bmesh_verts(bm)
    if record_edges:
        data['edges'] = get_bmesh_edges(bm)
    if record_faces or record_uv_verts or record_uv_faces:
        uv_layer = bm.loops.layers.uv[uv_layer_id]
        uv_verts = dict()
        def get_uv_vert_idx(uv_vert):
            uv_vert = tuple(uv_vert)
            if uv_vert in uv_verts:
                return uv_verts[uv_vert]
            else:
                new_idx = len(uv_verts)
                uv_verts[uv_vert] = new_idx
                return new_idx
        faces = list()
        uv_faces = list()
        for f in bm.faces:
            face = list()
            uv_face = list()
            for l in f.loops:
                v_idx = l.vert.index
                uv_idx = get_uv_vert_idx(l[uv_layer].uv)
                face.append(v_idx)
                uv_face.append(uv_idx)
            faces.append(face)
            uv_faces.append(uv_face)
        uv_verts_list = list(uv_verts.items())
        uv_verts_list = sorted(uv_verts_list, key=lambda x: x[1])
        uv_verts = np.array([x[0] for x in uv_verts_list])
        if record_faces:
            data['faces'] = faces
        if record_uv_verts:
            data['uv_verts'] = uv_verts
        if record_uv_faces:
            data['uv_faces'] = uv_faces
    return data

# mesh animation utils
# ====================
def mesh_animation_to_numpy(
    mesh_obj, start_frame, end_frame, 
    global_coordinate=True, 
    static_topology=True, static_uv=True):
    """
    Assumes that only vertecies can change according to the animation
    """
    assert(start_frame <= end_frame)
    scene = bpy.context.scene
    original_frame = scene.frame_current

    # record fixed data
    static_data = None
    scene.frame_set(start_frame)
    deps_graph = bpy.context.evaluated_depsgraph_get()
    with DepsGraphBMeshManager(mesh_obj, deps_graph) as bm:
        static_data = bmesh_to_numpy(
            bm, 
            record_verts=False, record_edges=True, record_faces=True, 
            record_uv_verts=True, record_uv_faces=True)
    
    # record animation data
    options = {
        'record_verts': True,
        'record_edges': False,
        'record_faces': False,
        'record_uv_verts': False,
        'record_uv_faces': False
    }
    if not static_uv:
        options['record_uv_verts']=True
        if not static_topology:
            options['record_edges'] = True
            options['record_faces'] = True
            options['record_uv_faces'] = True

    rows = list()
    for frame in range(start_frame, end_frame+1):
        scene.frame_set(frame)
        deps_graph = bpy.context.evaluated_depsgraph_get()
        with DepsGraphBMeshManager(mesh_obj, deps_graph) as bm:
            row = bmesh_to_numpy(bm, **options)
            if global_coordinate:
                row['matrix_world'] = mesh_obj.matrix_world
            rows.append(row)
    scene.frame_set(original_frame)

    # aggregate data
    agg_data = collections.defaultdict(list)
    for row in rows:
        for key, value in row.items():
            agg_data[key].append(value)
    if 'verts' in agg_data:
        result_verts = agg_data['verts']
        if 'matrix_world' in agg_data:
            result_verts = list()
            for verts, matrix_world in zip(agg_data['verts'], agg_data['matrix_world']):
                tx_global_local = np.array(matrix_world)
                verts_global = verts @ tx_global_local[:3, :3].T + tx_global_local[:3, 3]
                result_verts.append(verts_global)
            del agg_data['matrix_world']
        agg_data['verts'] = np.stack(result_verts, axis=0)
    if 'edges' in agg_data:
        agg_data['edges'] = np.stack(agg_data['edges'], axis=0)
    if 'uv_verts' in agg_data:
        agg_data['uv_verts'] = np.stack(agg_data['uv_verts'], axis=0)

    # merge static and dynamic
    final_data = static_data
    final_data.update(agg_data)
    return final_data


def mesh_to_numpy(mesh_obj, global_coordinate=True):
    options = {
        'record_verts': True,
        'record_edges': True,
        'record_faces': True,
        'record_uv_verts': True,
        'record_uv_faces': True
    }
    data = None
    deps_graph = bpy.context.evaluated_depsgraph_get()
    with DepsGraphBMeshManager(mesh_obj, deps_graph) as bm:
        data = bmesh_to_numpy(
            bm, **options)
    if global_coordinate:
        verts = data['verts']
        tx_global_local = np.array(mesh_obj.matrix_world)
        verts_global = verts @ tx_global_local[:3, :3].T + tx_global_local[:3, 3]
        data['verts'] = verts_global
    return data

# mesh object functions
# =====================

def require_mesh(name):
    if name not in bpy.data.meshes:
        bpy.data.meshes.new(name)
    mesh = bpy.data.meshes[name]

    if name not in bpy.data.objects:
        bpy.data.objects.new(name, mesh)
    mesh_obj = bpy.data.objects[name]

    mesh_collection = get_mesh_collection()
    if name not in mesh_collection.objects:
        mesh_collection.objects.link(mesh_obj)
    return mesh_obj

def create_cube(name, size, location):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    cube_obj = bpy.context.active_object
    cube_obj.name = name
    bpy.context.scene.collection.objects.unlink(cube_obj)
    mesh_collection = get_mesh_collection()
    mesh_collection.objects.link(cube_obj)
    return cube_obj

def get_material(mesh_obj):
    obj_mat = mesh_obj.data.materials
    if len(obj_mat) > 0:
        return obj_mat[0]
    else:
        return None

def set_material(mesh_obj, material):
    obj_mat = mesh_obj.data.materials
    if len(obj_mat) > 0:
        obj_mat[0] = material
    else:
        obj_mat.append(material)


# vertex group functions
# ======================
def require_vertex_group(mesh_obj, vertex_group_name):
    if vertex_group_name not in mesh_obj.vertex_groups:
        mesh_obj.vertex_groups.new(name=vertex_group_name)
    vertex_group = mesh_obj.vertex_groups[vertex_group_name]
    return vertex_group

def clear_vertex_group(mesh_obj, vertex_group_name):
    vertex_group = mesh_obj.vertex_groups[vertex_group_name]
    vertex_group.remove(list(range(len(mesh_obj.data.vertices))))

def get_soft_weight(verts, faces, vert_idxs, radius=0.1):
    """
    Assign weight to selected vertecies as well as their neighbors.
    """
    faces = np.array(faces)
    faces_tri = faces
    if faces.shape[1] == 4:
        faces_tri = quads2tris(faces)
    assert(faces_tri.shape[1] == 3)

    edges = faces_to_edges(faces)
    edge_length = np.linalg.norm(verts[edges[:,0]] - verts[edges[:,1]], axis=1)
    graph = nx.Graph()
    for edge, length in zip(edges, edge_length):
        graph.add_edge(edge[0], edge[1], length=length)
    
    weight = np.zeros(len(verts))
    for vert_idx in vert_idxs:
        neighbor_graph = nx.ego_graph(
            graph, vert_idx, radius=radius*2, distance='length')
        neighbor_nodes = np.array(neighbor_graph.nodes(), dtype=np.int32)

        this_vert = verts[vert_idx]
        neighbor_verts = verts[neighbor_nodes]
        neighbor_dists = np.linalg.norm(neighbor_verts - this_vert, axis=1)
        neighbor_weight = np.ones_like(neighbor_dists)
        neighbor_weight[neighbor_dists > radius] = 0
        weight[neighbor_nodes] += neighbor_weight
    final_weight = np.clip(weight, 0, 1)
    return final_weight

def set_vertex_group_weight(mesh_obj, vertex_group_name, weights):
    vertex_group = require_vertex_group(mesh_obj, vertex_group_name)
    clear_vertex_group(mesh_obj, vertex_group_name)
    for i, weight in enumerate(weights):
        if weight > 0:
            vertex_group.add([i], weight, 'ADD')
    return vertex_group

