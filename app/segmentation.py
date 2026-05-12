"""Mesh segmentation utilities — region-grow on dihedral-angle walls.

The dental geometry trick: teeth are separated from one another and from the gum
by sharp folds in the surface (interproximal valleys, gingival ridges). If we
build a face-to-face adjacency graph that ONLY connects neighbour faces whose
dihedral angle is gentler than some threshold, then the connected components of
that graph are roughly the individual teeth + gum. The component containing the
user's seed click is the prep tooth.
"""
import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def isolate_tooth(mesh: pv.PolyData, seed_point, dihedral_deg: float = 30.0) -> pv.PolyData | None:
    """Extract the connected sub-mesh that contains `seed_point`.

    Faces are connected only across edges whose dihedral angle deviates from
    flat by less than `dihedral_deg`. Sharp folds (margins, interproximal
    spaces) effectively cut the mesh.

    Returns a new PolyData of the prep region, or None on failure.
    """
    if mesh is None or mesh.n_cells == 0:
        return None

    # 1. Locate the face under the seed click.
    seed_face = int(mesh.find_closest_cell(np.asarray(seed_point, dtype=float)))
    if seed_face < 0:
        return None

    # 2. Triangle face indices. Assumes mesh is triangulated (true for STLs).
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    n_faces = len(faces)

    # 3. Per-face outward normals.
    pts = np.asarray(mesh.points)
    v0, v1, v2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)

    # 4. Build (face_a, face_b) pairs for every shared edge — vectorised.
    e_a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e_b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    face_of_edge = np.tile(np.arange(n_faces, dtype=np.int64), 3)
    edge_key = np.sort(np.column_stack([e_a, e_b]), axis=1)

    order = np.lexsort([edge_key[:, 1], edge_key[:, 0]])
    sorted_keys = edge_key[order]
    sorted_face = face_of_edge[order]

    # An edge is shared by two faces iff its key appears twice consecutively.
    same = np.all(sorted_keys[:-1] == sorted_keys[1:], axis=1)
    shared_pos = np.where(same)[0]
    f0 = sorted_face[shared_pos]
    f1 = sorted_face[shared_pos + 1]

    # 5. Keep only "smooth" adjacencies (dihedral close to flat).
    cos_thresh = float(np.cos(np.radians(dihedral_deg)))
    dots = np.einsum('ij,ij->i', fn[f0], fn[f1])
    smooth = dots > cos_thresh
    f0, f1 = f0[smooth], f1[smooth]

    # 6. Symmetric sparse adjacency over faces.
    rows = np.concatenate([f0, f1])
    cols = np.concatenate([f1, f0])
    data = np.ones(rows.shape[0], dtype=np.int8)
    adj = csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))

    # 7. Connected components → component containing the seed face is the prep.
    _, labels = connected_components(adj, directed=False)
    seed_label = labels[seed_face]
    keep = np.where(labels == seed_label)[0]
    if len(keep) == 0:
        return None

    # extract_cells returns UnstructuredGrid; downstream code (curvature,
    # .faces) needs PolyData, so convert back.
    return mesh.extract_cells(keep).extract_surface()


def split_prep_from_context(mesh: pv.PolyData, seed_point):
    """Split `mesh` into (prep, context) using pure topological connectivity.

    Many intra-oral scans deliver the prep tooth as a physically disconnected
    sub-mesh (visible black gaps around it). In that case the connected
    component containing the seed click is exactly the prep — no curvature
    threshold to tune, no risk of leaking across smooth crown surfaces.

    Returns (prep_mesh, context_mesh). Returns (None, None) if the seeded
    component is implausible as a tooth (covers most of the mesh, meaning the
    prep is NOT actually disconnected on this scan — caller should fall back).
    """
    if mesh is None or mesh.n_cells == 0:
        return None, None

    seed_face = int(mesh.find_closest_cell(np.asarray(seed_point, dtype=float)))
    if seed_face < 0:
        return None, None

    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    n_faces = len(faces)

    # Shared-edge face adjacency (no dihedral filter — every shared edge connects).
    e_a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e_b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    face_of_edge = np.tile(np.arange(n_faces, dtype=np.int64), 3)
    edge_key = np.sort(np.column_stack([e_a, e_b]), axis=1)
    order = np.lexsort([edge_key[:, 1], edge_key[:, 0]])
    sorted_keys = edge_key[order]
    sorted_face = face_of_edge[order]
    same = np.all(sorted_keys[:-1] == sorted_keys[1:], axis=1)
    shared_pos = np.where(same)[0]
    f0 = sorted_face[shared_pos]
    f1 = sorted_face[shared_pos + 1]

    rows = np.concatenate([f0, f1])
    cols = np.concatenate([f1, f0])
    data = np.ones(rows.shape[0], dtype=np.int8)
    adj = csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))

    _, labels = connected_components(adj, directed=False)
    seed_label = labels[seed_face]
    prep_mask = labels == seed_label
    n_prep = int(prep_mask.sum())

    # If the seeded component is essentially the whole arch, the prep isn't
    # topologically disconnected and connectivity-based isolation is useless.
    if n_prep < 50 or n_prep / n_faces > 0.85:
        return None, None

    # extract_cells returns UnstructuredGrid; downstream code (curvature on
    # prep, .faces access) needs PolyData, so convert both back.
    prep = mesh.extract_cells(np.where(prep_mask)[0]).extract_surface()
    context = mesh.extract_cells(np.where(~prep_mask)[0]).extract_surface()
    return prep, context
