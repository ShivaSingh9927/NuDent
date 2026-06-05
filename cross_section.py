import numpy as np
from collections import defaultdict


def get_cross_section(mesh, P, plane_normal, plane_up, radius):
    """
    Slice mesh at point P and return a 2D cross-section profile.

    Parameters
    ----------
    mesh        : trimesh.Trimesh
    P           : (3,) point on the mesh surface
    plane_normal: (3,) unit vector — camera right, NOT surface normal
    plane_up    : (3,) unit vector — camera up
    radius      : float — crop distance in mesh units (mm)

    Returns
    -------
    list of (x, y) floats — an ORDERED polyline through P, empty if the
    slice is degenerate or misses the mesh.
    """
    P = np.asarray(P, dtype=float)
    n = np.asarray(plane_normal, dtype=float)
    ln = np.linalg.norm(n)
    if ln < 1e-12:
        return []
    n = n / ln

    # --- 1. Slice ---
    section = mesh.section(plane_origin=P, plane_normal=n)
    if section is None:
        return []

    V = np.asarray(section.vertices, dtype=float)
    if len(V) < 2 or len(section.entities) == 0:
        return []

    # --- 2D in-plane axes (do this early; we project at the end) ---
    up = np.asarray(plane_up, dtype=float)
    up = up - np.dot(up, n) * n          # orthogonalise against n
    lu = np.linalg.norm(up)
    if lu < 1e-9:
        return []
    up = up / lu
    right = np.cross(n, up)
    right = right / np.linalg.norm(right)

    # --- 2. Build adjacency over ALL entities (handles split contours) ---
    # Section vertices are merged by trimesh, so a contour broken into several
    # entities still shares vertex indices at the split points — chain by index.
    adj = defaultdict(set)
    for ent in section.entities:
        idx = np.asarray(ent.points).ravel()
        for a, b in zip(idx[:-1], idx[1:]):
            a, b = int(a), int(b)
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
    if not adj:
        return []

    dist = np.linalg.norm(V - P, axis=1)
    seed = int(np.argmin(dist))
    if dist[seed] > radius:
        return []   # nothing local under the cursor

    # --- 3. Walk outward from the seed, staying within radius ---
    def walk(start, first):
        out = []
        prev, cur = start, first
        while dist[cur] <= radius:
            out.append(cur)
            cands = [x for x in adj[cur] if x != prev]
            if not cands:
                break
            if len(cands) == 1:
                nxt = cands[0]
            else:
                # junction: pick the straightest continuation
                vin = V[cur] - V[prev]
                nvin = np.linalg.norm(vin)
                if nvin < 1e-12:
                    nxt = cands[0]
                else:
                    vin /= nvin
                    nxt, best_dot = cands[0], -2.0
                    for c in cands:
                        vo = V[c] - V[cur]
                        nv = np.linalg.norm(vo)
                        if nv < 1e-12:
                            continue
                        d = float(np.dot(vin, vo / nv))
                        if d > best_dot:
                            best_dot, nxt = d, c
            prev, cur = cur, nxt
            if cur == start:    # closed loop
                break
        return out

    nb = list(adj[seed])
    dir1 = walk(seed, nb[0])
    dir2 = walk(seed, nb[1]) if len(nb) > 1 else []
    order = list(reversed(dir1)) + [seed] + dir2
    if len(order) < 2:
        return []

    # --- 4. Project ordered points to 2D, centred on P ---
    d = V[order] - P
    xs = d @ right
    ys = d @ up
    return list(zip(xs.tolist(), ys.tolist()))
