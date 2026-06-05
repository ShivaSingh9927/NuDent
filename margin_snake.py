"""Detect a dental prep margin as a closed loop on a tooth mesh.

Pipeline:
  1. User clicks a point on the picked tooth.
  2. Crop the mesh to a sphere around the click — removes neighboring teeth,
     sectioning artifacts, and the model base.
  3. PCA on the cropped mesh estimates the tooth's long axis (cusp → gingival).
  4. Compute a per-vertex "margin-ness" score from pointwise mean curvature.
  5. Roll a virtual ball along the tooth axis from the click until the surface
     normal tilts persistently — that's the anchor on the margin.
  6. Active contour ("snake"): a ring of points around the axis at anchor
     height is iteratively pulled toward high-score vertices, smoothed by
     elastic forces, and re-projected to the surface — yielding a closed loop.
"""

import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree


# ---------- Interactive seed picker ----------

def pick_seed_point(mesh_path):
    try:
        import pyvista as pv
    except ImportError:
        print("pyvista not installed. Run: pip install pyvista")
        return None

    pv_mesh = pv.read(mesh_path)
    picked = {"point": None}

    def _cb(point, *args, **kwargs):
        picked["point"] = np.asarray(point, dtype=float)
        print(f"Picked: {picked['point']}")

    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, color="lightgray")
    plotter.add_text("Hover over the target tooth, press 'P', then close.",
                     font_size=10)
    plotter.enable_point_picking(callback=_cb, show_message=False,
                                 use_picker=True, pickable_window=False)
    plotter.show()
    return picked["point"]


# ---------- Cropping ----------

def crop_to_sphere(mesh, center, radius):
    """Keep only the submesh inside a sphere. Triangles are kept iff all three
    vertices are inside, so the result has clean borders."""
    dists = np.linalg.norm(mesh.vertices - center, axis=1)
    vert_mask = dists <= radius
    face_keep = vert_mask[mesh.faces].all(axis=1)
    new_faces = mesh.faces[face_keep]
    used = np.unique(new_faces.flatten())
    remap = -np.ones(len(mesh.vertices), dtype=int)
    remap[used] = np.arange(len(used))
    return trimesh.Trimesh(vertices=mesh.vertices[used],
                           faces=remap[new_faces], process=False)


def crop_to_component(mesh, center):
    """Return the connected mesh component that contains the click point.
    If the model has each tooth as a separate piece (as in segmented dental
    scans), this isolates exactly the picked tooth — much cleaner than a
    sphere crop that bleeds across cut planes."""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return None  # nothing to choose from; caller falls back to sphere crop
    # Pick the component whose vertices come closest to the click point.
    best, best_d = None, np.inf
    for comp in components:
        d = float(np.linalg.norm(comp.vertices - center, axis=1).min())
        if d < best_d:
            best_d = d
            best = comp
    print(f"  component crop: {len(components)} pieces, "
          f"picked one with closest vertex {best_d:.2f} mm from click")
    return best


# ---------- Tooth axis via PCA ----------

def estimate_tooth_axis(mesh, pick_point, local_radius=3.0):
    """Tooth long axis = average surface normal in a small region around the
    click. The click is on the cusp top, so its outward normal points along
    the tooth's long axis. This is much more reliable than full-mesh PCA when
    the component includes the die pedestal under the tooth."""
    tree = cKDTree(mesh.vertices)
    idxs = tree.query_ball_point(pick_point, local_radius)
    if len(idxs) < 5:
        # Fall back to nearest neighbors if the radius captured almost nothing.
        _, idxs = tree.query(pick_point, k=min(50, len(mesh.vertices)))
        idxs = np.atleast_1d(idxs)
    normals = np.asarray(mesh.vertex_normals)[idxs]
    axis = normals.mean(axis=0)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    centroid = mesh.vertices[idxs].mean(axis=0)
    print(f"  axis from {len(idxs)} local normals around pick.")
    return axis, centroid


# ---------- Per-vertex margin score ----------

def compute_margin_score(mesh, radius=0.4):
    """Pointwise |mean curvature|, robustly normalized to [0, 1].
    The margin ridge has high mean curvature; flat occlusal/axial regions do not.
    The artificial cut-plane base also scores high, but cropping already removed it."""
    H_int = trimesh.curvature.discrete_mean_curvature_measure(
        mesh, mesh.vertices, radius)
    vertex_area = np.zeros(len(mesh.vertices))
    for k in range(3):
        np.add.at(vertex_area, mesh.faces[:, k], mesh.area_faces / 3.0)
    vertex_area = np.maximum(vertex_area, 1e-9)
    H = H_int / vertex_area
    score = np.abs(H)
    # Robust min-max via percentiles so outliers don't squash the range.
    lo = np.percentile(score, 50)
    hi = np.percentile(score, 95)
    score = np.clip((score - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return score


# ---------- Find the anchor by rolling along the tooth axis ----------

def find_anchor(mesh, pick_point, axis, max_steps=400,
                bump_angle_deg=65, bump_confirm=4, dir_window=10,
                score=None, score_threshold=0.5):
    """Roll a ball from the click point in the direction of −axis (gravity
    aligned with the tooth's long axis). Stop when the surface's downhill
    direction tilts persistently — that bump is the margin."""
    tree = cKDTree(mesh.vertices)
    _, start = tree.query(pick_point)
    start = int(start)
    normals = np.asarray(mesh.vertex_normals)

    current = start
    path = [start]
    visited = {start}
    recent_dirs = []
    bump_counter = 0

    for step in range(max_steps):
        # Stop on first contact with the ridge (blue) field.
        if score is not None and score[current] >= score_threshold and step > 0:
            print(f"  ridge contact at step {step} "
                  f"(score={score[current]:.2f}). Stopping at margin.")
            break

        n = normals[current]
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            break
        n = n / n_norm
        g = -axis  # local gravity = tooth's gingival direction
        d = g - np.dot(g, n) * n  # gravity projected onto tangent plane
        m = np.linalg.norm(d)
        if m < 1e-4:
            print(f"  surface flat under ball at step {step}; at rest.")
            break
        d = d / m

        cur_pos = mesh.vertices[current]
        best, best_align = None, -np.inf
        for nb in mesh.vertex_neighbors[current]:
            nb = int(nb)
            if nb in visited:
                continue  # no revisits — prevents oscillation on plateaus
            step_vec = mesh.vertices[nb] - cur_pos
            sn = np.linalg.norm(step_vec)
            if sn < 1e-9:
                continue
            align = float(np.dot(step_vec / sn, d))
            if align <= 0:
                continue  # don't go against tangent-plane gravity
            if align > best_align:
                best_align = align
                best = nb
        if best is None:
            print(f"  no descending neighbor at step {step}.")
            break

        # Bump detection on the smoothed downhill direction.
        if len(recent_dirs) >= dir_window:
            avg = np.mean(recent_dirs[-dir_window:], axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-9)
            cos_a = float(np.dot(d, avg))
            ang = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
            if ang > bump_angle_deg:
                bump_counter += 1
                if bump_counter >= bump_confirm:
                    print(f"  bump at step {step} ({ang:.1f}°, "
                          f"{bump_counter} consecutive).")
                    break
            else:
                bump_counter = 0
        recent_dirs.append(d)
        current = best
        visited.add(best)
        path.append(current)

    return current, path


# ---------- Margin = high-score component touched by red path ----------

def margin_component(mesh, score, path, score_threshold=0.5,
                     walk_threshold=0.25, attach_radius=1.0):
    """Pick the connected mesh component of high-score (ridge) vertices that
    is touched by the red roll-down path. This is the actual prep margin.

    Two-tier thresholds: `score_threshold` (strong) controls which vertices
    qualify as seeds; `walk_threshold` (weak) controls BFS connectivity, so
    the loop can bridge thin gaps in the ridge."""
    strong = score >= score_threshold
    weak = score >= walk_threshold
    if not strong.any():
        print("No vertices pass strong threshold; lowering automatically.")
        score_threshold = float(np.percentile(score, 90))
        strong = score >= score_threshold
        walk_threshold = min(walk_threshold, score_threshold * 0.5)
        weak = score >= walk_threshold
        print(f"  using strong={score_threshold:.3f}, weak={walk_threshold:.3f}, "
              f"{strong.sum()} strong / {weak.sum()} weak.")

    # Find STRONG ridge vertices within attach_radius of the red path.
    path_pts = mesh.vertices[np.asarray(path, dtype=int)]
    strong_idx = np.where(strong)[0]
    if len(strong_idx) == 0:
        return np.array([], dtype=int)
    tree = cKDTree(path_pts)
    dists, _ = tree.query(mesh.vertices[strong_idx], k=1)
    seeds = strong_idx[dists <= attach_radius]
    print(f"  {len(seeds)} strong ridge vertices within {attach_radius} mm "
          f"of the red path.")

    if len(seeds) == 0:
        nearest = int(strong_idx[np.argmin(dists)])
        seeds = np.array([nearest])
        print(f"  fallback: snap to nearest strong ridge vertex "
              f"(d={float(dists.min()):.2f} mm).")

    # BFS through mesh adjacency, walking through any WEAK ridge vertex —
    # bridges sub-threshold gaps in the margin signal.
    component = set()
    frontier = list(int(s) for s in seeds)
    component.update(frontier)
    while frontier:
        nxt = []
        for v in frontier:
            for nb in mesh.vertex_neighbors[v]:
                nb = int(nb)
                if weak[nb] and nb not in component:
                    component.add(nb)
                    nxt.append(nb)
        frontier = nxt

    print(f"  margin component size: {len(component)} vertices "
          f"(seeded from {len(seeds)} strong, walked via weak threshold).")
    return np.array(sorted(component), dtype=int)


# ---------- Active contour (snake) ----------

def run_snake(mesh, anchor_idx, axis, score,
              n_points=120, iters=200,
              alpha=0.25, beta=0.6, attract_r=1.2,
              max_step=0.5):
    """A closed ring of points lies in the plane perpendicular to `axis` and
    passes through the anchor. Each iteration applies an image force (toward
    nearby high-score vertices), an elastic force (toward neighbor midpoints),
    then projects back to the mesh surface."""
    anchor_pt = mesh.vertices[anchor_idx]

    # Build orthonormal basis (u, v) in the plane normal to `axis`.
    if abs(axis[0]) < 0.9:
        u = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    else:
        u = np.cross(axis, np.array([0.0, 1.0, 0.0]))
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    v = v / np.linalg.norm(v)

    # Ring center: use vertices near the anchor's axial height (within a thin
    # slab) so the die pedestal below doesn't pull the center off-tooth.
    along = mesh.vertices @ axis
    anchor_along = float(anchor_pt @ axis)
    # 1.5 mm slab around the anchor's height
    slab = np.abs(along - anchor_along) < 1.5
    if slab.sum() < 20:
        slab_idx = np.argsort(np.abs(along - anchor_along))[:200]
    else:
        slab_idx = np.where(slab)[0]
    slab_pts = mesh.vertices[slab_idx]
    slab_centroid = slab_pts.mean(axis=0)
    offset = float(np.dot(slab_centroid - anchor_pt, axis))
    center = slab_centroid - offset * axis

    # Initial radius from the slab's extent perpendicular to the axis.
    rel = slab_pts - center
    rel_planar = rel - np.outer(rel @ axis, axis)
    planar_dists = np.linalg.norm(rel_planar, axis=1)
    radius = float(np.percentile(planar_dists, 95)) * 1.05
    if radius < 0.5:
        radius = 3.0
    print(f"Snake init: center={center}, radius={radius:.2f}, "
          f"slab vertices={len(slab_idx)}, n_points={n_points}")

    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    pts = (center
           + radius * np.cos(angles)[:, None] * u
           + radius * np.sin(angles)[:, None] * v)

    # Project initial ring onto the surface.
    pts, _, _ = trimesh.proximity.closest_point(mesh, pts)
    vtree = cKDTree(mesh.vertices)

    for it in range(iters):
        # Image force — weighted attraction toward nearby high-score vertices.
        image_force = np.zeros_like(pts)
        for i in range(len(pts)):
            idxs = vtree.query_ball_point(pts[i], attract_r)
            if not idxs:
                continue
            idxs = np.asarray(idxs, dtype=int)
            w = score[idxs] ** 2  # bias strongly toward high-score
            wsum = float(w.sum())
            if wsum < 1e-6:
                continue
            target = (mesh.vertices[idxs] * w[:, None]).sum(axis=0) / wsum
            image_force[i] = target - pts[i]

        # Elastic force — pull each point toward the midpoint of its two
        # ring-neighbors (keeps the loop smooth and stops it from collapsing
        # into a single vertex).
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)
        elastic_force = (prev_pts + next_pts) / 2.0 - pts

        move = alpha * image_force + beta * elastic_force
        # Cap per-step movement so the image force can't yank the whole ring
        # into a single hot spot in one iteration.
        mags = np.linalg.norm(move, axis=1)
        scale = np.where(mags > max_step, max_step / np.maximum(mags, 1e-9), 1.0)
        move = move * scale[:, None]
        pts = pts + move
        # Re-project onto the surface every step.
        pts, _, _ = trimesh.proximity.closest_point(mesh, pts)

    return pts


# ---------- Main driver ----------

def main(mesh_path, crop_radius=9.0, pick=None):
    if pick is None:
        pick = pick_seed_point(mesh_path)
    if pick is None:
        print("No pick — exiting.")
        return
    pick = np.asarray(pick, dtype=float)
    print(f"Using seed: {pick}")

    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    print("Trying connected-component crop (isolates one tooth if the model "
          "is segmented)...")
    sub = crop_to_component(mesh, pick)
    if sub is None:
        print(f"  single-component mesh — falling back to sphere crop "
              f"r={crop_radius}.")
        sub = crop_to_sphere(mesh, pick, crop_radius)
    print(f"  cropped to {len(sub.vertices)} verts, {len(sub.faces)} faces")
    if len(sub.vertices) < 200:
        print("Crop too small — try a larger crop_radius or a more central pick.")
        return

    print("Estimating tooth axis (PCA)...")
    axis, centroid = estimate_tooth_axis(sub, pick)
    print(f"  axis={axis}, centroid={centroid}")

    print("Computing margin score field...")
    score = compute_margin_score(sub, radius=0.4)
    print(f"  score range: min={score.min():.3f} mean={score.mean():.3f} "
          f"max={score.max():.3f}")

    print("Rolling ball along tooth axis to find anchor...")
    anchor_idx, path = find_anchor(sub, pick, axis, score=score,
                                   score_threshold=0.5)
    print(f"  anchor vertex {anchor_idx} at {sub.vertices[anchor_idx]}")
    print(f"  roll-down path length: {len(path)}")

    print("Selecting margin component from ridge field...")
    margin_idx = margin_component(sub, score, path,
                                  score_threshold=0.5, attach_radius=1.0)

    # ----- Visualization -----
    print("Visualizing — colored mesh + red roll-down + green snake loop.")
    colors = np.full((len(sub.vertices), 4), [210, 210, 210, 255], dtype=np.uint8)
    # Faint BLUE shading where the margin score is high — so it's visually
    # distinct from the bright-green snake output.
    sc8 = (score * 120).astype(np.uint8)
    colors[:, 0] = 210 - sc8           # less red
    colors[:, 1] = 210 - (sc8 // 2)    # slightly less green
    colors[:, 2] = 255                 # full blue

    # Red roll-down path (with 1-ring dilation for visibility).
    red_set = set(int(v) for v in path)
    for v in list(red_set):
        for nb in sub.vertex_neighbors[v]:
            red_set.add(int(nb))
    for v in red_set:
        colors[v] = [230, 40, 40, 255]

    # Green margin — the high-score component connected to the red endpoint.
    # Dilate by one ring for visibility.
    green_set = set(int(v) for v in margin_idx)
    for v in list(green_set):
        for nb in sub.vertex_neighbors[v]:
            green_set.add(int(nb))
    for v in green_set:
        colors[v] = [0, 220, 0, 255]

    sub.visual.vertex_colors = colors
    sub.show()


if __name__ == "__main__":
    test_tooth = ("/home/shiva/Documents/NuDent/Anatomic_Crown/2023-10-01_99999-011/2023-10-01_99999-011-lowerjaw.stl")
    # Set to None to use the interactive picker; otherwise this point is used.
    DEFAULT_PICK = [8.16557121, -17.37622261, 21.44959259]
    if os.path.exists(test_tooth):
        main(test_tooth, pick=DEFAULT_PICK)
    else:
        print(f"Test file not found: {test_tooth}")
