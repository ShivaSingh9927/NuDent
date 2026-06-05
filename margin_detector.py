import os
import trimesh
import numpy as np
from scipy.spatial import cKDTree


def pick_seed_point(mesh_path):
    """Open an interactive pyvista window. User presses 'P' over a point to
    pick it. Returns the picked (x, y, z) or None if nothing was picked."""
    try:
        import pyvista as pv
    except ImportError:
        print("pyvista not installed. Run: pip install pyvista")
        return None

    pv_mesh = pv.read(mesh_path)
    picked = {"point": None}

    def _cb(point, *args, **kwargs):
        picked["point"] = np.asarray(point, dtype=float)
        print(f"Picked seed point: {picked['point']}")

    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, color="lightgray", show_edges=False)
    plotter.add_text(
        "Hover over the target tooth and press 'P' to pick a seed point,\n"
        "then close this window to run detection.",
        font_size=10,
    )
    plotter.enable_point_picking(
        callback=_cb, show_message=False, use_picker=True, pickable_window=False
    )
    plotter.show()
    return picked["point"]

def detect_margin(mesh_path, seed_point=None):
    # 1. Load the mesh
    mesh = trimesh.load(mesh_path)
    print(f"Loaded mesh: {mesh_path}")
    
    # 2. Calculate Curvature (The 'Sharpness')
    # We use vertex normals and face adjacencies to find sharp edges
    # 'Curvature' here is a proxy for how sharp the edge is.
    print("Analyzing surface curvature...")
    curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, 0.5)
    # Normalize curvature for pathfinding weights.
    # Use a robust percentile (not the absolute max) so the artificial
    # base/cut-plane spike doesn't squash all real tooth-ridge curvature
    # to ~0 after min-max scaling.
    curv_abs = np.abs(curvature)
    lo = np.percentile(curv_abs, 5)
    hi = np.percentile(curv_abs, 95)
    curv_norm = np.clip((curv_abs - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    print(f"Curvature stats: abs min={curv_abs.min():.4f} "
          f"max={curv_abs.max():.4f} p95={hi:.4f}")
    print(f"Normalized curv >0.3: {(curv_norm > 0.3).sum()}, "
          f">0.15: {(curv_norm > 0.15).sum()}")

    # 3. Phase 1: Roll Down (Gradient Descent)
    if seed_point is None:
        # Default seed: the highest point on the tooth
        seed_idx = np.argmax(mesh.vertices[:, 2])
        pick_xy = mesh.vertices[seed_idx, :2].astype(float)
    else:
        # Find vertex closest to user click
        tree = cKDTree(mesh.vertices)
        _, seed_idx = tree.query(seed_point)
        pick_xy = np.asarray(seed_point[:2], dtype=float)

    # Tooth-scale horizontal radius: ~12% of the XY bbox diagonal.
    # This keeps the roll-down and the margin loop confined to one tooth.
    xy_extent = np.linalg.norm(mesh.vertices[:, :2].max(axis=0)
                               - mesh.vertices[:, :2].min(axis=0))
    tooth_radius = 0.12 * xy_extent
    print(f"Pick XY: {pick_xy}, tooth_radius: {tooth_radius:.2f}")

    print(f"Rolling ball from vertex {seed_idx}...")

    # We need vertex normals to project gravity onto the tangent plane.
    # (Same array is reused later for the bike ride.)
    vertex_normals_pre = np.asarray(mesh.vertex_normals)
    GRAVITY = np.array([0.0, 0.0, -1.0])
    # Z-range scale, needed for the descent tie-breaker.
    z_min_pre = float(mesh.vertices[:, 2].min())
    z_max_pre = float(mesh.vertices[:, 2].max())
    z_scale = max(z_max_pre - z_min_pre, 1e-6)

    current_idx = int(seed_idx)
    path_down = [current_idx]

    # Bump detector — uses tangent-plane downhill direction (not raw step
    # vector) so a step that's slightly sideways doesn't false-trigger.
    recent_downs = []
    DIR_WINDOW = 8
    DIR_BREAK_ANGLE_DEG = 50
    BUMP_CONFIRM_STEPS = 3
    bump_counter = 0

    def tangent_downhill(v_idx):
        """Project gravity onto the tangent plane at v_idx → downhill dir.
        Returns (unit_vector, raw_magnitude). magnitude is small on near-
        horizontal surfaces (the ball is at rest there)."""
        n = vertex_normals_pre[v_idx]
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            return None, 0.0
        n = n / n_norm
        d = GRAVITY - np.dot(GRAVITY, n) * n  # tangent-plane gravity
        m = np.linalg.norm(d)
        if m < 1e-6:
            return None, 0.0
        return d / m, m

    for step in range(500):
        # Stop if the ball has drifted off the picked tooth horizontally.
        cur_xy = mesh.vertices[current_idx, :2]
        if np.linalg.norm(cur_xy - pick_xy) > tooth_radius:
            print(f"Roll-down left tooth radius at step {step}; stopping.")
            break

        downhill, slope_mag = tangent_downhill(current_idx)
        if downhill is None:
            print(f"Surface horizontal at step {step} — ball at rest.")
            break

        cur_pos = mesh.vertices[current_idx]

        # Pick the neighbor whose displacement is most aligned with the
        # tangent-plane downhill direction AND that drops in Z.
        best = None
        best_score = -np.inf
        for n_idx in mesh.vertex_neighbors[current_idx]:
            n_idx = int(n_idx)
            step_vec = mesh.vertices[n_idx] - cur_pos
            s_norm = np.linalg.norm(step_vec)
            if s_norm < 1e-9:
                continue
            if mesh.vertices[n_idx, 2] >= cur_pos[2]:
                continue  # never climb
            step_dir = step_vec / s_norm
            align = float(np.dot(step_dir, downhill))
            if align <= 0:
                continue
            # Tie-break by absolute Z drop (faster descent preferred among
            # equally-aligned neighbors).
            score = align + 0.1 * (cur_pos[2] - mesh.vertices[n_idx, 2]) / z_scale
            if score > best_score:
                best_score = score
                best = n_idx

        if best is None:
            print(f"No downhill neighbor at step {step}; stopping.")
            break

        # Bump detection on the smoothed downhill direction itself.
        if len(recent_downs) >= DIR_WINDOW:
            avg = np.mean(recent_downs[-DIR_WINDOW:], axis=0)
            avg /= (np.linalg.norm(avg) + 1e-9)
            cos_a = float(np.dot(downhill, avg))
            ang_deg = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
            if ang_deg > DIR_BREAK_ANGLE_DEG:
                bump_counter += 1
                if bump_counter >= BUMP_CONFIRM_STEPS:
                    print(f"Bump confirmed at step {step}: downhill tilted "
                          f"{ang_deg:.1f}° for {bump_counter} steps. Stopping at margin.")
                    break
            else:
                bump_counter = 0
        recent_downs.append(downhill)

        current_idx = best
        path_down.append(current_idx)

    margin_start_node = current_idx

    # 4. Phase 2: The Bike Ride (Edge Tracking)
    print("Riding the bike along the margin ledge...")

    # First, mask out the artificial base/cut-plane of the scan (sharpest
    # edge globally) so it doesn't dominate the result.
    z_vals = mesh.vertices[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    base_cutoff = z_min + 0.05 * (z_max - z_min)  # ignore bottom 5% slab

    curv_thresh = 0.15
    high_curv = curv_norm > curv_thresh
    not_base = z_vals > base_cutoff
    candidate_mask = high_curv & not_base
    cand_idx = np.where(candidate_mask)[0]
    print(f"Candidate ridge vertices: {len(cand_idx)} "
          f"(curv > {curv_thresh}, z > {base_cutoff:.2f})")

    # Snap the roll-down endpoint to the nearest actual ridge vertex —
    # the ball usually stops on a flank, not on the sharp edge itself.
    if len(cand_idx) > 0:
        tree = cKDTree(mesh.vertices[cand_idx])
        _, nearest = tree.query(mesh.vertices[margin_start_node])
        seed = int(cand_idx[nearest])
    else:
        seed = margin_start_node

    # Build a "margin tracing" graph restricted to a Z-band around the
    # anchor's height. Edges are cheap when (a) the edge has high curvature
    # and (b) the edge stays near the anchor's Z. This biases Dijkstra to
    # follow the prep margin around the tooth instead of wandering up/down.
    anchor = seed
    anchor_z = float(mesh.vertices[anchor, 2])
    z_scale = max(z_max - z_min, 1e-6)
    # Asymmetric Z-band: allow room below the anchor, very little above it,
    # so the tracer can't climb onto the occlusal/cusp surface.
    z_above = 0.03 * z_scale
    z_below = 0.10 * z_scale
    in_z_band = (z_vals >= anchor_z - z_below) & (z_vals <= anchor_z + z_above)
    xy_dist = np.linalg.norm(mesh.vertices[:, :2] - pick_xy, axis=1)
    in_xy_radius = xy_dist <= tooth_radius
    walkable = candidate_mask & in_z_band & in_xy_radius
    # Looser mask used for *connectivity* — neighbor doesn't need to be
    # high-curv itself, just near the tooth. Curvature still drives scoring.
    near_tooth = in_z_band & in_xy_radius & (z_vals > base_cutoff)
    print(f"Walkable ridge vertices: {walkable.sum()}, "
          f"near-tooth vertices: {near_tooth.sum()}")

    # Vertex normals — needed for the gravity-perpendicular roll direction.
    vertex_normals = np.asarray(mesh.vertex_normals)
    gravity = np.array([0.0, 0.0, -1.0])
    Z_PENALTY = 8.0       # how strongly to pull back toward anchor_z
    MIN_DOT = 0.1         # neighbor must align with along-ledge dir at least this much
    Z_DRIFT_LIMIT = 0.08  # hard cap on |z - anchor_z| / z_scale (kill switch)
    LOOP_RETURN_RADIUS = None  # set after we know vertex spacing

    # Estimate average edge length so we can pick a "returned to start" radius.
    sample_n = min(2000, len(mesh.faces))
    sample_faces = mesh.faces[np.random.choice(len(mesh.faces), sample_n,
                                                replace=False)]
    edge_lengths = []
    for face in sample_faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge_lengths.append(np.linalg.norm(
                mesh.vertices[v1] - mesh.vertices[v2]))
    avg_edge = float(np.mean(edge_lengths))
    LOOP_RETURN_RADIUS = 3.0 * avg_edge
    print(f"avg edge length {avg_edge:.3f}, "
          f"loop-close radius {LOOP_RETURN_RADIUS:.3f}")

    def along_ledge_dir(vert_idx, sign_hint=None):
        """Compute the along-ledge tangent at vert_idx.
        Project gravity onto the tangent plane to get downhill, then take
        normal × downhill to get the perpendicular along-ledge direction.
        `sign_hint` (a vector) is used to lock direction sign so the ball
        doesn't reverse around the loop."""
        n = vertex_normals[vert_idx]
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            return None
        n = n / n_norm
        # downhill on the tangent plane = gravity - (gravity·n) n
        d = gravity - np.dot(gravity, n) * n
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-6:
            return None  # nearly flat (normal parallel to gravity)
        d = d / d_norm
        t = np.cross(n, d)  # along-ledge tangent
        t_norm = np.linalg.norm(t)
        if t_norm < 1e-9:
            return None
        t = t / t_norm
        if sign_hint is not None and np.dot(t, sign_hint) < 0:
            t = -t
        return t

    def roll_along_ledge(start, initial_sign_hint):
        """Roll a virtual ball along the ledge starting at `start`.
        At each step, pick the neighbor that best matches the
        gravity-perpendicular along-ledge direction, while staying near
        anchor_z. Stops on dead-end, Z drift, or loop closure."""
        path = [start]
        visited = {start}
        current = start
        sign_hint = np.asarray(initial_sign_hint, dtype=float)
        sign_hint /= (np.linalg.norm(sign_hint) + 1e-9)

        for step in range(5000):
            t = along_ledge_dir(current, sign_hint)
            if t is None:
                print(f"  step {step}: no along-ledge tangent (flat normal)")
                break

            cur_pos = mesh.vertices[current]
            best = None
            best_score = -np.inf
            for n in mesh.vertex_neighbors[current]:
                n = int(n)
                if n in visited:
                    # Allow loop closure when we get back near the start
                    # after enough steps.
                    if (n == start and step > 20):
                        path.append(n)
                        print(f"  loop closed at step {step}")
                        return path
                    continue
                if not near_tooth[n]:
                    continue
                step_vec = mesh.vertices[n] - cur_pos
                s_norm = np.linalg.norm(step_vec)
                if s_norm < 1e-9:
                    continue
                step_dir = step_vec / s_norm
                align = float(np.dot(step_dir, t))
                if align < MIN_DOT:
                    continue
                z_dev = abs(mesh.vertices[n, 2] - anchor_z) / z_scale
                if z_dev > Z_DRIFT_LIMIT:
                    continue
                # Score: alignment with along-ledge − Z drift penalty.
                # Curvature is a tiebreaker only (the gravity model
                # already implies the ledge is where the ball wants to be).
                score = align - Z_PENALTY * z_dev + 0.3 * curv_norm[n]
                if score > best_score:
                    best_score = score
                    best = n

            # Geometric loop-close check (we may not have stepped back to
            # the exact start vertex but landed very near it).
            if step > 20 and np.linalg.norm(cur_pos - mesh.vertices[start]) \
                    < LOOP_RETURN_RADIUS and best is not None:
                back_to_start = mesh.vertices[start] - cur_pos
                if np.linalg.norm(back_to_start) > 1e-9:
                    bts = back_to_start / np.linalg.norm(back_to_start)
                    if float(np.dot(bts, t)) > 0.3:
                        path.append(start)
                        print(f"  geometric loop close at step {step}")
                        return path

            if best is None:
                print(f"  step {step}: no valid neighbor (dead end)")
                break

            # Update sign_hint so the tangent flips don't reverse us.
            new_dir = mesh.vertices[best] - cur_pos
            new_dir /= (np.linalg.norm(new_dir) + 1e-9)
            sign_hint = 0.7 * new_dir + 0.3 * sign_hint  # low-pass

            path.append(best)
            visited.add(best)
            current = best

        return path

    loop_nodes = []
    if near_tooth[anchor]:
        # Initial sign hint: along-ledge tangent at the anchor itself.
        seed_t = along_ledge_dir(anchor, sign_hint=np.array([1.0, 0.0, 0.0]))
        if seed_t is None:
            print("Anchor has degenerate tangent (normal aligned with gravity).")
            seed_t = np.array([1.0, 0.0, 0.0])
        print("Rolling along ledge — forward direction")
        forward = roll_along_ledge(anchor, seed_t)
        print("Rolling along ledge — reverse direction")
        backward = roll_along_ledge(anchor, -seed_t)
        # Stitch: reversed(backward) + forward (anchor shared, drop dup).
        # If forward already closed the loop, just use it.
        if len(forward) > 1 and forward[-1] == anchor:
            loop_nodes = forward
        else:
            loop_nodes = list(reversed(backward)) + forward[1:]
        print(f"Ledge roller: forward {len(forward)}, backward {len(backward)}, "
              f"total {len(set(loop_nodes))}")
    else:
        print("Anchor not in near-tooth set (out of band/radius).")

    if loop_nodes:
        candidates = np.array(sorted(set(loop_nodes)), dtype=int)
    else:
        # Fallback: only show walkable ridge vertices near the picked tooth
        # (NOT every high-curv vertex in the entire arch).
        candidates = np.where(walkable)[0]
        print("Tracer failed — falling back to walkable set (local only).")
    print(f"Margin loop: {len(candidates)} vertices")
    
    # Simplify: Trace a path that stays in the curvature valley
    # For a production tool, we'd use a more complex loop-closure.
    # For now, let's highlight the 'Ledge' we found.
    
    # 5. Visualization
    # Color the mesh: Grey for tooth, Green for the Margin candidates
    colors = np.full((len(mesh.vertices), 4), [200, 200, 200, 255], dtype=np.uint8)
    # Dilate the ridge by one neighbor ring so it's actually visible.
    if len(candidates) > 0:
        dilated = set(int(v) for v in candidates)
        for v in list(dilated):
            for n in mesh.vertex_neighbors[v]:
                dilated.add(int(n))
        colors[np.array(sorted(dilated), dtype=int)] = [0, 255, 0, 255]
    colors[path_down] = [255, 0, 0, 255]   # Red for the 'Roll Down' path
    
    mesh.visual.vertex_colors = colors
    
    print("\n--- Margin Detection Complete ---")
    print(f"Red Line: Rolling down from the top.")
    print(f"Green Area: The detected margin ledge.")
    
    mesh.show()

if __name__ == "__main__":
    # Test on one of our converted teeth
    test_tooth = "/home/shiva/Documents/NuDent/Anatomic_Crown/sample/2023-10-01_99999-011-lowerjaw.stl"
    if os.path.exists(test_tooth):
        print("Opening seed picker — press 'P' on the target tooth, then close the window.")
        seed = pick_seed_point(test_tooth)
        if seed is None:
            print("No seed picked; falling back to highest-Z vertex.")
        detect_margin(test_tooth, seed_point=seed)
    else:
        print(f"Test file not found: {test_tooth}")
