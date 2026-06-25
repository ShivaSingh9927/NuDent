"""Crown-border (Exocad-style "Crown Bottoms") geometry.

Ported from the sibling `nudent` project's core/cross_section.py so NuDent's
Cement stage can sweep the same 5-parameter border profile along the closed
margin loop and render it in the PyVista viewer.

Two pure-numpy functions, no GUI / PyVista dependency:

    compute_border_profile_2d(...)  -> ordered 2D polyline (local frame)
    build_border_band(...)          -> {verts, faces, ok} ribbon mesh
"""
import numpy as np


def compute_border_profile_2d(horizontal: float = 0.2,
                              angled: float = 0.0,
                              angle_deg: float = 45.0,
                              vertical: float = 0.0,
                              below_margin: float = 0.0):
    """Build the 2D cross-section polyline for the crown border, in a local
    frame attached to a single margin point:

        x : "outward"  — away from the tooth axis (lateral)
        y : "vertical" — along the insertion axis (positive = occlusal/up)

    Segments (all skipped when their length is 0):
        1. Horizontal  — flat bottom flange width
        2. Angled      — sloped segment length
        3. Angle       — slope of the angled segment, deg from horizontal
        4. Vertical    — straight wall height above the slope
        below_margin   — how far below the drawn margin the profile starts

    Returns an ordered list of (x, y). First point is (0, -below_margin).
    If horizontal/angled/vertical are all 0 the list has a single point and
    callers should treat that as "no band geometry" (build_border_band
    returns ok=False).
    """
    x, y = 0.0, -float(below_margin)
    pts = [(x, y)]

    if horizontal > 0:
        x += float(horizontal)
        pts.append((x, y))

    if angled > 0:
        rad = np.radians(float(angle_deg))
        x += float(angled) * np.cos(rad)
        y += float(angled) * np.sin(rad)
        pts.append((x, y))

    if vertical > 0:
        y += float(vertical)
        pts.append((x, y))

    return pts


def _local_outward(pts, i, n, closed, centroid):
    """Per-point local "outward" direction used to sweep the profile.

    Built from the curve tangent and world Z, then flipped to point away
    from the curve centroid (ignoring Z). Assumes the margin loop roughly
    encircles a vertical (Z) insertion axis — true for typical arch preps.
    """
    world_z = np.array([0.0, 0.0, 1.0])
    p = pts[i]

    if closed:
        p_prev = pts[(i - 1) % n]
        p_next = pts[(i + 1) % n]
    else:
        p_prev = pts[i - 1] if i > 0 else pts[i]
        p_next = pts[i + 1] if i < n - 1 else pts[i]

    tangent = p_next - p_prev
    tnorm = np.linalg.norm(tangent)
    tangent = tangent / tnorm if tnorm > 1e-9 else np.array([1.0, 0.0, 0.0])

    outward = np.cross(tangent, world_z)
    onorm = np.linalg.norm(outward)
    outward = outward / onorm if onorm > 1e-9 else np.array([1.0, 0.0, 0.0])

    radial = (p - centroid).copy()
    radial[2] = 0.0
    if np.dot(outward, radial) < 0:
        outward = -outward
    return outward


def build_border_band(margin_curve_pts, profile_2d, closed: bool = False):
    """Sweep a 2D border profile along a 3D margin curve into a ribbon mesh.

    For each margin point a local frame (outward, world_z) is built and every
    profile point (x, y) is placed at  curve_point + x*outward + y*world_z.

    Returns dict(verts, faces, ok). ok is False when inputs are degenerate
    (fewer than 2 curve points, or a single-point profile).
    """
    pts = np.asarray(margin_curve_pts, dtype=float)
    n = len(pts)
    m = len(profile_2d)
    if n < 2 or m < 2:
        return {"verts": [], "faces": [], "ok": False}

    world_z = np.array([0.0, 0.0, 1.0])
    centroid = pts.mean(axis=0)

    verts = []
    for i in range(n):
        p = pts[i]
        outward = _local_outward(pts, i, n, closed, centroid)
        for (lx, ly) in profile_2d:
            verts.append((p + outward * lx + world_z * ly).tolist())

    faces = []
    i_range = range(n) if closed else range(n - 1)
    for i in i_range:
        i_next = (i + 1) % n
        for j in range(m - 1):
            a = i * m + j
            b = i * m + (j + 1)
            c = i_next * m + j
            d = i_next * m + (j + 1)
            faces.append([a, b, c])
            faces.append([b, d, c])

    return {"verts": verts, "faces": faces, "ok": len(faces) > 0}
