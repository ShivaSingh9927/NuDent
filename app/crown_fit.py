"""Conform a preset crown's outer surface onto the margin ring.

Mental model (the user's): the margin loop is a rigid steel ring; the preset
tooth is rubber. Wherever the ring currently sits inside the tooth (set by the
tooth's pose), the tooth's outer cross-section at that level is forced to take
the ring's shape — zero gap, because that contour becomes the crown's finish
line. The deformation fades out above the ring so the occlusal anatomy is
preserved; below the ring it tapers back to the original shape (that part is
removed later by the Trim stage).

Method (no ICP, no iteration — O(n)):

  * Build a cylindrical frame around the margin's insertion axis: every point
    gets (height h along the axis, angle theta around it, radius from it).
  * Describe the ring by angle:  radm(theta), hm(theta)  (interpolated).
  * Measure the tooth's outer radius at the ring's height, per angle:
    rad_tooth_ring(theta).
  * Optional gentle uniform pre-scale so rad_tooth_ring ~ radm (small stretch).
  * Displace each tooth vertex radially by (radm - rad_tooth_ring) at its angle,
    weighted by a height falloff centred on the ring level. At the ring level
    the outer contour lands exactly on the ring; away from it the shift tapers
    to zero.

Pure numpy; takes/returns vertex arrays so it is trivially unit-testable.
"""
import numpy as np


def insertion_frame(margin_points, jaw_points=None):
    """Return (c, a, u, w): margin centroid, unit insertion axis, and two
    orthonormal in-plane axes. `a` is the margin loop's plane-normal (smallest
    PCA direction), sign-disambiguated to point toward the occlusal side using
    local jaw geometry when available."""
    pts = np.asarray(margin_points, dtype=float)
    c = pts.mean(axis=0)
    centered = pts - c
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    a = vh[-1]
    a = a / max(np.linalg.norm(a), 1e-12)

    if jaw_points is not None and len(jaw_points):
        jaw = np.asarray(jaw_points, dtype=float)
        extent = max(np.linalg.norm(centered, axis=1).max(), 1e-3)
        d = np.linalg.norm(jaw - c, axis=1)
        near = jaw[d < extent * 3]
        if len(near) == 0:
            near = jaw
        margin_proj = float(c @ a)
        proj = near @ a
        if int(np.sum(proj > margin_proj)) > int(np.sum(proj < margin_proj)):
            a = -a

    # Any vector not parallel to a, projected out -> first in-plane axis.
    tmp = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = tmp - (tmp @ a) * a
    u = u / max(np.linalg.norm(u), 1e-12)
    w = np.cross(a, u)
    return c, a, u, w


def _cylindrical(points, c, a, u, w):
    """Return (h, theta, rad, radial_dir) for points in the insertion frame.
    radial_dir is the unit outward in-plane direction per point (zeros where
    rad ~ 0)."""
    rel = np.asarray(points, dtype=float) - c
    h = rel @ a
    ru = rel @ u
    rw = rel @ w
    theta = np.arctan2(rw, ru)
    rad = np.hypot(ru, rw)
    perp = rel - np.outer(h, a)
    norm = np.linalg.norm(perp, axis=1)
    safe = np.maximum(norm, 1e-12)
    radial_dir = perp / safe[:, None]
    radial_dir[norm < 1e-9] = 0.0
    return h, theta, rad, radial_dir


def _periodic_interp(query, xp, fp):
    """Linear interpolation of fp(xp) at `query`, treating the angle domain as
    periodic over [-pi, pi]."""
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    order = np.argsort(xp)
    xs = xp[order]
    fs = fp[order]
    two_pi = 2.0 * np.pi
    xs_ext = np.concatenate([xs[-1:] - two_pi, xs, xs[:1] + two_pi])
    fs_ext = np.concatenate([fs[-1:], fs, fs[:1]])
    return np.interp(query, xs_ext, fs_ext)


def _outer_contour_by_angle(points, h, theta, rad, ring_h_at_vertex, band, n_bins):
    """Per-angle outer contour of the tooth at the ring's height.

    Vertices within `band` mm of the ring level (per their angle) are binned by
    angle; in each bin the *outermost* vertex (max radius) defines the contour
    there — both its radius and its full 3D position. Empty bins are filled by
    periodic interpolation from neighbours.

    Returns (centers, bin_radius, bin_pos):
        centers    : (n_bins,) bin-centre angles
        bin_radius : (n_bins,) outer radius per bin
        bin_pos    : (n_bins, 3) 3D position of that outer vertex per bin
    """
    dh_signed = h - ring_h_at_vertex
    dh = np.abs(dh_signed)
    sel = dh < band
    bins = np.floor((theta + np.pi) / (2.0 * np.pi) * n_bins).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)

    bin_radius = np.full(n_bins, np.nan)
    bin_pos = np.full((n_bins, 3), np.nan)
    idx_all = np.arange(len(points))
    if sel.any():
        sb = bins[sel]
        sr = rad[sel]
        sidx = idx_all[sel]
        sdh = dh[sel]
        for b in range(n_bins):
            m = sb == b
            if not m.any():
                continue
            rb = sr[m]; ib = sidx[m]; dhb = sdh[m]
            # The contour the ring maps to is the crown's outer surface *at the
            # ring plane* — not the widest point in the whole band. A crown that
            # flares wider going up would otherwise pick a vertex above the ring,
            # landing the conformed band too high. So restrict to the vertices
            # closest to the ring height, then take the outermost among those.
            order = np.argsort(dhb)
            k = max(1, int(np.ceil(0.4 * len(order))))
            near = order[:k]
            j = ib[near[np.argmax(rb[near])]]
            bin_radius[b] = rad[j]
            bin_pos[b] = points[j]

    centers = (np.arange(n_bins) + 0.5) / n_bins * 2.0 * np.pi - np.pi
    valid = ~np.isnan(bin_radius)
    if not valid.any():
        fallback_r = rad[sel].max() if sel.any() else rad.max()
        bin_radius[:] = fallback_r
        bin_pos[:] = points.mean(axis=0)
    elif not valid.all():
        bin_radius = _periodic_interp(centers, centers[valid], bin_radius[valid])
        for k in range(3):
            bin_pos[:, k] = _periodic_interp(centers, centers[valid], bin_pos[valid, k])

    # Smooth the contour circularly. Picking the single outermost vertex per bin
    # is noisy; without this the displacement field varies jaggedly between
    # adjacent angles and leaves vertical striations / a creased finish line.
    bin_radius = _circular_smooth(bin_radius, _CONTOUR_SMOOTH_BINS)
    for k in range(3):
        bin_pos[:, k] = _circular_smooth(bin_pos[:, k], _CONTOUR_SMOOTH_BINS)
    return centers, bin_radius, bin_pos


# Half-width (in bins) of the circular smoothing window applied to the sampled
# contour. ~5% of n_bins gives a clean finish line without rounding off real
# margin detail.
_CONTOUR_SMOOTH_BINS = 9


def _circular_smooth(values, half_width):
    """Periodic moving-average smoothing of a 1D array indexed by angle bin."""
    n = len(values)
    if half_width < 1 or n < 3:
        return values
    k = 2 * half_width + 1
    kernel = np.ones(k) / k
    padded = np.concatenate([values[-half_width:], values, values[:half_width]])
    return np.convolve(padded, kernel, mode="valid")


def _smoothstep_falloff(dh, blend_up, blend_down):
    """Weight in [0, 1]: 1 at the ring level (dh=0), tapering to 0 by blend_up
    above and blend_down below, with a smoothstep so there is no crease."""
    blend = np.where(dh >= 0, blend_up, blend_down)
    t = np.clip(np.abs(dh) / np.maximum(blend, 1e-6), 0.0, 1.0)
    return 1.0 - (t * t * (3.0 - 2.0 * t))


def _snap_to_margin(points, margin, radius=0.8, iters=3):
    """Final refinement: pull the (already roughly conformed) crown surface so it
    passes exactly through the margin ring.

    Each iteration finds, for every margin point, the nearest surface vertex and
    the residual to close; that residual is spread to nearby surface vertices
    with a smooth (quadratic) falloff over `radius`. A few iterations drive the
    margin-to-surface gap to near zero without disturbing geometry away from the
    ring. Requires SciPy's cKDTree (already a dependency)."""
    from scipy.spatial import cKDTree
    pts = np.asarray(points, dtype=float).copy()
    margin = np.asarray(margin, dtype=float)
    for _ in range(iters):
        tree = cKDTree(pts)
        _, idx = tree.query(margin, k=1)
        resid = margin - pts[idx]
        neighbors = tree.query_ball_point(margin, radius)
        corr = np.zeros_like(pts)
        wsum = np.zeros(len(pts))
        for mi, nb in enumerate(neighbors):
            if not nb:
                continue
            nb = np.asarray(nb, dtype=int)
            dd = np.linalg.norm(pts[nb] - margin[mi], axis=1)
            wt = np.clip(1.0 - dd / radius, 0.0, 1.0) ** 2
            corr[nb] += wt[:, None] * resid[mi]
            wsum[nb] += wt
        m = wsum > 0
        pts[m] += corr[m] / wsum[m][:, None]
    return pts


def _rotate_to(points, pivot, from_dir, to_dir):
    """Rigidly rotate `points` about `pivot` so unit `from_dir` maps to `to_dir`
    (Rodrigues). Pure rotation — preserves handedness/winding."""
    f = from_dir / max(np.linalg.norm(from_dir), 1e-12)
    t = to_dir / max(np.linalg.norm(to_dir), 1e-12)
    v = np.cross(f, t)
    s = np.linalg.norm(v)
    cth = float(np.clip(np.dot(f, t), -1.0, 1.0))
    if s < 1e-9:
        if cth > 0:
            return points.copy()
        # antiparallel: 180° about any axis perpendicular to f
        perp = np.array([1.0, 0.0, 0.0])
        if abs(f[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(f, perp); axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + 2.0 * (K @ K)
    else:
        axis = v / s
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + s * K + (1.0 - cth) * (K @ K)
    return (points - pivot) @ R.T + pivot


def auto_seat(crown_points, margin_points, jaw_points=None,
              search_mm=8.0, n_search=33):
    """Rigidly seat the crown onto the margin before deformation:

      1. Rotate the crown's long axis onto the insertion axis.
      2. Flip (180° about an in-plane axis) if needed so the bulky crown/cusp
         end points to the occlusal (+axis) side and the narrow root toward the
         prep (-axis) side.
      3. Slide along the axis to the offset that needs the *least* deformation —
         i.e. where the tooth's natural cross-section best matches the ring (the
         cervical finish line), instead of burying the ring mid-tooth.

    Returns (seated_points, c, a) where (c, a) is the margin centroid / axis.
    """
    pts = np.asarray(crown_points, dtype=float).copy()
    margin = np.asarray(margin_points, dtype=float)
    c, a, u, w = insertion_frame(margin, jaw_points)

    # 1. Long axis -> insertion axis.
    cen = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - cen, full_matrices=False)
    long_axis = vh[0]
    if np.dot(long_axis, a) < 0:
        long_axis = -long_axis
    pts = _rotate_to(pts, cen, long_axis, a)

    # 2. Ensure the wider (crown) end is on the +axis side.
    h = (pts - c) @ a
    upper = pts[h > np.median(h)]
    lower = pts[h <= np.median(h)]
    r_up = np.linalg.norm(upper - c - np.outer((upper - c) @ a, a), axis=1).mean()
    r_lo = np.linalg.norm(lower - c - np.outer((lower - c) @ a, a), axis=1).mean()
    if r_lo > r_up:  # crown is currently on the -axis side -> flip about u
        pts = _rotate_to(pts, cen, a, -a)

    # Move centroid to the margin centroid as the search origin.
    pts = pts - pts.mean(axis=0) + c

    # 3. Min-deformation axial slide.
    shifts = np.linspace(-search_mm, search_mm, n_search)
    best_shift, best_res = 0.0, np.inf
    for sft in shifts:
        trial = pts + a * sft
        r = fit_crown(trial, margin, jaw_points=jaw_points, prescale=True)
        if not r["ok"]:
            continue
        res = float(np.linalg.norm(r["points"] - trial, axis=1).mean())
        if res < best_res:
            best_res, best_shift = res, sft
    return pts + a * best_shift, c, a


def fit_crown(crown_points, margin_points, jaw_points=None,
              blend_up=2.5, blend_down=1.0, prescale=True,
              n_bins=360, contour_band=1.0,
              snap=True, snap_radius=0.8, snap_iters=4):
    """Conform `crown_points` onto the margin ring.

    Parameters
    ----------
    crown_points  : (N,3) array — the posed (undeformed) crown vertices.
    margin_points : (M,3) array — the closed margin loop.
    jaw_points    : (K,3) or None — used only to orient the insertion axis.
    blend_up      : mm, how far above the ring the deformation reaches.
    blend_down    : mm, how far below the ring it tapers (below is trimmed later,
                    so this is kept short just to avoid a discontinuity).
    prescale      : if True, uniformly scale the crown about the margin centroid
                    so its cross-section starts close to the ring (gentle warp).
    n_bins        : angular resolution for the tooth's outer contour.
    contour_band  : mm band around the ring level used to sample that contour.

    Returns
    -------
    dict:
        points : (N,3) deformed vertices
        scale  : the pre-scale factor applied (1.0 if prescale=False)
        ok     : False if inputs are degenerate
    """
    pts = np.asarray(crown_points, dtype=float).copy()
    margin = np.asarray(margin_points, dtype=float)
    if len(pts) < 4 or len(margin) < 3:
        return {"points": pts, "scale": 1.0, "ok": False}

    c, a, u, w = insertion_frame(margin, jaw_points)

    # Ring described by angle.
    mh, mtheta, mrad, _ = _cylindrical(margin, c, a, u, w)

    # Optional gentle uniform pre-scale about the margin centroid so the tooth's
    # cross-section near the ring already matches the ring size.
    scale = 1.0
    if prescale:
        h0, th0, rad0, _ = _cylindrical(pts, c, a, u, w)
        ring_h0 = _periodic_interp(th0, mtheta, mh)
        _, tooth_rad0, _ = _outer_contour_by_angle(
            pts, h0, th0, rad0, ring_h0, contour_band, n_bins)
        mean_tooth = float(np.nanmean(tooth_rad0))
        mean_ring = float(np.mean(mrad))
        if mean_tooth > 1e-6 and 0.2 < mean_ring / mean_tooth < 5.0:
            scale = mean_ring / mean_tooth
            pts = c + (pts - c) * scale

    # Recompute the tooth frame coords after scaling.
    h, theta, rad, _ = _cylindrical(pts, c, a, u, w)
    ring_h_at = _periodic_interp(theta, mtheta, mh)

    # The tooth's outer contour at the ring level, as full 3D points per angle.
    centers, _, contour_pos = _outer_contour_by_angle(
        pts, h, theta, rad, ring_h_at, contour_band, n_bins)

    # Ring as full 3D points per angle, and the tooth's contour at the same
    # angles. The displacement is the full 3D vector that takes the contour onto
    # the ring (lateral + tangential + vertical) — so the outer cross-section
    # lands exactly on the margin, not just at a matching radius.
    ring_pos_at = np.empty((len(pts), 3))
    contour_pos_at = np.empty((len(pts), 3))
    for k in range(3):
        ring_pos_at[:, k] = _periodic_interp(theta, mtheta, margin[:, k])
        contour_pos_at[:, k] = _periodic_interp(theta, centers, contour_pos[:, k])
    disp_full = ring_pos_at - contour_pos_at

    dh = h - ring_h_at
    weight = _smoothstep_falloff(dh, blend_up, blend_down)
    disp = disp_full * weight[:, None]
    out = pts + disp

    # Final refinement: snap the surface exactly onto the margin ring. The radial
    # conform gets within ~0.1-0.2 mm; this closes the rest so there is no gap.
    if snap:
        out = _snap_to_margin(out, margin, radius=snap_radius, iters=snap_iters)

    return {"points": out, "scale": scale, "ok": True}
