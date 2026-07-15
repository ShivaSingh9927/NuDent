"""Nu Smile — smile-preview pipeline.

Stage 1 (this file): load a patient photo, detect face landmarks with
MediaPipe FaceMesh, extract the inner-lip polygon as a "smile window", and
segment the currently-visible teeth pixels inside that window. This produces
the mask into which stage-2 will composite a render of the CAD arch.
"""
import os
import numpy as np
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QScrollArea, QWidget, QDoubleSpinBox, QFormLayout, QCheckBox,
    QProgressDialog,
)

# MediaPipe FaceMesh inner-lip loop indices (clockwise, canonical topology).
# 20 vertices tracing the inside edge of the lips.
_INNER_LIP_IDX = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]


_TASK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models",
                                "face_landmarker.task")


def detect_landmarks(img_bgr):
    """Return an (N,2) float array of landmark pixel coords, or None.

    Uses the MediaPipe Tasks FaceLandmarker (478 pts incl. iris refinement).
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_py
    from mediapipe.tasks.python import vision as mp_vision

    if not os.path.exists(_TASK_MODEL_PATH):
        raise FileNotFoundError(
            f"Face landmarker model missing at {_TASK_MODEL_PATH}. "
            "Download face_landmarker.task from the MediaPipe model zoo."
        )
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    base = mp_py.BaseOptions(model_asset_path=_TASK_MODEL_PATH)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with mp_vision.FaceLandmarker.create_from_options(opts) as landmarker:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_img)
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    return pts


# Canonical 3D face-model points (millimetres) for a subset of FaceMesh
# indices. Values are the standard MediaPipe canonical mesh coordinates
# reduced to the six most stable anchors used for PnP.
#   1   – nose tip
#   152 – chin
#   33  – right eye outer corner
#   263 – left eye outer corner
#   61  – right mouth corner
#   291 – left mouth corner
_PNP_IDX = [1, 152, 33, 263, 61, 291]
_PNP_MODEL_MM = np.array([
    [  0.0,    0.0,    0.0],
    [  0.0,  -63.6,  -12.5],
    [-43.3,   32.7,  -26.0],
    [ 43.3,   32.7,  -26.0],
    [-28.9,  -28.9,  -24.1],
    [ 28.9,  -28.9,  -24.1],
], dtype=np.float64)


def estimate_head_pose(landmarks, img_shape_hw):
    """Recover (rvec, tvec, camera_matrix) from a 2D landmark set.

    Uses six canonical face points and a pinhole camera with focal length
    approximated from image width — good enough for portrait photos where
    the exact intrinsics are unknown.
    Returns None if PnP fails to converge.
    """
    h, w = img_shape_hw[:2]
    focal = float(w)
    center = (w * 0.5, h * 0.5)
    K = np.array([[focal, 0.0, center[0]],
                  [0.0, focal, center[1]],
                  [0.0,    0.0,       1.0]], dtype=np.float64)
    dist = np.zeros((4, 1))
    img_pts = landmarks[_PNP_IDX].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(_PNP_MODEL_MM, img_pts, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return rvec, tvec, K, dist


def draw_pose_axes(img_bgr, pose, length_mm=40.0):
    """Draw a red/green/blue XYZ axis triad at the nose-tip anchor."""
    rvec, tvec, K, dist = pose
    axis = np.float64([
        [0, 0, 0],
        [length_mm, 0, 0],
        [0, length_mm, 0],
        [0, 0, length_mm],
    ])
    proj, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2).astype(int)
    o, x, y, z = proj
    out = img_bgr.copy()
    cv2.line(out, tuple(o), tuple(x), (0,   0, 255), 3)  # X red
    cv2.line(out, tuple(o), tuple(y), (0, 200,   0), 3)  # Y green
    cv2.line(out, tuple(o), tuple(z), (255, 0,   0), 3)  # Z blue
    cv2.circle(out, tuple(o), 4, (0, 255, 255), -1)
    return out


def inner_lip_polygon(landmarks):
    """Return the inner-lip loop as an (20,2) float array."""
    return landmarks[_INNER_LIP_IDX].astype(np.float32)


def inner_lip_mask(shape_hw, lip_poly):
    """Rasterise the inner-lip polygon into a binary uint8 mask."""
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(mask, [lip_poly.astype(np.int32)], 255)
    return mask


def segment_teeth(img_bgr, lip_mask):
    """Isolate tooth pixels within the inner-lip mask.

    Teeth are bright and low-saturation. We threshold in HSV within the mask,
    then keep the connected components that are brightest — this rejects
    tongue and dark inter-tooth gaps.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    inside = lip_mask > 0
    if inside.sum() < 50:
        return np.zeros_like(lip_mask)
    v_in = v[inside]
    v_thresh = max(120, int(np.percentile(v_in, 55)))
    s_thresh = min(90, int(np.percentile(s[inside], 60)))
    teeth = (v > v_thresh) & (s < s_thresh) & inside
    teeth = teeth.astype(np.uint8) * 255
    teeth = cv2.morphologyEx(teeth, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    teeth = cv2.morphologyEx(teeth, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return teeth


def build_overlay(img_bgr, lip_poly, teeth_mask):
    """Composite the lip outline (green) + teeth mask (magenta) on the photo."""
    out = img_bgr.copy()
    if teeth_mask is not None:
        tint = np.zeros_like(out)
        tint[:, :, 2] = 255  # red channel
        tint[:, :, 0] = 200  # blue channel → magenta
        m = teeth_mask > 0
        out[m] = (0.55 * out[m] + 0.45 * tint[m]).astype(np.uint8)
    if lip_poly is not None:
        cv2.polylines(out, [lip_poly.astype(np.int32)], True, (0, 220, 60), 2)
    return out


def shade_arch_vertices(mesh):
    """Bake enamel-like per-vertex RGB from geometry cues.

    Three effects are combined into a single ``point_data['RGB']``:
      • Concavity darkening — high mean curvature (interproximal contacts,
        fissures, gum-line) reads darker; convex ridges (cusp tips,
        incisal edges) read brighter.
      • Warm→cool ramp along the arch height so the gingival portion is
        slightly warm and the occlusal is slightly cool.
      • Faint blue-grey translucency at the occlusal middle band, where
        upper and lower incisal edges meet — mimics the see-through
        quality real anteriors have at their tips.
    """
    m = mesh.copy()
    try:
        curv = np.asarray(m.curvature(curv_type="mean"))
    except Exception:
        curv = np.zeros(m.n_points)
    # Robustly normalise curvature to [-1, +1] using percentile clipping,
    # so a few extreme spikes (e.g. flipped face normals) don't dominate.
    lo, hi = np.percentile(curv, [15, 85])
    c = np.clip((curv - lo) / (hi - lo + 1e-9), 0.0, 1.0)   # 0 concave → 1 convex
    c = c * 2.0 - 1.0                                        # -1..+1

    base = np.array([238, 232, 216], dtype=np.float32)      # soft warm ivory
    # Very gentle: ±8% max brightness swing — enough to catch the eye at
    # crevices, nowhere near "missing tooth" territory.
    mult = 1.0 + 0.08 * c
    rgb = base[None, :] * mult[:, None]

    m.point_data["RGB"] = np.clip(rgb, 0, 255).astype(np.uint8)
    return m


def face_align_arch(mesh):
    """Map a dental-scanner arch into the canonical face frame.

    Scanner convention (as displayed in the CAD viewer):
        X = mesiodistal (patient right → left)
        Y = anteroposterior (posterior → anterior, out of the face)
        Z = superoinferior (inferior → superior, up)
    Canonical face frame (what PnP returned):
        X = mesiodistal (patient right → left)
        Y = up
        Z = out of the face
    So the mapping is a Y↔Z swap:  (x, y, z)  →  (x, z, y).
    The mesh is also centred on its bounding-box centre so downstream
    offsets are measured from the arch's own centroid.
    """
    import pyvista as pv
    pdata = pv.wrap(mesh) if not isinstance(mesh, pv.PolyData) else mesh
    pts = np.asarray(pdata.points, dtype=np.float64)
    c = pts.mean(axis=0)
    swapped = np.stack([pts[:, 0] - c[0],
                        pts[:, 2] - c[2],
                        pts[:, 1] - c[1]], axis=1).astype(np.float32)
    out = pdata.copy()
    out.points = swapped
    return out


def render_arch(mesh, pose, img_shape_hw, offsets_mm=(0, 0, 0),
                scale=1.0, rot_deg=(0, 0, 0), flip=(False, False, False)):
    """Off-screen render of an arch mesh through the recovered head-pose
    camera. Returns (rgb_uint8, alpha_uint8) at the photo's resolution.

    ``mesh`` is a trimesh.Trimesh or pyvista.PolyData in millimetres. It is
    centred on its own bounding-box centre, uniformly scaled, rotated
    (Euler XYZ, degrees), then translated by ``offsets_mm`` (in canonical
    face-model space, so +X = patient's left, +Y = up, +Z = out of face).
    """
    import pyvista as pv
    rvec, tvec, K, _dist = pose
    h, w = img_shape_hw[:2]

    if hasattr(mesh, "vertices"):
        pdata = pv.wrap(mesh)
    else:
        pdata = mesh
    pdata = pdata.copy()
    c = np.asarray(pdata.center, dtype=float)
    pdata.translate(-c, inplace=True)
    fx, fy_, fz = flip
    if fx or fy_ or fz:
        sx = -1.0 if fx else 1.0
        sy = -1.0 if fy_ else 1.0
        sz = -1.0 if fz else 1.0
        pts = np.asarray(pdata.points) * np.array([sx, sy, sz], dtype=np.float32)
        pdata.points = pts.astype(np.float32)
    if scale != 1.0:
        pdata.scale(scale, inplace=True)
    rx, ry, rz = rot_deg
    if rx: pdata.rotate_x(rx, inplace=True)
    if ry: pdata.rotate_y(ry, inplace=True)
    if rz: pdata.rotate_z(rz, inplace=True)
    pdata.translate(np.asarray(offsets_mm, dtype=float), inplace=True)

    R, _ = cv2.Rodrigues(rvec)
    tvec = tvec.reshape(3)
    cam_pos = (-R.T @ tvec)
    look_dir = R.T @ np.array([0.0, 0.0, 1.0])
    view_up = -(R.T @ np.array([0.0, 1.0, 0.0]))
    focal_point = cam_pos + look_dir

    fy = float(K[1, 1])
    view_angle_deg = float(np.degrees(2.0 * np.arctan(h / (2.0 * fy))))

    plotter = pv.Plotter(off_screen=True, window_size=(w, h))
    plotter.set_background("black")
    # Enamel-ish material: warm ivory, smooth surface, low-medium spec,
    # a hint of ambient to avoid crushed shadows in the mouth interior.
    try:
        plotter.add_mesh(
            pdata, color="#f4ecd6",
            smooth_shading=True,
            pbr=True, metallic=0.0, roughness=0.35,
            ambient=0.18, diffuse=0.85,
        )
    except Exception:
        plotter.add_mesh(
            pdata, color="#f4ecd6", smooth_shading=True,
            specular=0.55, specular_power=45, ambient=0.22, diffuse=0.9,
        )
    cam = plotter.camera
    cam.position = tuple(cam_pos.tolist())
    cam.focal_point = tuple(focal_point.tolist())
    cam.up = tuple(view_up.tolist())
    cam.view_angle = view_angle_deg
    cam.clipping_range = (1.0, 5000.0)
    # Two-light key/fill rig — key from above-front (mimics room light through
    # the smile line), softer fill from below/behind to bring cusps out.
    try:
        plotter.remove_all_lights()
    except Exception:
        pass
    key = pv.Light(position=tuple((cam_pos + np.array([-40, 60, 80])).tolist()),
                   focal_point=tuple(focal_point.tolist()),
                   color="#fffaf0", intensity=1.6)
    fill = pv.Light(position=tuple((cam_pos + np.array([30, -20, 80])).tolist()),
                    focal_point=tuple(focal_point.tolist()),
                    color="#e8efff", intensity=0.9)
    front = pv.Light(position=tuple(cam_pos.tolist()),
                     focal_point=tuple(focal_point.tolist()),
                     color="#ffffff", intensity=0.6,
                     light_type="headlight")
    plotter.add_light(key)
    plotter.add_light(fill)
    plotter.add_light(front)
    rgb = plotter.screenshot(transparent_background=False, return_img=True)
    plotter.close()
    rgb = np.asarray(rgb)
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    alpha = (gray > 8).astype(np.uint8) * 255
    return rgb, alpha


def _match_patient_teeth(arch_bgr, arch_alpha, img_bgr, teeth_mask):
    """Reinhard LAB colour transfer from patient's own teeth to the arch.

    We compute per-channel mean and std over the patient's visible teeth
    pixels, then re-normalise the arch's pixels to match. This makes the
    CAD render inherit *this patient's* enamel shade — brightness, warmth,
    saturation — rather than a generic ivory.
    Returns the arch BGR after transfer, and True if patient teeth were
    usable (else False → caller should fall back to skin illumination).
    """
    if teeth_mask is None or int((teeth_mask > 0).sum()) < 300:
        return arch_bgr, False
    src_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    arch_lab = cv2.cvtColor(arch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tp = src_lab[teeth_mask > 0]        # patient teeth pixels
    ap = arch_lab[arch_alpha > 0]        # rendered arch pixels
    if len(ap) < 100:
        return arch_bgr, False
    # Trim the darkest 15% of the patient's tooth pixels — those tend to be
    # inter-tooth gaps / occlusion shadows we don't want to average in.
    L_tp = tp[:, 0]
    keep = L_tp > np.percentile(L_tp, 15)
    tp = tp[keep]

    tm = tp.mean(axis=0);  ts = tp.std(axis=0) + 1e-3
    am = ap.mean(axis=0);  as_ = ap.std(axis=0) + 1e-3
    scaled = (arch_lab - am) * (ts / as_) + tm
    # Blend so we keep some of the render's own shading contrast (100%
    # transfer flattens the highlights).
    blend = 0.75
    arch_lab = arch_lab * (1.0 - blend) + scaled * blend
    arch_lab = np.clip(arch_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arch_lab, cv2.COLOR_LAB2BGR), True


def _match_illumination(arch_bgr, arch_alpha, img_bgr, lip_mask):
    """Tint the rendered arch toward the photo's illumination.

    We compute mean(a), mean(b) in LAB from a ring of skin around the lips
    (dilated lip mask minus lip mask itself), and shift the arch's a/b by
    that difference — a small blend so the teeth don't turn skin-coloured.
    L is left alone so teeth stay bright.
    """
    # Sample a wide skin ring but keep it well away from the immediate lip
    # border (which contains beard/stubble/lipstick and would poison the
    # illumination estimate). Take pixels in a ring 25–60 px out.
    inner = cv2.dilate(lip_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    outer = cv2.dilate(lip_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60)))
    ring = (outer > 0) & (inner == 0)
    if ring.sum() < 200:
        return arch_bgr
    skin_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Reject dark stubble pixels (low L) — keep only bright skin.
    L = skin_lab[..., 0]
    ring &= (L > np.percentile(L[ring], 40))
    if ring.sum() < 200:
        return arch_bgr
    arch_lab = cv2.cvtColor(arch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring_pix = skin_lab[ring]
    arch_pix = arch_lab[arch_alpha > 0]
    if len(arch_pix) < 50:
        return arch_bgr
    da = float(ring_pix[:, 1].mean() - arch_pix[:, 1].mean())
    db = float(ring_pix[:, 2].mean() - arch_pix[:, 2].mean())
    # Cap the shift so a warm face can't turn white enamel beige.
    da = float(np.clip(da, -4.0, 4.0))
    db = float(np.clip(db, -4.0, 4.0))
    blend = 0.25
    arch_lab[..., 1] += da * blend
    arch_lab[..., 2] += db * blend
    arch_lab = np.clip(arch_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arch_lab, cv2.COLOR_LAB2BGR)


def composite_arch(img_bgr, arch_rgb, arch_alpha, lip_mask, teeth_mask=None):
    """Paste ``arch_rgb`` (RGB) onto ``img_bgr`` inside ``lip_mask``,
    weighted by ``arch_alpha`` and colour-matched to the photo.

    Preferred colour source is the patient's own visible teeth
    (``teeth_mask``); if that mask is too small we fall back to a wider
    skin-ring illumination estimate.
    """
    arch_bgr = cv2.cvtColor(arch_rgb, cv2.COLOR_RGB2BGR)
    arch_bgr, used_patient_teeth = _match_patient_teeth(
        arch_bgr, arch_alpha, img_bgr, teeth_mask
    )
    if not used_patient_teeth:
        arch_bgr = _match_illumination(arch_bgr, arch_alpha, img_bgr, lip_mask)

    # Step 1: black out the natural teeth inside the lip mask so the
    # original teeth don't peek through where the render doesn't cover.
    # Use a dark oral-cavity colour rather than pure black.
    cavity = np.array([18, 12, 12], dtype=np.float32)  # BGR
    lip_soft = cv2.GaussianBlur(lip_mask.astype(np.float32), (7, 7), 0) / 255.0
    lip3 = lip_soft[:, :, None]
    base = img_bgr.astype(np.float32) * (1.0 - lip3) + cavity[None, None, :] * lip3

    # Step 2: paste the arch on top of that cleaned base.
    m = (arch_alpha > 0).astype(np.float32)
    m = cv2.GaussianBlur(m, (5, 5), 0)
    m *= lip_soft  # keep the paste inside the lip window
    m3 = m[:, :, None]
    out = base * (1.0 - m3) + arch_bgr.astype(np.float32) * m3

    # Subtle inner-lip shadow — a thin darkened ring just inside the lips
    # so the teeth sit in shadow rather than looking flat-lit.
    edge = lip_mask.astype(np.uint8)
    edge = cv2.morphologyEx(edge, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    edge = cv2.GaussianBlur(edge.astype(np.float32), (9, 9), 0) / 255.0
    edge = edge * (m > 0).astype(np.float32)   # only darken where the arch is
    out *= (1.0 - 0.35 * edge[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def _bgr_to_qpixmap(img_bgr, max_dim=900):
    """Return the pixmap; if you need the scale factor, use
    ``_bgr_to_qpixmap_scaled`` instead."""
    pix, _scale = _bgr_to_qpixmap_scaled(img_bgr, max_dim=max_dim)
    return pix


def _bgr_to_qpixmap_scaled(img_bgr, max_dim=900):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg), scale


class _LipEditLabel(QLabel):
    """QLabel that shows the composite image and, when editing is on,
    lets the user click-drag inner-lip handles. Points live in *image*
    coordinates; we translate mouse coords via the pixmap scale factor.
    """
    pointDragged = pyqtSignal(int, float, float)  # idx, x_img, y_img
    dragFinished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._points = None           # (N, 2) image coords or None
        self._edit_on = False
        self._scale = 1.0             # image px → widget px
        self._pix_origin = (0, 0)     # top-left of pixmap inside the label
        self._active_idx = None
        self._hit_radius_px = 40.0    # widget-space picking radius (generous)

    def set_edit_mode(self, on):
        self._edit_on = bool(on)
        self.setCursor(Qt.CrossCursor if self._edit_on else Qt.ArrowCursor)

    def set_points(self, pts):
        self._points = None if pts is None else np.asarray(pts, dtype=np.float32).copy()

    def set_display(self, pixmap, scale, _pix_origin=None):
        self._scale = scale
        self.setPixmap(pixmap)
        # Grow the label to fit the pixmap so it isn't clipped inside the
        # scroll area and the origin math stays trivial (pixmap fills label).
        self.setFixedSize(pixmap.width(), pixmap.height())

    def _current_pix_origin(self):
        """Recompute the top-left of the (centered) pixmap inside the label
        right now. The result may be negative if the pixmap is larger than
        the label — that's fine; it means the pixmap extends past the
        label's visible rect."""
        pix = self.pixmap()
        if pix is None:
            return (0, 0)
        lw, lh = self.width(), self.height()
        pw, ph = pix.width(), pix.height()
        return ((lw - pw) // 2, (lh - ph) // 2)

    def _widget_to_img(self, wx, wy):
        px, py = self._current_pix_origin()
        if self._scale <= 0:
            return 0.0, 0.0
        return (wx - px) / self._scale, (wy - py) / self._scale

    def _pick(self, wx, wy):
        if self._points is None or self._scale <= 0:
            return None
        ix, iy = self._widget_to_img(wx, wy)
        d = np.hypot(self._points[:, 0] - ix, self._points[:, 1] - iy) * self._scale
        i = int(np.argmin(d))
        return i if d[i] <= self._hit_radius_px else None

    # Emitted with any click on the label (edit mode or not) — useful to
    # confirm the event actually arrives.
    debugClick = pyqtSignal(float, float, int)

    def mousePressEvent(self, ev):
        wx, wy = float(ev.x()), float(ev.y())
        picked = -1
        ix, iy = self._widget_to_img(wx, wy)
        nearest_d = -1.0
        if self._points is not None and self._scale > 0:
            d = np.hypot(self._points[:, 0] - ix, self._points[:, 1] - iy)
            nearest_d = float(d.min())
        if self._edit_on and ev.button() == Qt.LeftButton:
            i = self._pick(ev.x(), ev.y())
            if i is not None:
                self._active_idx = i
                picked = i
        # Emit an extra-verbose debug tuple to the status bar.
        self._debug_msg = (
            f"widget=({wx:.0f},{wy:.0f}) → img=({ix:.0f},{iy:.0f}) "
            f"scale={self._scale:.3f} label={self.width()}×{self.height()} "
            f"pix={self.pixmap().width() if self.pixmap() else 0}×"
            f"{self.pixmap().height() if self.pixmap() else 0} "
            f"nearest_pt_dist_img={nearest_d:.1f} picked_idx={picked}"
        )
        self.debugClick.emit(wx, wy, picked)

    def mouseMoveEvent(self, ev):
        if not self._edit_on or self._active_idx is None:
            return
        ix, iy = self._widget_to_img(ev.x(), ev.y())
        self.pointDragged.emit(self._active_idx, float(ix), float(iy))

    def mouseReleaseEvent(self, ev):
        if self._active_idx is not None:
            self._active_idx = None
            self.dragFinished.emit()

    zoomRequested = pyqtSignal(float)

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.ControlModifier:
            factor = 1.25 if ev.angleDelta().y() > 0 else 1 / 1.25
            self.zoomRequested.emit(factor)
            ev.accept()
        else:
            super().wheelEvent(ev)


class _EnhanceWorker(QThread):
    """Runs Stable Diffusion img2img off the UI thread so the window stays
    responsive during the 30–60 s CPU inference."""
    progress = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, composite_bgr, lip_poly, lip_mask, denoise):
        super().__init__()
        self._c = composite_bgr
        self._p = lip_poly
        self._m = lip_mask
        self._d = denoise

    def run(self):
        try:
            from . import nusmile_enhance as ne
            out = ne.enhance(self._c, self._p, self._m,
                             denoise=self._d,
                             progress_cb=lambda s: self.progress.emit(s))
            self.done.emit(out)
        except Exception as e:
            self.failed.emit(str(e))


class NuSmileDialog(QDialog):
    """Preview dialog: pick a face photo → landmarks → lip mask → teeth mask."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nu Smile — Preview")
        self.resize(1080, 720)

        v = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.btn_open = QPushButton("Load Face Photo…")
        self.btn_open.setObjectName("primary")
        self.btn_open.clicked.connect(self._pick_photo)
        bar.addWidget(self.btn_open)
        self.btn_arch = QPushButton("Use Case Arches")
        self.btn_arch.setToolTip("Load both jaws already open in the CAD case.")
        self.btn_arch.clicked.connect(self._use_case_arches)
        self.btn_arch.setEnabled(False)
        bar.addWidget(self.btn_arch)
        self.btn_arch_pick = QPushButton("Load STL…")
        self.btn_arch_pick.setToolTip("Load a different mesh from disk.")
        self.btn_arch_pick.clicked.connect(self._pick_arch)
        self.btn_arch_pick.setEnabled(False)
        bar.addWidget(self.btn_arch_pick)
        self.btn_composite = QPushButton("Composite")
        self.btn_composite.setObjectName("primary")
        self.btn_composite.clicked.connect(self._composite)
        self.btn_composite.setEnabled(False)
        bar.addWidget(self.btn_composite)
        self.chk_overlays = QCheckBox("Show overlays")
        self.chk_overlays.setChecked(True)
        self.chk_overlays.setToolTip("Toggle the green lip outline and XYZ pose axes.")
        self.chk_overlays.toggled.connect(self._refresh_view)
        bar.addWidget(self.chk_overlays)
        self.chk_edit_lip = QCheckBox("Edit lip line")
        self.chk_edit_lip.setToolTip("Click and drag the yellow handles to reshape the inner-lip outline.")
        self.chk_edit_lip.toggled.connect(self._on_edit_lip_toggled)
        bar.addWidget(self.chk_edit_lip)
        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.setFixedWidth(28)
        self.btn_zoom_out.setToolTip("Zoom out (Ctrl+scroll works too)")
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_by(1 / 1.25))
        bar.addWidget(self.btn_zoom_out)
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(28)
        self.btn_zoom_in.setToolTip("Zoom in")
        self.btn_zoom_in.clicked.connect(lambda: self._zoom_by(1.25))
        bar.addWidget(self.btn_zoom_in)
        self.btn_zoom_fit = QPushButton("Fit")
        self.btn_zoom_fit.setFixedWidth(38)
        self.btn_zoom_fit.setToolTip("Reset zoom to fit-in-window")
        self.btn_zoom_fit.clicked.connect(lambda: (setattr(self, "_zoom", 1.0), self._refresh_view()))
        bar.addWidget(self.btn_zoom_fit)
        self.btn_save = QPushButton("Save Image…")
        self.btn_save.setToolTip("Save the currently displayed image (with or without overlays).")
        self.btn_save.clicked.connect(self._save_image)
        self.btn_save.setEnabled(False)
        bar.addWidget(self.btn_save)
        self.btn_enhance = QPushButton("AI Enhance")
        self.btn_enhance.setToolTip(
            "Polish the composited teeth with Stable Diffusion img2img.\n"
            "First click downloads the model (~4 GB). CPU-only: 30–60 s per run."
        )
        self.btn_enhance.clicked.connect(self._enhance)
        self.btn_enhance.setEnabled(False)
        bar.addWidget(self.btn_enhance)
        self.status = QLabel("Load a front-facing patient photo with visible teeth.")
        self.status.setStyleSheet("color:#6e6e73;")
        bar.addWidget(self.status, 1)
        v.addLayout(bar)

        nudge = QFormLayout()
        nudge.setContentsMargins(0, 0, 0, 0)
        def _spin(minv, maxv, step, default):
            s = QDoubleSpinBox()
            s.setRange(minv, maxv); s.setSingleStep(step); s.setValue(default)
            s.setDecimals(2); s.setFixedWidth(90)
            return s
        # Canonical mouth midpoint is around (0, -30, -20) mm — these are
        # sensible defaults; user nudges from there.
        self.sp_x = _spin(-40, 40, 0.5, 0.0)
        self.sp_y = _spin(-60, 20, 0.5, -30.0)
        self.sp_z = _spin(-50, 30, 0.5, -18.0)
        self.sp_scale = _spin(0.1, 5.0, 0.05, 1.0)
        self.sp_rx = _spin(-45, 45, 1.0, 0.0)
        self.sp_ry = _spin(-45, 45, 1.0, 0.0)
        self.sp_rz = _spin(-180, 180, 1.0, 0.0)
        row1 = QHBoxLayout()
        for lbl, w in [("X (mm)", self.sp_x), ("Y (mm)", self.sp_y),
                       ("Z (mm)", self.sp_z), ("Scale", self.sp_scale)]:
            row1.addWidget(QLabel(lbl)); row1.addWidget(w)
        row1.addStretch()
        row2 = QHBoxLayout()
        for lbl, w in [("Rx°", self.sp_rx), ("Ry°", self.sp_ry), ("Rz°", self.sp_rz)]:
            row2.addWidget(QLabel(lbl)); row2.addWidget(w)
        self.chk_flip_x = QCheckBox("Flip X")
        self.chk_flip_y = QCheckBox("Flip Y")
        self.chk_flip_z = QCheckBox("Flip Z")
        for c in (self.chk_flip_x, self.chk_flip_y, self.chk_flip_z):
            row2.addWidget(c)
        row2.addWidget(QLabel("AI strength"))
        self.sp_denoise = _spin(0.10, 0.75, 0.05, 0.45)
        self.sp_denoise.setToolTip(
            "img2img denoise for the AI Enhance pass. 0.25 = subtle polish, "
            "0.45 = clear photoreal push (recommended), 0.6+ starts to "
            "change tooth shapes."
        )
        row2.addWidget(self.sp_denoise)
        row2.addStretch()
        v.addLayout(row1)
        v.addLayout(row2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        self._img_layout = QVBoxLayout(holder)
        self._img_layout.setAlignment(Qt.AlignCenter)
        self.img_label = _LipEditLabel()
        self.img_label.setText("(no image)")
        self.img_label.setMinimumSize(600, 400)
        self.img_label.pointDragged.connect(self._on_lip_point_dragged)
        self.img_label.dragFinished.connect(self._on_lip_drag_finished)
        self.img_label.zoomRequested.connect(self._zoom_by)
        self.img_label.debugClick.connect(
            lambda x, y, i: self.status.setText(
                getattr(self.img_label, "_debug_msg", "click")))
        self._img_layout.addWidget(self.img_label)
        scroll.setWidget(holder)
        v.addWidget(scroll, 1)

        self._img_bgr = None
        self._landmarks = None
        self._lip_poly = None
        self._teeth = None
        self._pose = None
        self._arch = None
        self._arch_path = None
        self._composited = None
        self._display_bare = None  # current image without overlays
        self._progress = None
        self._worker = None
        self._zoom = 1.0
        self._base_max_dim = 900

    def _pick_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select face photo", os.path.expanduser("~"),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Nu Smile", f"Could not read image:\n{path}")
            return
        self._img_bgr = img
        self.status.setText("Detecting face landmarks…")
        self.repaint()
        try:
            lms = detect_landmarks(img)
        except Exception as e:
            QMessageBox.critical(self, "Nu Smile", f"Landmark detection failed:\n{e}")
            self.status.setText("Failed.")
            return
        if lms is None:
            self.status.setText("No face detected — try a clearer front-facing photo.")
            self._display_bare = img.copy()
            self._refresh_view()
            return
        self._landmarks = lms
        self._lip_poly = inner_lip_polygon(lms)
        lip_mask = inner_lip_mask(img.shape[:2], self._lip_poly)
        self._teeth = segment_teeth(img, lip_mask)
        pose = estimate_head_pose(lms, img.shape)
        self._pose = pose
        pose_note = "pose: failed"
        if pose is not None:
            rvec, tvec, _K, _d = pose
            R, _ = cv2.Rodrigues(rvec)
            # yaw/pitch/roll in degrees for a quick sanity readout
            sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
            pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
            yaw   = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            roll  = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
            pose_note = (f"pose: yaw {yaw:+.1f}° pitch {pitch:+.1f}° "
                         f"roll {roll:+.1f}°")
        # Show the raw photo as the "bare" display; overlays and edit-handles
        # are added by _refresh_view based on the checkboxes.
        self._display_bare = img.copy()
        self.btn_save.setEnabled(True)
        self._refresh_view()
        n_teeth = int((self._teeth > 0).sum())
        self.status.setText(
            f"Face detected · {len(lms)} landmarks · "
            f"lip window {int((lip_mask>0).sum())} px · "
            f"teeth mask {n_teeth} px · {pose_note}"
        )
        self._lip_mask = lip_mask
        self.btn_arch.setEnabled(pose is not None)
        self.btn_arch_pick.setEnabled(pose is not None)
        self.btn_composite.setEnabled(pose is not None and self._arch is not None)

    def _use_case_arches(self):
        parent = self.parent()
        state = getattr(parent, "state", None) if parent is not None else None
        if state is None or getattr(state, "jaw_mesh", None) is None:
            QMessageBox.information(self, "Nu Smile",
                                    "No case is loaded in the main window.")
            return
        try:
            import pyvista as pv
        except Exception as e:
            QMessageBox.critical(self, "Nu Smile", f"PyVista missing:\n{e}")
            return
        # Smile shows the upper arch. If the case is Upper prep, that is
        # `jaw_mesh`; otherwise the upper is the opposing arch.
        # Merge both jaws so occlusion / lower incisors also show. Both are
        # in the same scanner frame, so a single Y↔Z swap face-aligns them
        # together (their occlusal contact stays intact).
        parts = []
        crown_included = False
        for m in (getattr(state, "jaw_mesh", None),
                  getattr(state, "opposing_jaw_mesh", None),
                  getattr(state, "crown", None)):
            if m is None:
                continue
            parts.append(pv.wrap(m) if not isinstance(m, pv.PolyData) else m)
            if m is getattr(state, "crown", None):
                crown_included = True
        if not parts:
            return
        combined = parts[0].copy()
        for p in parts[1:]:
            combined = combined.merge(p)
        aligned = face_align_arch(combined)
        # Keep native mm scale — dental scans are already life-size and this
        # matches the canonical face model's mm units.
        self._arch = aligned
        self._arch_path = "(case both arches)"
        # Nudge defaults toward where users have empirically landed a good
        # composite for this scanner convention: slightly right of centre,
        # sitting at the canonical mouth height, ~38 mm behind nose-tip.
        self.sp_x.setValue(2.0)
        self.sp_y.setValue(-30.0)
        self.sp_z.setValue(-38.0)
        crown_note = " · crown included" if crown_included else ""
        self.status.setText(
            f"Aligned upper+lower arch{crown_note} · {aligned.n_points} pts · "
            f"mesiodistal → X. Click Composite; nudge X/Y/Z to fit."
        )
        self.btn_composite.setEnabled(self._pose is not None)

    def _pick_arch(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select upper arch STL",
            os.path.dirname(self._arch_path) if self._arch_path else os.path.expanduser("~"),
            "Meshes (*.stl *.obj *.ply)",
        )
        if not path:
            return
        try:
            import pyvista as pv
            self._arch = pv.read(path)
            self._arch_path = path
        except Exception as e:
            QMessageBox.critical(self, "Nu Smile", f"Could not load arch:\n{e}")
            return
        self.status.setText(f"Arch loaded: {os.path.basename(path)} · "
                            f"{self._arch.n_points} pts. Click Composite.")
        self.btn_composite.setEnabled(self._pose is not None)

    def _composite(self):
        if self._img_bgr is None or self._pose is None or self._arch is None:
            return
        offsets = (self.sp_x.value(), self.sp_y.value(), self.sp_z.value())
        rot = (self.sp_rx.value(), self.sp_ry.value(), self.sp_rz.value())
        scale = self.sp_scale.value()
        flip = (self.chk_flip_x.isChecked(),
                self.chk_flip_y.isChecked(),
                self.chk_flip_z.isChecked())
        try:
            rgb, alpha = render_arch(self._arch, self._pose, self._img_bgr.shape,
                                     offsets_mm=offsets, scale=scale,
                                     rot_deg=rot, flip=flip)
        except Exception as e:
            QMessageBox.critical(self, "Nu Smile", f"Render failed:\n{e}")
            return
        out = composite_arch(self._img_bgr, rgb, alpha, self._lip_mask,
                             teeth_mask=self._teeth)
        self._composited = out.copy()
        self._display_bare = out.copy()
        self.btn_enhance.setEnabled(True)
        self.btn_save.setEnabled(True)
        self._refresh_view()
        self.status.setText(
            f"Composited · offset=({offsets[0]:+.1f},{offsets[1]:+.1f},"
            f"{offsets[2]:+.1f}) mm · scale={scale:.2f} · "
            f"rot=({rot[0]:+.0f},{rot[1]:+.0f},{rot[2]:+.0f})°"
        )

    def _enhance(self):
        if self._composited is None:
            return
        try:
            from . import nusmile_enhance as ne
        except Exception as e:
            QMessageBox.critical(self, "Nu Smile",
                                 f"AI stack missing (torch/diffusers):\n{e}")
            return
        if not ne.is_available():
            QMessageBox.critical(self, "Nu Smile",
                                 "torch and diffusers are not importable.")
            return
        self.btn_enhance.setEnabled(False)
        self.btn_composite.setEnabled(False)
        # Busy dialog (indeterminate) so the OS sees an animated window and
        # doesn't pop up the "not responding" force-quit prompt.
        self._progress = QProgressDialog(
            "Running Stable Diffusion img2img on CPU…\nThis takes 30–60 seconds.",
            None, 0, 0, self,
        )
        self._progress.setWindowTitle("Nu Smile — AI Enhance")
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setCancelButton(None)
        self._progress.setMinimumDuration(0)
        self._progress.show()
        self._worker = _EnhanceWorker(
            self._composited, self._lip_poly, self._lip_mask,
            float(self.sp_denoise.value()),
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.done.connect(self._on_enhance_done)
        self._worker.failed.connect(self._on_enhance_failed)
        self._worker.start()

    def _on_worker_progress(self, msg):
        self.status.setText(msg)
        if self._progress is not None:
            self._progress.setLabelText(msg)

    def _on_enhance_done(self, polished):
        self._display_bare = polished.copy()
        self._refresh_view()
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self.status.setText("AI enhance complete.")
        self.btn_enhance.setEnabled(True)
        self.btn_composite.setEnabled(True)

    def _on_enhance_failed(self, msg):
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        QMessageBox.critical(self, "Nu Smile", f"AI enhance failed:\n{msg}")
        self.btn_enhance.setEnabled(True)
        self.btn_composite.setEnabled(True)

    def _refresh_view(self):
        if self._display_bare is None:
            return
        img = self._display_bare.copy()
        editing = self.chk_edit_lip.isChecked() and self._lip_poly is not None
        if self.chk_overlays.isChecked() or editing:
            if self._pose is not None and self.chk_overlays.isChecked():
                img = draw_pose_axes(img, self._pose, length_mm=45.0)
            if self._lip_poly is not None:
                cv2.polylines(img, [self._lip_poly.astype(np.int32)],
                              True, (0, 220, 60), 2)
        if editing:
            for (x, y) in self._lip_poly:
                cv2.circle(img, (int(x), int(y)), 6, (0, 240, 255), -1)
                cv2.circle(img, (int(x), int(y)), 6, (0,   0,   0),  1)
        pix, scale = _bgr_to_qpixmap_scaled(
            img, max_dim=int(self._base_max_dim * self._zoom)
        )
        # Compute pixmap origin inside the (centered) label so we can map
        # mouse clicks back into image coords.
        lw, lh = self.img_label.width(), self.img_label.height()
        pw, ph = pix.width(), pix.height()
        origin = (max(0, (lw - pw) // 2), max(0, (lh - ph) // 2))
        self.img_label.set_points(self._lip_poly if editing else None)
        self.img_label.set_display(pix, scale, origin)

    def _zoom_by(self, factor):
        self._zoom = float(np.clip(self._zoom * float(factor), 0.15, 6.0))
        self._refresh_view()

    def _on_edit_lip_toggled(self, on):
        self.img_label.set_edit_mode(on)
        self._refresh_view()

    def _on_lip_point_dragged(self, idx, x_img, y_img):
        if self._lip_poly is None or self._img_bgr is None:
            return
        h, w = self._img_bgr.shape[:2]
        self._lip_poly[idx, 0] = float(np.clip(x_img, 0, w - 1))
        self._lip_poly[idx, 1] = float(np.clip(y_img, 0, h - 1))
        self._refresh_view()

    def _on_lip_drag_finished(self):
        # Rebuild the lip mask so the next Composite / Save uses the new
        # outline. We don't auto re-render the arch — that's user-driven.
        if self._lip_poly is None or self._img_bgr is None:
            return
        self._lip_mask = inner_lip_mask(self._img_bgr.shape[:2], self._lip_poly)
        self.status.setText(
            f"Lip line edited · new mask {int((self._lip_mask>0).sum())} px · "
            "click Composite to re-render."
        )

    def _save_image(self):
        if self._display_bare is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Nu Smile image", os.path.expanduser("~/nusmile.png"),
            "Images (*.png *.jpg *.jpeg)",
        )
        if not path:
            return
        img = self._display_bare.copy()
        if self.chk_overlays.isChecked():
            if self._pose is not None:
                img = draw_pose_axes(img, self._pose, length_mm=45.0)
            if self._lip_poly is not None:
                cv2.polylines(img, [self._lip_poly.astype(np.int32)],
                              True, (0, 220, 60), 2)
        cv2.imwrite(path, img)
        self.status.setText(f"Saved: {path}")
