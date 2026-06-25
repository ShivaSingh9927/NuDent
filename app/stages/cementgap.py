"""Stage 2 — define the cement-gap and no-cement zones on the prep cap.

The closed margin loop from stage 1 separates the prep into a "cap" (the
prepared crown-bearing surface above the margin) and a "die" (everything
below — gum, neighbouring teeth, model base). On the cap we ask two
parameters that downstream Shell generation will apply:

- **Cement gap thickness** (mm): the uniform luting space between the crown's
  inner surface and the prep across the body of the cap. Orange zone.
- **No-cement band width** (mm from margin): a narrow seating band right at
  the finish line where the crown sits hard against the prep (zero gap), so
  the margin closes cleanly. Light-blue zone.

This stage only stores the parameters and previews where each zone applies.
The actual offset is computed by ShellStage.
"""
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox, QMessageBox,
    QSlider,
)

from ..config import STAGES
from ..ui import section_label
from ..border_diagram import BorderProfileDiagram
from ..border_geometry import compute_border_profile_2d, build_border_band
from .base import Stage


# Zone colors (RGB 0–255). These match the user's reference: orange for the
# cement-gap zone, light blue for the seating band along the margin.
COLOR_CEMENT_GAP = np.array([255, 149, 0], dtype=np.uint8)    # orange
COLOR_NO_CEMENT  = np.array([137, 207, 240], dtype=np.uint8)  # light blue


class CementGapStage(Stage):
    name = "Cement"
    description = STAGES[1][1]

    def __init__(self, app):
        super().__init__(app)
        self._cap_actor = None
        self._dim_prep_actor = None  # faint full-prep underneath for context
        self._margin_actor = None
        self._border_actor = None    # swept crown-border band
        self._border_sliders = {}    # key -> (slider, scale)
        self._suppress = False        # guards spin-box signal cycles during restore

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # --- CEMENT GAP ---
        layout.addWidget(section_label("CEMENT GAP"))
        hint = QLabel(
            "Orange zone: uniform luting space between the crown and the prep. "
            "Light-blue zone: seating band along the margin (no cement)."
        )
        hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 4px 0;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        gap_row = QHBoxLayout()
        gap_row.addWidget(QLabel("Cement gap thickness (mm):"))
        self.spin_gap = QDoubleSpinBox()
        self.spin_gap.setRange(0.0, 1.0)
        self.spin_gap.setSingleStep(0.01)
        self.spin_gap.setDecimals(3)
        self.spin_gap.setValue(self.app.state.cement_gap_thickness)
        self.spin_gap.valueChanged.connect(self._on_gap_changed)
        gap_row.addWidget(self.spin_gap)
        layout.addLayout(gap_row)

        # --- BORDER ---
        layout.addWidget(section_label("BORDER (NO-CEMENT BAND)"))
        band_row = QHBoxLayout()
        band_row.addWidget(QLabel("From margin (mm):"))
        self.spin_band = QDoubleSpinBox()
        self.spin_band.setRange(0.0, 5.0)
        self.spin_band.setSingleStep(0.1)
        self.spin_band.setDecimals(2)
        self.spin_band.setValue(self.app.state.no_cement_band_width)
        self.spin_band.valueChanged.connect(self._on_band_changed)
        band_row.addWidget(self.spin_band)
        layout.addLayout(band_row)

        nc_row = QHBoxLayout()
        nc_row.addWidget(QLabel("No-cement thickness (mm):"))
        self.spin_no_cement = QDoubleSpinBox()
        self.spin_no_cement.setRange(0.0, 1.0)
        self.spin_no_cement.setSingleStep(0.01)
        self.spin_no_cement.setDecimals(3)
        self.spin_no_cement.setValue(self.app.state.no_cement_thickness)
        self.spin_no_cement.valueChanged.connect(self._on_no_cement_changed)
        nc_row.addWidget(self.spin_no_cement)
        layout.addLayout(nc_row)

        # --- CROWN BORDER (Crown Bottoms) ---
        layout.addWidget(section_label("CROWN BORDER"))
        border_hint = QLabel(
            "Sweep an Exocad-style border profile along the margin loop. "
            "Numbers match the diagram below."
        )
        border_hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 4px 0;")
        border_hint.setWordWrap(True)
        layout.addWidget(border_hint)

        # (key, label, min, max, default, scale, suffix). Slider values are
        # ints; the real value = slider_value / scale.
        for key, label, mn, mx, default, scale, suffix in [
            ("horizontal",   "1. Horizontal",   0, 200, 20, 100, "mm"),
            ("angled",       "2. Angled",       0, 200,  0, 100, "mm"),
            ("angle_deg",    "3. Angle",        0,  90, 45,   1, "°"),
            ("vertical",     "4. Vertical",     0, 200,  0, 100, "mm"),
            ("below_margin", "5. Below margin", 0, 200,  0, 100, "mm"),
        ]:
            self._add_border_slider(layout, key, label, mn, mx, default, scale, suffix)

        self.border_diagram = BorderProfileDiagram()
        layout.addWidget(self.border_diagram)

        legend = QLabel(
            "1 Horizontal · 2 Angled · 3 Angle · 4 Vertical · 5 Below margin")
        legend.setWordWrap(True)
        legend.setStyleSheet("color: #86868b; font-size: 10px; padding: 2px 0;")
        layout.addWidget(legend)

        # --- STATUS ---
        self.status = QLabel("Mark and close the margin loop first.")
        self.status.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 4px 0;")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        # --- TOOLS ---
        layout.addWidget(section_label("TOOLS"))
        self.btn_recompute = QPushButton("Recompute Cap")
        self.btn_recompute.clicked.connect(self._recompute)
        layout.addWidget(self.btn_recompute)

        layout.addStretch()

        # Push the slider defaults into the diagram + app.state so everything
        # is consistent before the first margin is drawn.
        self._apply_border_params(redraw=False)

    # ----- Stage lifecycle -----

    def is_complete(self):
        return self.app.state.cap_mesh is not None and self.app.state.cap_zone_labels is not None

    def on_enter(self):
        if not self.app.state.margin_loop_closed or not self.app.state.margin_points:
            self.status.setText(
                "Margin loop must be closed before defining cement zones. "
                "Go back to step 1."
            )
            self._clear_actors()
            return
        # If the cap was wiped (e.g. case reload, undo back through margin),
        # don't show the previous case's stats — reset the status line.
        if self.app.state.cap_mesh is None:
            self.status.setText("Computing prep cap...")
            self._recompute()
        else:
            self._redraw()
            self._update_status()
        self.app.set_status(self.description)

    def on_exit(self):
        self._clear_actors()

    # ----- Cap selection + zone labelling -----

    def _recompute(self):
        """Build the cap sub-mesh from prep_mesh by cutting at the margin's
        plane, then label every cap vertex as cement-gap or no-cement based on
        Euclidean distance to the margin polyline."""
        prep = self.app.state.prep_mesh
        if prep is None:
            QMessageBox.information(
                self, "No prep",
                "Open a case, mark and close the margin loop first."
            )
            return
        margin = np.asarray(self.app.state.margin_points)
        if len(margin) < 3:
            QMessageBox.information(self, "No margin", "Margin loop not yet closed.")
            return

        # The margin loop is a closed fence on the mesh. The cap is just the
        # set of mesh vertices reachable from a known cap-side point without
        # crossing the fence. No axis math, no "above" guess.
        cap_seed = self.app.state.cap_seed_point
        if cap_seed is None:
            QMessageBox.warning(
                self, "No cap seed",
                "No cap-side seed point recorded. Re-mark the margin so the "
                "first click is captured as the cap-side reference."
            )
            return
        cap_vmask = self._cap_by_fence_flood(prep, margin, cap_seed)
        if cap_vmask is None or not cap_vmask.any():
            QMessageBox.warning(self, "Cap empty",
                                "Could not flood a cap region from the margin.")
            return

        faces = np.asarray(prep.faces).reshape(-1, 4)[:, 1:]
        face_keep = cap_vmask[faces].all(axis=1)
        if not face_keep.any():
            QMessageBox.warning(self, "Cap empty",
                                "No triangles inside the flooded cap region.")
            return
        cap = prep.extract_cells(np.where(face_keep)[0]).extract_surface()
        if cap.n_points == 0:
            QMessageBox.warning(self, "Cap empty", "Cap extraction produced no points.")
            return

        labels = self._compute_labels(cap, margin, self.spin_band.value())

        self.app.state.cap_mesh = cap
        self.app.state.cap_zone_labels = labels
        self.app.state.cement_gap_thickness = float(self.spin_gap.value())
        self.app.state.no_cement_band_width = float(self.spin_band.value())
        self.app.state.no_cement_thickness = float(self.spin_no_cement.value())

        self._redraw()
        self.completion_changed.emit()
        n_no = int((labels == 1).sum())
        n_gap = int((labels == 0).sum())
        self.status.setText(
            f"Cap: {cap.n_points} verts — {n_gap} cement-gap, {n_no} no-cement."
        )

    def _cap_by_fence_flood(self, prep, margin_pts, cap_seed,
                            sample_spacing=0.05):
        """BFS the prep mesh from `cap_seed`, blocked by an airtight fence
        built from the margin loop.

        Fence construction:
          1. Densely resample the closed margin polyline at ~0.05 mm spacing.
          2. Snap every dense sample to its nearest mesh vertex.
          3. Expand by one 1-ring to guarantee at least two mesh layers of
             sealing (handles meshes whose edge length exceeds the sample
             spacing without leaking).
        Result is the connected cap region bounded by the margin — no axis
        assumptions, no "above" guess. The fence vertices themselves are
        included in the returned cap mask so the no-cement band lies flush
        against the margin instead of sitting on a stripped border."""
        pts = np.asarray(prep.points)
        n = len(pts)
        margin = np.asarray(margin_pts)

        # 1. Densify the closed loop so samples are ~sample_spacing apart.
        loop = np.vstack([margin, margin[0:1]])
        seg_vecs = loop[1:] - loop[:-1]
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        total = float(seg_lens.sum())
        if total <= 0:
            return None
        n_samples = max(64, int(np.ceil(total / sample_spacing)))
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        ts = np.linspace(0.0, total, n_samples, endpoint=False)
        seg_idx = np.clip(np.searchsorted(cum, ts, side="right") - 1, 0, len(seg_lens) - 1)
        local = (ts - cum[seg_idx]) / np.maximum(seg_lens[seg_idx], 1e-9)
        dense = loop[seg_idx] + local[:, None] * seg_vecs[seg_idx]

        # 2. Snap each sample to the nearest mesh vertex.
        tree = cKDTree(pts)
        _, snap = tree.query(dense, k=1)
        fence = np.zeros(n, dtype=bool)
        fence[np.asarray(snap, dtype=int)] = True

        # Build vertex adjacency from triangle faces.
        faces = np.asarray(prep.faces).reshape(-1, 4)[:, 1:]
        adj = [[] for _ in range(n)]
        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            adj[a].append(b); adj[a].append(c)
            adj[b].append(a); adj[b].append(c)
            adj[c].append(a); adj[c].append(b)

        # 3. Expand the fence by one 1-ring on the margin-snap layer — that
        # second layer is what guarantees topological closure even when mesh
        # edge length exceeds the polyline sample spacing.
        snap_set = np.where(fence)[0]
        for v in snap_set:
            for w in adj[v]:
                fence[w] = True

        # Snap the cap-side seed; nudge inward if it landed on the fence.
        _, seed_vid = tree.query(np.asarray(cap_seed), k=1)
        seed_vid = int(seed_vid)
        if fence[seed_vid]:
            _, near = tree.query(np.asarray(cap_seed), k=128)
            for i in np.atleast_1d(near):
                if not fence[int(i)]:
                    seed_vid = int(i)
                    break
            else:
                return None

        # 4. Plain BFS from cap seed, blocked by fence vertices.
        visited = np.zeros(n, dtype=bool)
        visited[seed_vid] = True
        frontier = [seed_vid]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if visited[v] or fence[v]:
                        continue
                    visited[v] = True
                    nxt.append(v)
            frontier = nxt

        # 5. Re-attach every fence vertex that has at least one cap-side
        # neighbour. Including the whole margin layer in the cap is what
        # makes the no-cement band sit flush against the red margin line —
        # die-side fence neighbours are still excluded because their
        # adjacent triangles bridge to non-visited vertices and get
        # dropped by the face_keep test downstream.
        for v in np.where(fence)[0]:
            for w in adj[v]:
                if visited[w]:
                    visited[v] = True
                    break
        # Snap-include the original margin-snap vertices unconditionally —
        # those are the very loop the user drew, and they belong on the cap
        # boundary regardless of any local 1-ring quirks.
        for s in snap_set:
            visited[int(s)] = True
        return visited

    def _keep_margin_component(self, cap_mesh, margin_pts):
        """Run connectivity on the post-cut cap and keep only the component
        whose vertices come closest to the margin loop. Filters out unrelated
        jaw fragments, opposing-arch blobs, neighbouring teeth, etc."""
        conn = cap_mesh.connectivity()
        rid = np.asarray(conn.point_data.get("RegionId"))
        if rid is None or len(rid) == 0:
            return cap_mesh  # nothing labelled — keep as-is
        pts = np.asarray(conn.points)
        tree = cKDTree(pts)
        # For each margin point, find its nearest cap-vertex's region.
        _, near = tree.query(np.asarray(margin_pts), k=1)
        region_votes = rid[near]
        # Pick the region the margin touches most often.
        unique, counts = np.unique(region_votes, return_counts=True)
        best_region = int(unique[int(np.argmax(counts))])
        keep_v = rid == best_region
        faces = np.asarray(conn.faces).reshape(-1, 4)[:, 1:]
        face_keep = keep_v[faces].all(axis=1)
        if not face_keep.any():
            return None
        return conn.extract_cells(np.where(face_keep)[0]).extract_surface()

    def _compute_labels(self, cap_mesh, margin_pts, band_width):
        """Per-vertex zone labels: 0 = cement gap, 1 = no-cement (within `band_width`
        Euclidean mm of any margin point)."""
        tree = cKDTree(np.asarray(margin_pts))
        d, _ = tree.query(np.asarray(cap_mesh.points), k=1)
        return (d < float(band_width)).astype(np.int32)

    # ----- Live-update handlers -----

    def _on_gap_changed(self, v):
        if self._suppress:
            return
        self.app.state.cement_gap_thickness = float(v)
        # Thickness changes the height of the cement layer — rebuild it.
        if self.app.state.cap_mesh is not None:
            self._redraw()
        self._update_status()

    def _on_no_cement_changed(self, v):
        if self._suppress:
            return
        self.app.state.no_cement_thickness = float(v)
        if self.app.state.cap_mesh is not None:
            self._redraw()
        self._update_status()

    def _on_band_changed(self, v):
        if self._suppress:
            return
        if self.app.state.cap_mesh is None:
            self.app.state.no_cement_band_width = float(v)
            return
        # Re-label the existing cap with the new band width and redraw.
        labels = self._compute_labels(
            self.app.state.cap_mesh, np.asarray(self.app.state.margin_points), v
        )
        self.app.state.cap_zone_labels = labels
        self.app.state.no_cement_band_width = float(v)
        self._redraw()
        self._update_status()

    def _update_status(self):
        cap = self.app.state.cap_mesh
        labels = self.app.state.cap_zone_labels
        if cap is None or labels is None:
            return
        n_no = int((labels == 1).sum())
        n_gap = int((labels == 0).sum())
        self.status.setText(
            f"Cap: {cap.n_points} verts — {n_gap} cement-gap, {n_no} no-cement. "
            f"Gap={self.spin_gap.value():.3f} mm, band={self.spin_band.value():.2f} mm."
        )

    # ----- Layer construction -----

    # Width over which the offset smoothly ramps from 0 (at the no-cement
    # band edge) to the full cement gap. Wider than ShellStage's 0.3 mm
    # because here it also controls the visible blue→orange colour blend.
    RAMP_WIDTH_MM = 0.5

    def _build_cement_layer(self, cap_mesh, labels):
        """Lift the cap into a real offset layer: vertices in the cement-gap
        zone rise along their normals by `cement_gap_thickness`, vertices in
        the no-cement band stay on the prep, with a smoothstep ramp in
        between. Returns (layer_polydata, blend_t) where blend_t is a
        per-vertex value in [0, 1] used to interpolate colours so the
        blue→orange boundary doesn't show a hard step."""
        gap = float(self.spin_gap.value())
        band = float(self.spin_band.value())
        no_cement = float(self.spin_no_cement.value())
        ramp = self.RAMP_WIDTH_MM

        layer = cap_mesh.copy()
        # Use the prep mesh's vertex normals (it's closed, so auto_orient
        # reliably points outward) and copy them onto cap vertices by
        # nearest-neighbour. Computing normals directly on the cap can
        # mis-orient because it's an open patch and the heuristic can flip.
        try:
            prep = self.app.state.prep_mesh
            prep_n = prep.compute_normals(
                point_normals=True, cell_normals=False,
                auto_orient_normals=True, inplace=False,
            )
            prep_normals = np.asarray(prep_n["Normals"])
            tree = cKDTree(np.asarray(prep.points))
            _, near = tree.query(np.asarray(cap_mesh.points), k=1)
            normals = prep_normals[np.asarray(near, dtype=int)]
        except Exception:
            normals = np.zeros((cap_mesh.n_points, 3), dtype=float)

        # Per-vertex offset: `no_cement` inside the band, ramping smoothly
        # to `gap` past the band edge. Smoothstep (3t² − 2t³) keeps the
        # surface's derivative continuous at both ends so there's no crease.
        # A small VISUAL_LIFT guarantees the layer never coplanes with the
        # prep mesh — eliminates z-fight regardless of the user's values.
        VISUAL_LIFT = 0.03
        margin = np.asarray(self.app.state.margin_points)
        tree = cKDTree(margin)
        d, _ = tree.query(np.asarray(cap_mesh.points), k=1)
        t_raw = np.clip((d - band) / max(ramp, 1e-6), 0.0, 1.0)
        t = t_raw * t_raw * (3.0 - 2.0 * t_raw)  # smoothstep
        # Lerp uniformly between the two zone thicknesses; both lift as a
        # solid slab in their own region, with a soft join across the ramp.
        offset_amt = no_cement * (1.0 - t) + gap * t
        offset_amt = np.maximum(offset_amt, VISUAL_LIFT)

        layer.points = np.asarray(cap_mesh.points) + normals * offset_amt[:, None]
        return layer, t

    # ----- Visualization -----

    def _clear_actors(self):
        for a in (self._cap_actor, self._dim_prep_actor, self._margin_actor,
                  self._border_actor):
            if a is not None:
                try: self.app.plotter.remove_actor(a)
                except Exception: pass
        self._cap_actor = None
        self._dim_prep_actor = None
        self._margin_actor = None
        self._border_actor = None

    def _redraw(self):
        self._clear_actors()
        cap = self.app.state.cap_mesh
        labels = self.app.state.cap_zone_labels
        if cap is None or labels is None:
            return

        # Don't add our own prep actor here — the main window's `jaw_actor`
        # is already rendering the prep jaw with the correct realistic-colors
        # state and opacity. Adding a second actor at the same position
        # masked the realistic colors with plain white.

        # Build the cement-gap layer and a smooth blend factor: 0 at the
        # margin (blue), 1 in the body (orange). Smoothstep interpolation on
        # both the offset and the colour eliminates the hard boundary line.
        layer, blend_t = self._build_cement_layer(cap, labels)
        blue = COLOR_NO_CEMENT.astype(np.float32)
        orange = COLOR_CEMENT_GAP.astype(np.float32)
        colors = (blue[None, :] * (1.0 - blend_t[:, None]) +
                  orange[None, :] * blend_t[:, None])
        layer["zone_rgb"] = np.clip(colors, 0, 255).astype(np.uint8)
        self._cap_actor = self.app.plotter.add_mesh(
            layer, scalars="zone_rgb", rgb=True,
            pickable=False, reset_camera=False,
        )

        # Margin loop overlay as a thin red tube.
        margin = np.asarray(self.app.state.margin_points)
        if len(margin) >= 2:
            arr = np.vstack([margin, margin[0]]) if self.app.state.margin_loop_closed else margin
            n = len(arr)
            poly = pv.PolyData(arr)
            poly.lines = np.hstack([[n], np.arange(n)])
            tube = poly.tube(radius=0.08)
            self._margin_actor = self.app.plotter.add_mesh(
                tube, color="red", pickable=False, reset_camera=False,
            )
        self._build_border_actor()
        self.app.plotter.render()

    # ----- Crown border (Crown Bottoms) -----

    def _fmt_border_val(self, raw, scale, suffix):
        if suffix == "°":
            return f"{int(raw)}{suffix}"
        return f"{raw / scale:.2f}{suffix}"

    def _add_border_slider(self, layout, key, label, mn, mx, default, scale, suffix):
        row = QVBoxLayout()
        row.setSpacing(2)

        lr = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #424245; font-size: 12px;")
        val_lbl = QLabel(self._fmt_border_val(default, scale, suffix))
        val_lbl.setStyleSheet("color: #1d1d1f; font-size: 12px; font-weight: 600;")
        val_lbl.setAlignment(Qt.AlignRight)
        lr.addWidget(lbl)
        lr.addStretch()
        lr.addWidget(val_lbl)
        row.addLayout(lr)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(mn)
        slider.setMaximum(mx)
        slider.setValue(default)
        slider.valueChanged.connect(
            lambda v, k=key, vl=val_lbl, s=scale, sx=suffix:
                self._on_border_slider_move(k, v, vl, s, sx))
        slider.sliderReleased.connect(lambda: self._apply_border_params(redraw=True))
        row.addWidget(slider)

        self._border_sliders[key] = (slider, scale)
        layout.addLayout(row)

    def _on_border_slider_move(self, key, raw, val_lbl, scale, suffix):
        """On every tick: update the label and the live diagram (cheap). The
        3D band is rebuilt only on slider release (see _apply_border_params)."""
        val_lbl.setText(self._fmt_border_val(raw, scale, suffix))
        diagram = getattr(self, "border_diagram", None)
        if diagram is not None:
            p = self._get_border_params()
            diagram.set_params(**p)

    def _get_border_params(self):
        return {key: slider.value() / scale
                for key, (slider, scale) in self._border_sliders.items()}

    def _apply_border_params(self, redraw=True):
        """Copy the slider values into app.state, sync the diagram, and
        (optionally) rebuild the 3D band actor."""
        if self._suppress:
            return
        p = self._get_border_params()
        st = self.app.state
        st.border_horizontal = p["horizontal"]
        st.border_angled = p["angled"]
        st.border_angle_deg = p["angle_deg"]
        st.border_vertical = p["vertical"]
        st.border_below_margin = p["below_margin"]

        diagram = getattr(self, "border_diagram", None)
        if diagram is not None:
            diagram.set_params(**p)

        if redraw:
            self._build_border_actor()
            self.app.plotter.render()

    def _build_border_actor(self):
        """Sweep the current border profile along the closed margin loop and
        (re)add it to the plotter. No-op until a closed margin loop exists."""
        if self._border_actor is not None:
            try: self.app.plotter.remove_actor(self._border_actor)
            except Exception: pass
            self._border_actor = None

        st = self.app.state
        if not st.margin_loop_closed or len(st.margin_points) < 3:
            return

        profile = compute_border_profile_2d(
            horizontal=st.border_horizontal,
            angled=st.border_angled,
            angle_deg=st.border_angle_deg,
            vertical=st.border_vertical,
            below_margin=st.border_below_margin,
        )
        margin = [np.asarray(p, dtype=float) for p in st.margin_points]
        band = build_border_band(margin, profile, closed=True)
        if not band["ok"]:
            return

        verts = np.asarray(band["verts"], dtype=float)
        tri = np.asarray(band["faces"], dtype=np.int64)
        faces = np.hstack([np.full((len(tri), 1), 3, dtype=np.int64), tri]).ravel()
        mesh = pv.PolyData(verts, faces)
        self._border_actor = self.app.plotter.add_mesh(
            mesh, color="#2dd4bf", opacity=0.65, show_edges=False,
            pickable=False, reset_camera=False,
        )

    # ----- Persistence -----

    def serialize(self):
        return {
            "cement_gap_thickness": float(self.app.state.cement_gap_thickness),
            "no_cement_band_width": float(self.app.state.no_cement_band_width),
            "no_cement_thickness": float(self.app.state.no_cement_thickness),
            "border_horizontal": float(self.app.state.border_horizontal),
            "border_angled": float(self.app.state.border_angled),
            "border_angle_deg": float(self.app.state.border_angle_deg),
            "border_vertical": float(self.app.state.border_vertical),
            "border_below_margin": float(self.app.state.border_below_margin),
        }

    def restore(self, data):
        self._suppress = True
        self.spin_gap.setValue(float(data.get("cement_gap_thickness", 0.08)))
        self.spin_band.setValue(float(data.get("no_cement_band_width", 1.0)))
        self.spin_no_cement.setValue(float(data.get("no_cement_thickness", 0.0)))

        border_defaults = {
            "border_horizontal": 0.2, "border_angled": 0.0,
            "border_angle_deg": 45.0, "border_vertical": 0.0,
            "border_below_margin": 0.0,
        }
        slider_keys = {
            "horizontal": "border_horizontal", "angled": "border_angled",
            "angle_deg": "border_angle_deg", "vertical": "border_vertical",
            "below_margin": "border_below_margin",
        }
        for skey, dkey in slider_keys.items():
            slider, scale = self._border_sliders[skey]
            val = float(data.get(dkey, border_defaults[dkey]))
            slider.setValue(int(round(val * scale)))

        self._suppress = False
        self.app.state.cement_gap_thickness = float(self.spin_gap.value())
        self.app.state.no_cement_band_width = float(self.spin_band.value())
        self.app.state.no_cement_thickness = float(self.spin_no_cement.value())
        self._apply_border_params(redraw=False)
        # Cap mesh isn't persisted — it's derived from prep + margin. Recompute
        # on next on_enter().
        self.app.state.cap_mesh = None
        self.app.state.cap_zone_labels = None
