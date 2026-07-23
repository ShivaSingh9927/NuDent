"""Stage 3 — offset the placed crown inward to build the inner shell.

The inner-surface offset is per-vertex:
- Right at the margin (within `margin_offset` mm) the inner sits hard against
  the outer so the crown seats precisely on the prep finish line (zero gap).
- Past `margin_offset` it ramps up to the full cement gap so a uniform luting
  channel exists across the body of the crown.

This mirrors the exocad / 3Shape "cement gap + distance to margin" pair.
"""
import numpy as np
import vtk
from scipy.spatial import cKDTree
from PyQt5.QtWidgets import (
    QVBoxLayout, QPushButton, QLabel, QMessageBox, QComboBox,
    QSpinBox, QDoubleSpinBox,
)

from ..config import STAGES
from ..ui import section_label
from .base import Stage
from .place import PlaceStage


class ShellStage(Stage):
    name = "Shell"
    description = STAGES[4][1]

    # (Material name, default wall thickness in microns)
    MATERIALS = [
        ("Zirconia (monolithic)",        800),
        ("Lithium Disilicate (e.max)",   500),
        ("PFM (porcelain-fused-to-metal)", 1000),
        ("Gold",                        1500),
        ("Custom",                      None),
    ]

    # Width over which the cement gap ramps from 0 (at margin_offset) to its
    # full value. A short, smooth blend avoids a visible step in the inner
    # surface where the gap kicks in.
    RAMP_WIDTH_MM = 0.3

    def __init__(self, app):
        super().__init__(app)
        self.inner_actor = None
        self._suppress = False  # guard against material<->thickness signal cycles

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # --- MATERIAL ---
        layout.addWidget(section_label("MATERIAL"))
        self.material_box = QComboBox()
        for name, t in self.MATERIALS:
            self.material_box.addItem(f"{name} ({t} μm)" if t else name)
        self.material_box.currentIndexChanged.connect(self._on_material_change)
        layout.addWidget(self.material_box)

        # --- THICKNESS ---
        layout.addWidget(section_label("WALL THICKNESS (μm)"))
        self.thickness_spin = QSpinBox()
        self.thickness_spin.setRange(100, 3000)
        self.thickness_spin.setSingleStep(50)
        self.thickness_spin.setValue(self.MATERIALS[0][1])
        self.thickness_spin.valueChanged.connect(self._on_thickness_change)
        layout.addWidget(self.thickness_spin)

        # --- CEMENT GAP ---
        layout.addWidget(section_label("CEMENT GAP (μm)"))
        gap_hint = QLabel(
            "Luting channel between inner crown and prep, away from the margin."
        )
        gap_hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 2px 0;")
        gap_hint.setWordWrap(True)
        layout.addWidget(gap_hint)
        self.gap_spin = QSpinBox()
        self.gap_spin.setRange(0, 150)
        self.gap_spin.setSingleStep(5)
        self.gap_spin.setValue(40)
        layout.addWidget(self.gap_spin)

        # --- MIN WALL THICKNESS ---
        layout.addWidget(section_label("MIN WALL THICKNESS (mm)"))
        min_hint = QLabel(
            "Guaranteed floor for the zirconia wall. Wherever the inner surface "
            "gets closer than this to the outer, the outer is bulged out locally."
        )
        min_hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 2px 0;")
        min_hint.setWordWrap(True)
        layout.addWidget(min_hint)
        self.min_thickness_spin = QDoubleSpinBox()
        self.min_thickness_spin.setRange(0.0, 2.0)
        self.min_thickness_spin.setSingleStep(0.05)
        self.min_thickness_spin.setDecimals(2)
        self.min_thickness_spin.setValue(0.5)
        self.min_thickness_spin.setSuffix(" mm")
        layout.addWidget(self.min_thickness_spin)

        # --- MARGIN OFFSET ---
        layout.addWidget(section_label("DISTANCE TO MARGIN (mm)"))
        margin_hint = QLabel(
            "Width of the zero-gap zone above the margin so the crown seats "
            "tightly on the finish line."
        )
        margin_hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 2px 0;")
        margin_hint.setWordWrap(True)
        layout.addWidget(margin_hint)
        self.margin_offset_spin = QDoubleSpinBox()
        self.margin_offset_spin.setRange(0.0, 2.0)
        self.margin_offset_spin.setSingleStep(0.1)
        self.margin_offset_spin.setDecimals(2)
        self.margin_offset_spin.setValue(1.0)
        self.margin_offset_spin.setSuffix(" mm")
        layout.addWidget(self.margin_offset_spin)


        # --- GENERATE ---
        self.btn_generate = QPushButton("Generate Shell")
        self.btn_generate.setObjectName("primary")
        self.btn_generate.clicked.connect(self.generate_shell)
        layout.addWidget(self.btn_generate)

        # --- STATUS ---
        layout.addWidget(section_label("STATUS"))
        self.lbl_status = QLabel("Shell not yet generated.")
        self.lbl_status.setStyleSheet("color: #6e6e73; font-size: 12px;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        self.btn_show_inner = QPushButton("Hide Inner Surface")
        self.btn_show_inner.setCheckable(True)
        self.btn_show_inner.setChecked(True)
        self.btn_show_inner.clicked.connect(self._toggle_inner)
        self.btn_show_inner.setEnabled(False)
        layout.addWidget(self.btn_show_inner)

        self.btn_show_outer = QPushButton("Hide Outer Crown")
        self.btn_show_outer.setCheckable(True)
        self.btn_show_outer.setChecked(True)
        self.btn_show_outer.clicked.connect(self._toggle_outer)
        self.btn_show_outer.setEnabled(False)
        layout.addWidget(self.btn_show_outer)

        layout.addStretch()

    # --- Stage lifecycle ---

    def is_complete(self):
        return self.app.state.shell_inner is not None

    def on_enter(self):
        if self.app.state.shell_inner is not None:
            # Returning from a later stage (Trim/Refine) — restore the shell
            # view: outer ghosted, inner visible, any trimmed/final actor
            # owned by downstream stages hidden so they don't mask the shell.
            self._show_shell_view()
            self.app.set_status(
                "Shell exists. Click 'Generate Shell' again if you've changed the crown."
            )
        else:
            self.app.set_status(self.description)

    def _show_shell_view(self):
        """Restore the canonical Shell-stage scene composition."""
        place = self._place_stage()
        if place is not None:
            place.set_outer_visible(True)
            place.set_outer_opacity(0.35)
        if self.inner_actor is not None:
            self.inner_actor.SetVisibility(True)
            self.btn_show_inner.setChecked(True)
            self.btn_show_inner.setText("Hide Inner Surface")
            self.btn_show_outer.setChecked(True)
            self.btn_show_outer.setText("Hide Outer Crown")
        # Hide downstream actors so the shell is the focal layer.
        for s in self.app.stages:
            if hasattr(s, "trimmed_actor") and s.trimmed_actor is not None:
                s.trimmed_actor.SetVisibility(False)
            if hasattr(s, "final_actor") and s.final_actor is not None:
                s.final_actor.SetVisibility(False)
        self.app.plotter.render()

    def reset_shell(self):
        if self.inner_actor is not None:
            try: self.app.plotter.remove_actor(self.inner_actor)
            except Exception: pass
        self.inner_actor = None
        self.app.state.shell_inner = None
        self.app.state.shell_outer = None
        # Restore outer crown to fully opaque + visible
        place = self._place_stage()
        if place is not None:
            place.set_outer_opacity(1.0)
            place.set_outer_visible(True)
        self.lbl_status.setText("Shell not yet generated.")
        self.btn_show_inner.setEnabled(False)
        self.btn_show_outer.setEnabled(False)
        self.btn_show_outer.setChecked(True)
        self.btn_show_outer.setText("Hide Outer Crown")
        self.completion_changed.emit()

    def _place_stage(self):
        for s in self.app.stages:
            if isinstance(s, PlaceStage):
                return s
        return None

    # --- Material / thickness sync ---

    def _on_material_change(self, idx):
        if self._suppress: return
        _name, thickness = self.MATERIALS[idx]
        if thickness is None:
            return  # "Custom" — leave the spinner alone
        self._suppress = True
        self.thickness_spin.setValue(thickness)
        self._suppress = False

    def _on_thickness_change(self, value):
        if self._suppress: return
        match = next(
            (i for i, (_, t) in enumerate(self.MATERIALS) if t == value),
            None,
        )
        custom_idx = next(i for i, (n, _) in enumerate(self.MATERIALS) if n == "Custom")
        target = match if match is not None else custom_idx
        if self.material_box.currentIndex() != target:
            self._suppress = True
            self.material_box.setCurrentIndex(target)
            self._suppress = False

    # --- Shell generation ---

    def generate_shell(self):
        crown = self.app.state.crown
        if crown is None:
            QMessageBox.warning(self, "No crown", "Place a crown preset first.")
            return

        cap = self.app.state.cap_mesh
        if cap is None:
            QMessageBox.warning(
                self, "No cap surface",
                "The inner surface now comes from stage 2 (Cement). Go back "
                "and click 'Recompute Cap' there before generating the shell."
            )
            return

        # Invalidate any downstream trim that was based on the old shell
        self.app.notify_shell_changed()

        if self.inner_actor is not None:
            try: self.app.plotter.remove_actor(self.inner_actor)
            except Exception: pass
            self.inner_actor = None

        thickness_um = self.thickness_spin.value()

        # Inner surface = the prep cap from stage 2, offset outward by the
        # cement-gap field. Per-vertex offset: `no_cement` inside the seating
        # band, ramping smoothly (smoothstep) to `cement_gap` past the band
        # edge. Orange zone = void for luting; blue zone = tight seat on prep.
        inner = self._build_inner_from_cap(cap)
        n_clamped = 0
        n_warn = 0

        # Reverse winding so inner-surface normals point into the crown cavity.
        if hasattr(inner, 'flip_faces'):
            inner.flip_faces(inplace=True)
        else:
            inner.flip_normals()

        # Enforce minimum wall thickness by bulging the OUTER crown outward
        # wherever the inner surface would otherwise poke through (or come
        # closer than the min). Inner stays glued to the prep for seating fit.
        min_thk = float(self.min_thickness_spin.value())
        n_bulged = 0
        if min_thk > 0.0:
            n_bulged = self._bulge_outer_for_thickness(crown, inner, min_thk)
            if n_bulged > 0:
                # The placed outer actor shares crown.points — mark it dirty
                # so the viewport picks up the new geometry.
                try: crown.GetPoints().Modified()
                except Exception: pass

        self.app.state.shell_outer = crown
        self.app.state.shell_inner = inner

        # Ghost the outer (gold) so the inner (blue) is visible through it.
        place = self._place_stage()
        if place is not None:
            place.set_outer_opacity(0.35)

        self.inner_actor = self.app.plotter.add_mesh(
            inner, color="#1f6feb", opacity=1.0, show_edges=False
        )
        self.app.plotter.render()

        st = self.app.state
        msg = (
            f"Shell generated · {thickness_um} μm wall · "
            f"cement gap {st.cement_gap_thickness*1000:.0f} μm, "
            f"no-cement {st.no_cement_thickness*1000:.0f} μm "
            f"(band {st.no_cement_band_width:.2f} mm)\n"
            f"Inner surface: {inner.n_points:,} verts, {inner.n_cells:,} faces"
        )
        if min_thk > 0.0:
            msg += (f"\nMin wall {min_thk:.2f} mm enforced — "
                    f"{n_bulged:,} outer verts bulged out.")
        self.lbl_status.setText(msg)

        self.btn_show_inner.setEnabled(True)
        self.btn_show_inner.setChecked(True)
        self.btn_show_inner.setText("Hide Inner Surface")
        self.btn_show_outer.setEnabled(True)
        self.btn_show_outer.setChecked(True)
        self.btn_show_outer.setText("Hide Outer Crown")
        self.completion_changed.emit()
        self.app.set_status(f"Shell generated with {thickness_um} μm wall thickness.")

    def _build_inner_from_cap(self, cap):
        """Lift the prep cap outward by the cement-gap field to produce the
        crown's inner surface. Per-vertex offset = no_cement inside the seating
        band, ramping smoothly to cement_gap past the band edge. Normals come
        from the prep mesh (closed, reliable auto-orient) and are copied to
        cap vertices via nearest-neighbour — copying mirrors the visualiser
        in stage 2 so the geometry matches what the user previewed."""
        st = self.app.state
        gap = float(st.cement_gap_thickness)
        no_cement = float(st.no_cement_thickness)
        band = float(st.no_cement_band_width)
        ramp = 0.5  # same RAMP_WIDTH_MM as CementGapStage

        prep = st.prep_mesh
        prep_n = prep.compute_normals(
            point_normals=True, cell_normals=False,
            auto_orient_normals=True, inplace=False,
        )
        prep_normals = np.asarray(prep_n["Normals"])
        tree = cKDTree(np.asarray(prep.points))
        _, near = tree.query(np.asarray(cap.points), k=1)
        normals = prep_normals[np.asarray(near, dtype=int)]

        margin = np.asarray(st.margin_points)
        m_tree = cKDTree(margin)
        d, _ = m_tree.query(np.asarray(cap.points), k=1)
        t_raw = np.clip((d - band) / max(ramp, 1e-6), 0.0, 1.0)
        t = t_raw * t_raw * (3.0 - 2.0 * t_raw)  # smoothstep
        offset_amt = no_cement * (1.0 - t) + gap * t

        inner = cap.copy()
        inner.points = (np.asarray(cap.points) + normals * offset_amt[:, None]
                        ).astype(cap.points.dtype)
        return inner


    def _bulge_outer_for_thickness(self, outer, inner, min_thickness):
        """Push OUTER-crown vertices outward along their own normals wherever
        the local distance to the inner shell is less than `min_thickness`.
        Inner stays untouched (glued to the prep for seating fit). Ends with a
        localised Laplacian smooth so the bulge blends into the anatomy.

        Returns the number of vertices that were moved.
        """
        impl = vtk.vtkImplicitPolyDataDistance()
        impl.SetInput(inner)

        outer_n = outer.compute_normals(
            point_normals=True, cell_normals=False,
            auto_orient_normals=True, inplace=False,
        )
        normals = np.asarray(outer_n["Normals"], dtype=np.float64)
        pts = np.asarray(outer.points, dtype=np.float64).copy()

        moved_mask = np.zeros(len(pts), dtype=bool)
        for _ in range(3):  # a few passes to catch curvature-induced residual
            any_moved = False
            for i in range(len(pts)):
                d = float(impl.EvaluateFunction(
                    [float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])]
                ))
                # We only care about magnitude — inner and outer are on
                # opposite sides of the wall, so `d` may be signed either way
                # for the surface pair; the wall thickness is |d|.
                dabs = abs(d)
                if dabs >= min_thickness:
                    continue
                push = min_thickness - dabs
                pts[i] += normals[i] * push
                moved_mask[i] = True
                any_moved = True
            if not any_moved:
                break

        if moved_mask.any():
            pts = self._smooth_local_region(outer, pts, moved_mask,
                                            iterations=6, expand_rings=2)

        outer.points = pts.astype(outer.points.dtype)
        return int(moved_mask.sum())

    def _smooth_local_region(self, mesh, pts, seed_mask,
                             iterations=6, expand_rings=2):
        """Laplacian-smooth vertices in `seed_mask` (and their neighbour rings
        out to `expand_rings`) toward their neighbourhood mean. Untouched
        vertices are held fixed so the smoothing blends the bulge into the
        surrounding anatomy without disturbing the rest of the crown."""
        n = len(pts)
        faces_arr = np.asarray(mesh.faces)
        tri = faces_arr.reshape(-1, 4)[:, 1:]
        neighbours = [[] for _ in range(n)]
        for a, b, c in tri:
            a, b, c = int(a), int(b), int(c)
            neighbours[a].extend((b, c))
            neighbours[b].extend((a, c))
            neighbours[c].extend((a, b))
        neighbours = [np.unique(np.asarray(nl, dtype=np.int64)) if nl else np.empty(0, dtype=np.int64)
                      for nl in neighbours]

        # Expand seed by N rings so the smooth blend reaches into unmoved
        # anatomy for a gentle taper.
        active = seed_mask.copy()
        for _ in range(int(expand_rings)):
            grow = active.copy()
            idxs = np.where(active)[0]
            for vi in idxs:
                grow[neighbours[vi]] = True
            active = grow

        # Weight: 1.0 on seed vertices, tapering to 0 at the outer ring.
        weight = np.zeros(n, dtype=np.float64)
        weight[seed_mask] = 1.0
        ring = seed_mask.copy()
        for r in range(int(expand_rings)):
            next_ring = np.zeros_like(ring)
            idxs = np.where(ring)[0]
            for vi in idxs:
                next_ring[neighbours[vi]] = True
            next_ring &= ~ring
            weight[next_ring] = (expand_rings - r) / float(expand_rings + 1)
            ring |= next_ring

        active_idx = np.where(active)[0]
        moved = pts.copy()
        for _ in range(int(iterations)):
            new_pts = moved.copy()
            for vi in active_idx:
                nl = neighbours[vi]
                if len(nl) == 0:
                    continue
                mean = moved[nl].mean(axis=0)
                w = float(weight[vi])
                new_pts[vi] = moved[vi] * (1.0 - w * 0.5) + mean * (w * 0.5)
            moved = new_pts
        return moved

    def _clamp_to_prep(self, inner, target_clearance):
        """Push inner-shell vertices outward so each one clears the prep by at
        least `target_clearance[i]` mm.

        Uses vtkImplicitPolyDataDistance for true signed distance to the prep
        triangle surface (positive outside, negative inside). The matching
        closest point on the prep gives the outward direction: when the inner
        vertex is outside, (vertex - closest) points away from the prep; when
        it's inside, (closest - vertex) does. Moving along that direction by
        (target - dist) restores the target clearance in one shot. A second
        pass cleans up curvature-induced residual.

        Returns the count of vertices that ended up being moved.
        """
        prep = self.app.state.prep_mesh
        if prep is None:
            return 0

        impl = vtk.vtkImplicitPolyDataDistance()
        impl.SetInput(prep)

        pts = np.asarray(inner.points)
        moved = np.zeros(len(pts), dtype=bool)
        closest = [0.0, 0.0, 0.0]

        for _ in range(2):  # two passes — second catches curvature residual
            any_moved = False
            for i in range(len(pts)):
                target = float(target_clearance[i])
                px, py, pz = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
                d = impl.EvaluateFunctionAndGetClosestPoint(
                    [px, py, pz], closest
                )
                if d >= target:
                    continue
                # Outward direction = sign(d) * (pt - closest) / |pt - closest|.
                # When d > 0 (outside), pt - closest already points outward.
                # When d < 0 (inside), it points inward, so flip via sign(d).
                dx = px - closest[0]
                dy = py - closest[1]
                dz = pz - closest[2]
                dnorm = (dx * dx + dy * dy + dz * dz) ** 0.5
                if dnorm < 1e-9:
                    continue  # degenerate (point sits exactly on the prep)
                sgn = 1.0 if d >= 0 else -1.0
                ux, uy, uz = sgn * dx / dnorm, sgn * dy / dnorm, sgn * dz / dnorm
                shift = target - d  # always positive at this branch
                pts[i, 0] = px + ux * shift
                pts[i, 1] = py + uy * shift
                pts[i, 2] = pz + uz * shift
                moved[i] = True
                any_moved = True
            if not any_moved:
                break

        inner.points = pts.astype(inner.points.dtype)
        return int(moved.sum())

    def _toggle_inner(self):
        if self.inner_actor is None: return
        visible = self.btn_show_inner.isChecked()
        self.inner_actor.SetVisibility(bool(visible))
        self.btn_show_inner.setText("Hide Inner Surface" if visible else "Show Inner Surface")
        self.app.plotter.render()

    def _toggle_outer(self):
        place = self._place_stage()
        if place is None: return
        visible = self.btn_show_outer.isChecked()
        place.set_outer_visible(visible)
        self.btn_show_outer.setText("Hide Outer Crown" if visible else "Show Outer Crown")

    # --- Persistence ---

    def serialize(self):
        return {
            "material_idx": int(self.material_box.currentIndex()),
            "thickness_um": int(self.thickness_spin.value()),
            "cement_gap_um": int(self.gap_spin.value()),
            "margin_offset_mm": float(self.margin_offset_spin.value()),
            "min_wall_mm": float(self.min_thickness_spin.value()),
        }

    def restore(self, data):
        self._suppress = True
        self.material_box.setCurrentIndex(int(data.get("material_idx", 0)))
        self.thickness_spin.setValue(int(data.get("thickness_um", self.MATERIALS[0][1])))
        self._suppress = False
        self.gap_spin.setValue(int(data.get("cement_gap_um", 40)))
        self.margin_offset_spin.setValue(float(data.get("margin_offset_mm", 1.0)))
        self.min_thickness_spin.setValue(float(data.get("min_wall_mm", 0.5)))

        if self.inner_actor is not None:
            try: self.app.plotter.remove_actor(self.inner_actor)
            except Exception: pass
            self.inner_actor = None

        if self.app.state.shell_inner is not None:
            place = self._place_stage()
            if place is not None:
                place.set_outer_opacity(0.35)
            self.inner_actor = self.app.plotter.add_mesh(
                self.app.state.shell_inner, color="#1f6feb", opacity=1.0, show_edges=False
            )
            self.btn_show_inner.setEnabled(True)
            self.btn_show_inner.setChecked(True)
            self.btn_show_inner.setText("Hide Inner Surface")
            self.btn_show_outer.setEnabled(True)
            self.btn_show_outer.setChecked(True)
            self.btn_show_outer.setText("Hide Outer Crown")
            inner = self.app.state.shell_inner
            self.lbl_status.setText(
                f"Shell restored · {self.thickness_spin.value()} μm wall · "
                f"{self.gap_spin.value()} μm cement gap "
                f"(above {self.margin_offset_spin.value():.2f} mm)\n"
                f"Inner surface: {inner.n_points:,} verts, {inner.n_cells:,} faces"
            )
