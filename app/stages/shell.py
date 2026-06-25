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
    QVBoxLayout, QPushButton, QLabel, QMessageBox, QComboBox, QSpinBox,
    QDoubleSpinBox,
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

        # Invalidate any downstream trim that was based on the old shell
        self.app.notify_shell_changed()

        if self.inner_actor is not None:
            try: self.app.plotter.remove_actor(self.inner_actor)
            except Exception: pass
            self.inner_actor = None

        thickness_um = self.thickness_spin.value()
        thickness_mm = thickness_um / 1000.0
        cement_gap_um = self.gap_spin.value()
        cement_gap_mm = cement_gap_um / 1000.0
        margin_offset_mm = float(self.margin_offset_spin.value())

        # Per-vertex outward normals on the placed crown
        with_normals = crown.compute_normals(
            point_normals=True, cell_normals=False,
            auto_orient_normals=True, consistent_normals=True,
            inplace=False,
        )
        normals = np.asarray(with_normals.point_normals)

        # Per-vertex cement-gap ramp based on distance from the margin curve.
        # Vertices within `margin_offset_mm` get 0 gap (seal at finish line);
        # past that, the gap ramps linearly to its full value over RAMP_WIDTH_MM.
        crown_pts = np.asarray(crown.points)
        margin_pts = np.asarray(self.app.state.margin_points)
        if cement_gap_mm > 0 and len(margin_pts) >= 2:
            tree = cKDTree(margin_pts)
            margin_dist, _ = tree.query(crown_pts, k=1)
            t = (margin_dist - margin_offset_mm) / max(self.RAMP_WIDTH_MM, 1e-6)
            ramp = np.clip(t, 0.0, 1.0)
            extra = ramp * cement_gap_mm
        else:
            # No margin curve (shouldn't happen at this stage) or zero gap —
            # fall back to a uniform shell.
            extra = np.zeros(len(crown_pts), dtype=float)

        offsets = (thickness_mm + extra)[:, None]  # broadcast per-vertex

        inner = crown.copy()
        inner.points = (crown_pts - normals * offsets).astype(crown.points.dtype)

        # Non-penetration clamp: ensure every inner-shell vertex sits at least
        # `extra(vertex)` away from the prep surface (so 0-clearance at the
        # margin, full cement gap elsewhere). Inner vertices that ended up
        # inside the prep — or closer than their target clearance — get pushed
        # outward along the prep's signed-distance gradient.
        n_clamped = self._clamp_to_prep(inner, extra)
        # Reverse winding so inner-surface normals point into the cavity.
        # `flip_faces` replaced the deprecated `flip_normals` in newer pyvista.
        if hasattr(inner, 'flip_faces'):
            inner.flip_faces(inplace=True)
        else:
            inner.flip_normals()

        # Coarse self-intersection check: each inner vertex's projection onto
        # its outer normal should land at roughly its expected per-vertex
        # offset. A much smaller projection means the offset overshot in a
        # high-curvature region (the surface folded back on itself).
        deltas = crown_pts - np.asarray(inner.points)
        dots = np.einsum('ij,ij->i', deltas, normals)
        expected = thickness_mm + extra
        n_warn = int(np.sum(dots < expected * 0.5))

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

        msg = (
            f"Shell generated · {thickness_um} μm wall · "
            f"{cement_gap_um} μm cement gap (above {margin_offset_mm:.2f} mm)\n"
            f"Inner surface: {inner.n_points:,} verts, {inner.n_cells:,} faces"
        )
        if n_clamped:
            msg += f"\n✓ {n_clamped} vertices clamped to clear the prep"
        if n_warn:
            msg += f"\n⚠ {n_warn} vertices may self-intersect (high curvature)"
        self.lbl_status.setText(msg)

        self.btn_show_inner.setEnabled(True)
        self.btn_show_inner.setChecked(True)
        self.btn_show_inner.setText("Hide Inner Surface")
        self.btn_show_outer.setEnabled(True)
        self.btn_show_outer.setChecked(True)
        self.btn_show_outer.setText("Hide Outer Crown")
        self.completion_changed.emit()
        self.app.set_status(f"Shell generated with {thickness_um} μm wall thickness.")

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
        }

    def restore(self, data):
        self._suppress = True
        self.material_box.setCurrentIndex(int(data.get("material_idx", 0)))
        self.thickness_spin.setValue(int(data.get("thickness_um", self.MATERIALS[0][1])))
        self._suppress = False
        self.gap_spin.setValue(int(data.get("cement_gap_um", 40)))
        self.margin_offset_spin.setValue(float(data.get("margin_offset_mm", 1.0)))

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
