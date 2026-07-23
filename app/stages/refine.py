"""Stage 5 — stitch the trim edges into a watertight solid, then optionally
sculpt the crown with a brush before STL export.

The sculpt brush moves outer + inner vertices together along a single brush
direction (the surface normal at the cursor), so wall thickness is preserved
locally — the crown's region "follows" the brush rather than thinning out.
"""
import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox,
    QDoubleSpinBox, QSpinBox, QComboBox,
)

from ..config import STAGES
from ..ui import section_label
from .base import Stage
from .trim import TrimStage


class RefineStage(Stage):
    name = "Refine"
    description = STAGES[6][1]

    def __init__(self, app):
        super().__init__(app)
        self.final_actor = None

        # Heatmap state — visualises clearance / collision vs neighbouring meshes.
        self._heatmap_actor = None
        self._heatmap_bar_name = "heatmap_bar"
        self._heatmap_on = False

        # Sculpt-brush state. Lazily populated when the user enables sculpting.
        self._sculpt_enabled = False
        self._push_mode = True            # True → push outward; False → pull inward
        self._brush_radius = 1.5          # mm
        self._brush_strength = 0.05       # mm per stroke step
        self._cursor_actor = None
        self._cursor_world_pos = None
        self._cursor_normal = None
        self._kdtree = None               # rebuilt at the start of each stroke
        self._is_dragging = False
        self._undo_stack = []             # list of np.ndarray copies of mesh.points
        self._redo_stack = []             # parallel stack for redo (Ctrl+Y / Ctrl+Shift+Z)
        self._UNDO_MAX = 20
        self._obs_ids = []                # vtk observer IDs for cleanup

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # --- SOLIDIFY ---
        layout.addWidget(section_label("SOLIDIFY"))
        hint = QLabel(
            "Stitches the two open trim edges (outer + inner) into a watertight band "
            "so the crown can be exported as a closed solid."
        )
        hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 4px 0;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.btn_solidify = QPushButton("Solidify (Stitch & Close)")
        self.btn_solidify.setObjectName("primary")
        self.btn_solidify.clicked.connect(self.solidify)
        layout.addWidget(self.btn_solidify)

        # --- STATS ---
        layout.addWidget(section_label("STATS"))
        self.lbl_stats = QLabel("Not yet solidified.")
        self.lbl_stats.setStyleSheet("color: #6e6e73; font-size: 12px;")
        self.lbl_stats.setWordWrap(True)
        layout.addWidget(self.lbl_stats)

        # --- SCULPT ---
        layout.addWidget(section_label("SCULPT"))
        sculpt_hint = QLabel(
            "Hover the red brush over the crown; click or drag to deform. "
            "Outer + inner move together so the wall thickness stays."
        )
        sculpt_hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 2px 0;")
        sculpt_hint.setWordWrap(True)
        layout.addWidget(sculpt_hint)

        self.btn_sculpt = QPushButton("Enable Sculpting")
        self.btn_sculpt.setCheckable(True)
        self.btn_sculpt.setStyleSheet(
            "QPushButton:checked { background-color: #0071e3; color: white; border-color: #0071e3; }"
        )
        self.btn_sculpt.setEnabled(False)
        self.btn_sculpt.clicked.connect(self._toggle_sculpt)
        layout.addWidget(self.btn_sculpt)

        self.btn_pushpull = QPushButton("Mode: Push outward")
        self.btn_pushpull.setCheckable(True)
        self.btn_pushpull.setStyleSheet(
            "QPushButton:checked { background-color: #d46a00; color: white; border-color: #d46a00; }"
        )
        self.btn_pushpull.setEnabled(False)
        self.btn_pushpull.clicked.connect(self._toggle_pushpull)
        layout.addWidget(self.btn_pushpull)

        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("Brush radius"))
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0.3, 5.0)
        self.spin_radius.setSingleStep(0.1)
        self.spin_radius.setDecimals(2)
        self.spin_radius.setSuffix(" mm")
        self.spin_radius.setValue(self._brush_radius)
        self.spin_radius.setEnabled(False)
        self.spin_radius.valueChanged.connect(self._on_radius_change)
        radius_row.addWidget(self.spin_radius)
        layout.addLayout(radius_row)

        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("Strength"))
        self.spin_strength = QDoubleSpinBox()
        self.spin_strength.setRange(0.005, 0.5)
        self.spin_strength.setSingleStep(0.01)
        self.spin_strength.setDecimals(3)
        self.spin_strength.setSuffix(" mm")
        self.spin_strength.setValue(self._brush_strength)
        self.spin_strength.setEnabled(False)
        self.spin_strength.valueChanged.connect(self._on_strength_change)
        strength_row.addWidget(self.spin_strength)
        layout.addLayout(strength_row)

        undo_row = QHBoxLayout()
        self.btn_undo_sculpt = QPushButton("Undo (Ctrl+Z)")
        self.btn_undo_sculpt.setEnabled(False)
        self.btn_undo_sculpt.clicked.connect(self._undo_sculpt)
        undo_row.addWidget(self.btn_undo_sculpt)
        self.btn_redo_sculpt = QPushButton("Redo (Ctrl+Y)")
        self.btn_redo_sculpt.setEnabled(False)
        self.btn_redo_sculpt.clicked.connect(self._redo_sculpt)
        undo_row.addWidget(self.btn_redo_sculpt)
        layout.addLayout(undo_row)

        # --- HEATMAP ---
        layout.addWidget(section_label("HEATMAP"))
        heat_hint = QLabel(
            "Colours the crown by distance to a target mesh. "
            "Red = penetration / too close, blue = clear."
        )
        heat_hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 2px 0;")
        heat_hint.setWordWrap(True)
        layout.addWidget(heat_hint)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self.cmb_heatmap_mode = QComboBox()
        self.cmb_heatmap_mode.addItem("Collision (opposing + adjacent)", "collision")
        self.cmb_heatmap_mode.addItem("Occlusal clearance (opposing)", "occlusion")
        self.cmb_heatmap_mode.addItem("Prep fit (internal)", "fit")
        self.cmb_heatmap_mode.currentIndexChanged.connect(self._on_heatmap_mode_change)
        mode_row.addWidget(self.cmb_heatmap_mode, 1)
        layout.addLayout(mode_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Range ±"))
        self.spin_heatmap_range = QDoubleSpinBox()
        self.spin_heatmap_range.setRange(0.05, 5.0)
        self.spin_heatmap_range.setSingleStep(0.05)
        self.spin_heatmap_range.setDecimals(2)
        self.spin_heatmap_range.setSuffix(" mm")
        self.spin_heatmap_range.setValue(0.5)
        self.spin_heatmap_range.valueChanged.connect(self._on_heatmap_range_change)
        range_row.addWidget(self.spin_heatmap_range)
        layout.addLayout(range_row)

        self.btn_heatmap = QPushButton("Show Heatmap")
        self.btn_heatmap.setCheckable(True)
        self.btn_heatmap.setStyleSheet(
            "QPushButton:checked { background-color: #d46a00; color: white; border-color: #d46a00; }"
        )
        self.btn_heatmap.setEnabled(False)
        self.btn_heatmap.clicked.connect(self._toggle_heatmap)
        layout.addWidget(self.btn_heatmap)

        # --- OCCLUSAL RELIEF ---
        layout.addWidget(section_label("OCCLUSAL RELIEF"))
        relief_hint = QLabel(
            "Push crown vertices that touch or penetrate the opposing arch "
            "inward until they clear it by the target amount."
        )
        relief_hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 2px 0;")
        relief_hint.setWordWrap(True)
        layout.addWidget(relief_hint)

        relief_row = QHBoxLayout()
        relief_row.addWidget(QLabel("Target clearance"))
        self.spin_relief_target = QDoubleSpinBox()
        self.spin_relief_target.setRange(0.00, 1.00)
        self.spin_relief_target.setSingleStep(0.01)
        self.spin_relief_target.setDecimals(2)
        self.spin_relief_target.setSuffix(" mm")
        self.spin_relief_target.setValue(0.05)
        relief_row.addWidget(self.spin_relief_target)
        layout.addLayout(relief_row)

        self.btn_relieve = QPushButton("Relieve Occlusion")
        self.btn_relieve.setEnabled(False)
        self.btn_relieve.clicked.connect(self._relieve_occlusion)
        layout.addWidget(self.btn_relieve)

        # --- SMOOTH ---
        layout.addWidget(section_label("SMOOTH"))
        smooth_hint = QLabel(
            "Softens the crown surface. Higher iterations = smoother but "
            "risks losing fine anatomy."
        )
        smooth_hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 2px 0;")
        smooth_hint.setWordWrap(True)
        layout.addWidget(smooth_hint)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Iterations"))
        self.spin_smooth_iter = QSpinBox()
        self.spin_smooth_iter.setRange(1, 100)
        self.spin_smooth_iter.setValue(10)
        smooth_row.addWidget(self.spin_smooth_iter)
        layout.addLayout(smooth_row)

        self.btn_smooth = QPushButton("Smooth Crown")
        self.btn_smooth.setEnabled(False)
        self.btn_smooth.clicked.connect(self._smooth_crown)
        layout.addWidget(self.btn_smooth)

        # --- EXPORT ---
        layout.addWidget(section_label("EXPORT"))
        self.btn_export_stl = QPushButton("Export STL...")
        self.btn_export_stl.clicked.connect(self.export_stl)
        self.btn_export_stl.setEnabled(False)
        layout.addWidget(self.btn_export_stl)

        layout.addStretch()

    # --- Stage lifecycle ---

    def is_complete(self):
        return self.app.state.final_crown is not None

    def on_enter(self):
        # When viewing the final crown, hide the unstitched trim so the user sees the closed solid.
        trim = self._trim_stage()
        if self.app.state.final_crown is not None:
            if trim is not None and trim.trimmed_actor is not None:
                trim.trimmed_actor.SetVisibility(False)
            if self.final_actor is not None:
                self.final_actor.SetVisibility(True)
            self._set_sculpt_controls_enabled(True)
        else:
            self._set_sculpt_controls_enabled(False)
        self.app.set_status(self.description)
        self.app.plotter.render()

    def on_exit(self):
        # Tear down sculpting before leaving so its observers / cursor don't leak.
        if self._sculpt_enabled:
            self._disable_sculpt()
        if self._heatmap_on:
            self._hide_heatmap()
        # Restore the trim view if we're leaving Refine without it being final
        trim = self._trim_stage()
        if trim is not None and trim.trimmed_actor is not None:
            trim.trimmed_actor.SetVisibility(True)
        self.app.plotter.render()

    def reset_final(self):
        if self._sculpt_enabled:
            self._disable_sculpt()
        if self.final_actor is not None:
            try: self.app.plotter.remove_actor(self.final_actor)
            except Exception: pass
        self.final_actor = None
        self.app.state.final_crown = None
        self.lbl_stats.setText("Not yet solidified.")
        self.btn_export_stl.setEnabled(False)
        self.btn_relieve.setEnabled(False)
        self.btn_smooth.setEnabled(False)
        self._set_sculpt_controls_enabled(False)
        self._undo_stack.clear()
        self._redo_stack.clear()
        # Restore trim visibility
        trim = self._trim_stage()
        if trim is not None and trim.trimmed_actor is not None:
            trim.trimmed_actor.SetVisibility(True)
        self.completion_changed.emit()
        self.app.plotter.render()

    # --- Solidify ---

    def solidify(self):
        trimmed = self.app.state.trimmed_crown
        if trimmed is None:
            QMessageBox.warning(self, "No trim", "Apply trim in Stage 4 first.")
            return

        # Extract open boundary edges
        boundary = trimmed.extract_feature_edges(
            boundary_edges=True, non_manifold_edges=False,
            feature_edges=False, manifold_edges=False,
        )
        loops = self._extract_ordered_loops(boundary)
        if len(loops) < 2:
            QMessageBox.warning(
                self, "Stitching failed",
                f"Expected 2 trim-edge loops (outer + inner), found {len(loops)}.\n"
                "Re-apply trim in Stage 4 and try again.",
            )
            return

        # Largest perimeter = outer trim, second = inner trim
        loops.sort(key=self._loop_perimeter, reverse=True)
        outer_loop, inner_loop = loops[0], loops[1]

        # The raw clip_scalar rim loops zigzag. Cyclic-Laplacian-smooth each
        # loop in place (same vertex count, same order), then push the
        # smoothed positions back into the trimmed mesh so the rim of the
        # crown itself becomes smooth. The band built from the same smoothed
        # coords then shares vertices with the trimmed crown after clean(),
        # so the merged solid is watertight.
        outer_loop, trimmed = self._smooth_loop_in_place(
            trimmed, outer_loop, passes=15)
        inner_loop, trimmed = self._smooth_loop_in_place(
            trimmed, inner_loop, passes=15)

        # Prefer the DESIGNED crown border (swept profile from stage 2) as the
        # bottom surface — that's the geometry the user shaped explicitly.
        # Fall back to the greedy triangulated band if no border was designed.
        band = self._designed_border_band(outer_loop, inner_loop)
        used_designed_border = band is not None and band.n_cells > 0
        if not used_designed_border:
            band = self._stitch_band(outer_loop, inner_loop)
            if band is None or band.n_cells == 0:
                QMessageBox.warning(self, "Stitching failed",
                                    "Could not build a triangulated band between the trim edges.")
                return

        # Tolerant clean so the band's rim verts weld to the (newly smoothed)
        # trimmed-mesh rim verts despite any sub-micron float drift.
        merged = trimmed.merge(band).clean(tolerance=1e-4)
        # Round off the crisp creases where the stitch band meets the outer
        # crown (at the fit-ring) and the inner shell (at the margin) — both
        # joints are otherwise sharp by construction. Smooth only vertices
        # within `band_smooth_dist` mm of either rim so the rest of the
        # anatomy is left alone.
        outer_rim = np.asarray(outer_loop)
        inner_rim = np.asarray(inner_loop)
        merged = self._smooth_band_edges(merged, outer_rim, inner_rim,
                                         radius=0.6, iterations=8)
        # Make face winding consistent so STL export looks right
        try:
            merged = merged.compute_normals(
                auto_orient_normals=True, consistent_normals=True,
                non_manifold_traversal=False, inplace=False,
            )
        except Exception:
            pass

        # Hide the open trim actor and show the closed final
        trim = self._trim_stage()
        if trim is not None and trim.trimmed_actor is not None:
            trim.trimmed_actor.SetVisibility(False)
        if self.final_actor is not None:
            try: self.app.plotter.remove_actor(self.final_actor)
            except Exception: pass

        self.app.state.final_crown = merged
        self.final_actor = self.app.plotter.add_mesh(merged, color='gold', show_edges=False)
        self.app.plotter.render()

        # Stats
        try: n_open = int(merged.n_open_edges)
        except Exception: n_open = -1
        try: vol = float(merged.volume)
        except Exception: vol = None

        band_source = "designed crown border" if used_designed_border else "auto-stitch"
        msg = (
            f"Crown: {merged.n_points:,} verts, {merged.n_cells:,} faces.\n"
            f"Band added: {band.n_cells:,} triangles ({band_source})."
        )
        if n_open == 0:
            msg += "\n✓ Watertight — ready for export."
        elif n_open > 0:
            msg += f"\n⚠ {n_open} open edges remain — STL may not be perfectly watertight."
        if vol is not None:
            msg += f"\nVolume: {vol:.1f} mm³"
        self.lbl_stats.setText(msg)

        self.btn_export_stl.setEnabled(True)
        self.btn_relieve.setEnabled(True)
        self.btn_smooth.setEnabled(True)
        self._set_sculpt_controls_enabled(True)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.completion_changed.emit()
        self.app.set_status("Crown solidified. You can now export STL.")

    # --- Sculpt brush ---

    def _vtk_iren(self):
        """Return the raw vtkRenderWindowInteractor.

        Recent PyVista wraps it in a snake-case helper class
        (`pyvista.plotting.render_window_interactor.RenderWindowInteractor`)
        that doesn't expose `AddObserver` / `RemoveObserver` / `SetAbortFlag`.
        Reach through to whichever attribute the wrapper uses to hold the
        real VTK object.
        """
        iren = self.app.plotter.iren
        if hasattr(iren, "AddObserver"):
            return iren  # already a raw vtk interactor
        for attr in ("interactor", "_iren"):
            inner = getattr(iren, attr, None)
            if inner is not None and hasattr(inner, "AddObserver"):
                return inner
        # Last resort: ask the render window directly.
        return self.app.plotter.render_window.GetInteractor()

    def _set_sculpt_controls_enabled(self, enabled):
        """Greys out the whole sculpt section when there's no final crown to edit."""
        self.btn_sculpt.setEnabled(enabled)
        self.btn_pushpull.setEnabled(enabled)
        self.spin_radius.setEnabled(enabled)
        self.spin_strength.setEnabled(enabled)
        self.btn_heatmap.setEnabled(enabled)
        if not enabled:
            self.btn_sculpt.setChecked(False)
            self.btn_sculpt.setText("Enable Sculpting")
            self.btn_undo_sculpt.setEnabled(False)
            self.btn_redo_sculpt.setEnabled(False)
            if self._heatmap_on:
                self._hide_heatmap()

    def _toggle_sculpt(self):
        if self.btn_sculpt.isChecked():
            self._enable_sculpt()
        else:
            self._disable_sculpt()

    def _toggle_pushpull(self):
        self._push_mode = not self.btn_pushpull.isChecked()
        self.btn_pushpull.setText(
            "Mode: Pull inward" if self.btn_pushpull.isChecked() else "Mode: Push outward"
        )

    def _on_radius_change(self, v):
        self._brush_radius = float(v)
        # Rescale the cursor indicator if it's already visible.
        if self._cursor_actor is not None:
            self._cursor_actor.SetScale(
                self._brush_radius, self._brush_radius, self._brush_radius
            )
            self.app.plotter.render()

    def _on_strength_change(self, v):
        self._brush_strength = float(v)

    def _enable_sculpt(self):
        mesh = self.app.state.final_crown
        if mesh is None:
            self.btn_sculpt.setChecked(False)
            return
        self._sculpt_enabled = True
        self.btn_sculpt.setText("Disable Sculpting")
        # Cursor indicator: unit sphere we just rescale + reposition per frame.
        if self._cursor_actor is None:
            unit = pv.Sphere(radius=1.0, center=(0.0, 0.0, 0.0))
            self._cursor_actor = self.app.plotter.add_mesh(
                unit, color="red", opacity=0.25,
                pickable=False, reset_camera=False,
            )
            self._cursor_actor.SetScale(
                self._brush_radius, self._brush_radius, self._brush_radius
            )
        self._cursor_actor.SetVisibility(False)

        iren = self._vtk_iren()
        # High priority (>1.0) lets us call SetAbortFlag and pre-empt the default
        # camera-rotate when the user actually starts sculpting.
        self._obs_ids = [
            iren.AddObserver("MouseMoveEvent",         self._on_mouse_move,   10.0),
            iren.AddObserver("LeftButtonPressEvent",   self._on_left_press,   10.0),
            iren.AddObserver("LeftButtonReleaseEvent", self._on_left_release, 10.0),
        ]
        self.app.set_status(
            "Sculpting on. Click/drag the red brush on the crown to push or pull."
        )

    def _disable_sculpt(self):
        self._sculpt_enabled = False
        self._is_dragging = False
        self.btn_sculpt.setText("Enable Sculpting")
        self.btn_sculpt.setChecked(False)
        try:
            iren = self._vtk_iren()
            for oid in self._obs_ids:
                iren.RemoveObserver(oid)
        except Exception:
            pass
        self._obs_ids = []
        if self._cursor_actor is not None:
            try: self.app.plotter.remove_actor(self._cursor_actor)
            except Exception: pass
            self._cursor_actor = None
        self._kdtree = None
        self.app.plotter.render()

    def _on_mouse_move(self, obj, _event):
        if not self._sculpt_enabled:
            return
        x, y = obj.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.001)
        picker.Pick(x, y, 0, self.app.plotter.renderer)
        if picker.GetCellId() < 0:
            # Cursor not over the crown — hide the brush indicator.
            self._cursor_world_pos = None
            if self._cursor_actor is not None and self._cursor_actor.GetVisibility():
                self._cursor_actor.SetVisibility(False)
                self.app.plotter.render()
            return
        world_pos = np.asarray(picker.GetPickPosition(), dtype=float)
        normal = np.asarray(picker.GetPickNormal(), dtype=float)
        n = float(np.linalg.norm(normal))
        if n > 1e-9:
            normal = normal / n
        self._cursor_world_pos = world_pos
        self._cursor_normal = normal
        self._cursor_actor.SetPosition(*world_pos)
        if not self._cursor_actor.GetVisibility():
            self._cursor_actor.SetVisibility(True)
        if self._is_dragging:
            self._apply_brush(world_pos, normal)
            obj.SetAbortFlag(True)  # eat the event so the camera doesn't rotate
        self.app.plotter.render()

    def _on_left_press(self, obj, _event):
        if not self._sculpt_enabled or self._cursor_world_pos is None:
            return
        # Build / rebuild the KDTree at stroke start — point coords are valid
        # until we deform them, then a fresh tree is needed for the next stroke.
        self._rebuild_kdtree()
        self._save_undo_state()
        self._is_dragging = True
        self._apply_brush(self._cursor_world_pos, self._cursor_normal)
        obj.SetAbortFlag(True)
        self.app.plotter.render()

    def _on_left_release(self, obj, _event):
        if not self._sculpt_enabled:
            return
        if self._is_dragging:
            self._is_dragging = False
            obj.SetAbortFlag(True)
            # KDTree is stale now that points moved — drop it; next stroke rebuilds.
            self._kdtree = None
            # Recolour the heatmap so it reflects the deformed geometry.
            self._refresh_heatmap_if_on()

    def _rebuild_kdtree(self):
        mesh = self.app.state.final_crown
        if mesh is None:
            self._kdtree = None
            return
        self._kdtree = cKDTree(np.asarray(mesh.points))

    def _apply_brush(self, world_pos, brush_normal):
        """Move all vertices within `brush_radius` of `world_pos` along the
        brush normal, with a smoothstep falloff so the deformation tapers off
        at the brush boundary. Outer and inner vertices share the direction,
        so wall thickness is locally preserved.
        """
        mesh = self.app.state.final_crown
        if mesh is None or self._kdtree is None:
            return
        idxs = self._kdtree.query_ball_point(world_pos, r=self._brush_radius)
        if not idxs:
            return
        pts = np.asarray(mesh.points)
        nearby = pts[idxs]
        dists = np.linalg.norm(nearby - world_pos, axis=1)
        t = np.clip(1.0 - dists / max(self._brush_radius, 1e-9), 0.0, 1.0)
        falloff = t * t * (3.0 - 2.0 * t)  # smoothstep
        direction = brush_normal if self._push_mode else -brush_normal
        pts[idxs] = nearby + direction * (self._brush_strength * falloff)[:, None]
        mesh.points = pts
        # Tell VTK the geometry changed so the renderer picks it up.
        try:
            mesh.GetPoints().Modified()
        except Exception:
            pass
        self._mark_dirty()

    def _save_undo_state(self):
        mesh = self.app.state.final_crown
        if mesh is None:
            return
        self._undo_stack.append(np.array(mesh.points, copy=True))
        # New action invalidates any redo history.
        self._redo_stack.clear()
        # Bound the stack so a long session doesn't eat memory.
        if len(self._undo_stack) > self._UNDO_MAX:
            self._undo_stack.pop(0)
        self.btn_undo_sculpt.setEnabled(True)
        self.btn_redo_sculpt.setEnabled(False)

    def _undo_sculpt(self):
        if not self._undo_stack:
            return
        mesh = self.app.state.final_crown
        if mesh is None:
            return
        # Push the current state to redo before reverting.
        self._redo_stack.append(np.array(mesh.points, copy=True))
        if len(self._redo_stack) > self._UNDO_MAX:
            self._redo_stack.pop(0)
        mesh.points = self._undo_stack.pop()
        try:
            mesh.GetPoints().Modified()
        except Exception:
            pass
        self.btn_undo_sculpt.setEnabled(bool(self._undo_stack))
        self.btn_redo_sculpt.setEnabled(True)
        self._kdtree = None  # invalidate — points moved
        self._mark_dirty()
        self._refresh_heatmap_if_on()
        self.app.plotter.render()

    def _redo_sculpt(self):
        if not self._redo_stack:
            return
        mesh = self.app.state.final_crown
        if mesh is None:
            return
        # Push the current state to undo so the redo itself is undoable.
        self._undo_stack.append(np.array(mesh.points, copy=True))
        if len(self._undo_stack) > self._UNDO_MAX:
            self._undo_stack.pop(0)
        mesh.points = self._redo_stack.pop()
        try:
            mesh.GetPoints().Modified()
        except Exception:
            pass
        self.btn_undo_sculpt.setEnabled(True)
        self.btn_redo_sculpt.setEnabled(bool(self._redo_stack))
        self._kdtree = None
        self._mark_dirty()
        self.app.plotter.render()

    # Expose stage-level undo/redo so the global Ctrl+Z / Ctrl+Y shortcuts
    # in main_window.py can dispatch to whichever stage is active.
    def stage_undo(self):
        self._undo_sculpt()

    def stage_redo(self):
        self._redo_sculpt()

    def _mark_dirty(self):
        # Notify the main window that the project has unsaved changes.
        try:
            self.app._mark_dirty()
        except Exception:
            pass

    # --- Heatmap ---

    def _on_heatmap_mode_change(self, _idx):
        if self._heatmap_on:
            self._show_heatmap()

    def _on_heatmap_range_change(self, _v):
        if self._heatmap_on:
            self._show_heatmap()

    def _toggle_heatmap(self):
        if self.btn_heatmap.isChecked():
            self._show_heatmap()
        else:
            self._hide_heatmap()

    def _current_target_mesh(self):
        """Build the target mesh for the selected heatmap mode. Returns
        (mesh_or_None, signed_flag). `signed_flag` = True → negative distance
        means the crown vertex is INSIDE the target (a real penetration)."""
        st = self.app.state
        mode = self.cmb_heatmap_mode.currentData()
        if mode == "occlusion":
            return st.opposing_jaw_mesh, False
        if mode == "fit":
            # Prefer the isolated prep mesh; fall back to the full jaw.
            return (st.prep_mesh if st.prep_mesh is not None else st.jaw_mesh), True

        # Collision: opposing jaw + jaw_mesh (excluding the prep tooth region
        # around the crown, otherwise the crown always "penetrates" its own
        # prep). Merge whatever is available into one PolyData.
        parts = []
        if st.opposing_jaw_mesh is not None:
            parts.append(st.opposing_jaw_mesh)
        # Adjacent teeth = full jaw_mesh minus a spherical bubble around the
        # crown centroid, so the prep tooth doesn't paint itself red.
        if st.jaw_mesh is not None and st.final_crown is not None:
            crown_c = np.asarray(st.final_crown.center)
            bbox = np.asarray(st.final_crown.bounds).reshape(3, 2)
            crown_r = float(np.linalg.norm(bbox[:, 1] - bbox[:, 0]) * 0.5)
            jaw_pts = np.asarray(st.jaw_mesh.points)
            keep_mask = np.linalg.norm(jaw_pts - crown_c, axis=1) > (crown_r + 0.5)
            # Extract cells whose ALL vertices are outside the crown bubble.
            try:
                sel = st.jaw_mesh.extract_points(
                    np.where(keep_mask)[0], adjacent_cells=False,
                ).extract_surface()
                if sel is not None and sel.n_cells > 0:
                    parts.append(sel)
            except Exception:
                pass
        if not parts:
            return None, True
        merged = parts[0]
        for p in parts[1:]:
            try:
                merged = merged.merge(p)
            except Exception:
                pass
        return merged, True

    def _show_heatmap(self):
        crown = self.app.state.final_crown
        if crown is None:
            return
        target, signed = self._current_target_mesh()
        if target is None or target.n_points == 0:
            QMessageBox.information(
                self, "No target mesh",
                "The selected heatmap target isn't loaded in this case.",
            )
            self.btn_heatmap.setChecked(False)
            return

        try:
            # Attach the distance field to the crown itself (inplace) so the
            # heatmap actor shares its point array with the sculpt brush.
            crown.compute_implicit_distance(target, inplace=True)
        except Exception as e:
            QMessageBox.critical(self, "Heatmap failed", str(e))
            self.btn_heatmap.setChecked(False)
            return

        d = np.asarray(crown["implicit_distance"], dtype=float)
        rng = float(self.spin_heatmap_range.value())
        title = "Signed dist (mm)" if signed else "Clearance (mm)"

        # Build a per-vertex RGB colour ramp:
        #   d = -rng  → deep red    (heavy penetration)
        #   d =  0    → yellow      (contact)
        #   d = +rng  → gold        (comfortable clearance)
        # Attaching colours directly (instead of two overlapping actors with
        # scalars + opacity) removes z-fighting entirely and lets us keep the
        # gold "base" look in low-interest regions without a second layer.
        rgb = self._heatmap_colors(d, rng)
        crown["heatmap_rgb"] = rgb

        # Replace the gold actor with a single scalars-coloured actor on the
        # same PolyData. Since it's the same points array the sculpt brush
        # deforms, the heatmap stays glued to the crown while you sculpt.
        if self._heatmap_actor is not None:
            try: self.app.plotter.remove_actor(self._heatmap_actor)
            except Exception: pass
            self._heatmap_actor = None
        try:
            self.app.plotter.remove_scalar_bar(self._heatmap_bar_name)
        except Exception:
            pass
        if self.final_actor is not None:
            self.final_actor.SetVisibility(False)

        self._heatmap_actor = self.app.plotter.add_mesh(
            crown, scalars="heatmap_rgb", rgb=True,
            show_edges=False, reset_camera=False, pickable=True,
        )
        self._heatmap_on = True
        self.btn_heatmap.setText("Hide Heatmap")
        n_pen = int(np.sum(d < 0)) if signed else int(np.sum(np.abs(d) < 0.1))
        self.app.set_status(
            f"Heatmap: {n_pen:,} crown verts flagged (range ±{rng:.2f} mm)."
        )
        self.app.plotter.render()

    def _heatmap_colors(self, d, rng):
        """Vertex RGB ramp from deep red (penetration) → yellow (contact) →
        gold (clearance ≥ rng). Uint8, shape (N, 3)."""
        d = np.asarray(d, dtype=np.float32)
        rng = max(float(rng), 1e-6)

        # Normalise into two halves so the colour ramp is smooth on both sides
        # of zero regardless of asymmetric distance distribution.
        t_neg = np.clip(-d / rng, 0.0, 1.0)   # 0 at contact, 1 at deep pen
        t_pos = np.clip(d / rng, 0.0, 1.0)    # 0 at contact, 1 at rng clear

        # Anchor colours (RGB 0..1)
        C_YELLOW = np.array([1.00, 0.95, 0.30])   # contact
        C_RED    = np.array([0.60, 0.00, 0.00])   # deep penetration
        C_GOLD   = np.array([0.83, 0.68, 0.21])   # clear (base crown look)

        rgb = np.empty((d.shape[0], 3), dtype=np.float32)
        neg_mask = d < 0
        pos_mask = ~neg_mask

        # Smoothstep for a soft midband
        def ss(t): return t * t * (3.0 - 2.0 * t)

        wn = ss(t_neg[neg_mask])[:, None]
        rgb[neg_mask] = C_YELLOW * (1.0 - wn) + C_RED * wn

        wp = ss(t_pos[pos_mask])[:, None]
        rgb[pos_mask] = C_YELLOW * (1.0 - wp) + C_GOLD * wp

        return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _refresh_heatmap_if_on(self):
        """Called at the end of a sculpt stroke so the colours re-sync with
        the deformed crown geometry."""
        if not self._heatmap_on:
            return
        crown = self.app.state.final_crown
        target, _ = self._current_target_mesh()
        if crown is None or target is None:
            return
        try:
            crown.compute_implicit_distance(target, inplace=True)
        except Exception:
            return
        d = np.asarray(crown["implicit_distance"], dtype=float)
        rng = float(self.spin_heatmap_range.value())
        crown["heatmap_rgb"] = self._heatmap_colors(d, rng)
        try:
            crown.GetPointData().Modified()
        except Exception:
            pass
        self.app.plotter.render()

    def _smooth_crown(self):
        """Laplacian-smooth the whole final crown for `iterations` passes.
        Pure numpy (no VTK filter), so no segfault risk. Uses a Taubin-like
        alternating shrink/inflate pair so the mesh doesn't collapse."""
        crown = self.app.state.final_crown
        if crown is None:
            return
        n_iter = int(self.spin_smooth_iter.value())
        if n_iter <= 0:
            return

        pts = np.asarray(crown.points, dtype=np.float64).copy()
        n = len(pts)

        # Build vertex adjacency once.
        faces_arr = np.asarray(crown.faces)
        tri = faces_arr.reshape(-1, 4)[:, 1:]
        neighbours = [[] for _ in range(n)]
        for a, b, c in tri:
            a, b, c = int(a), int(b), int(c)
            neighbours[a].extend((b, c))
            neighbours[b].extend((a, c))
            neighbours[c].extend((a, b))
        neighbours = [np.unique(np.asarray(nl, dtype=np.int64)) if nl
                      else np.empty(0, dtype=np.int64) for nl in neighbours]

        # Taubin: alternate a positive Laplacian step and a slightly larger
        # negative step to keep volume roughly constant while smoothing.
        LAMBDA = 0.5
        MU = -0.53
        moved = pts
        for it in range(n_iter):
            factor = LAMBDA if it % 2 == 0 else MU
            new_pts = moved.copy()
            for vi in range(n):
                nl = neighbours[vi]
                if len(nl) == 0:
                    continue
                mean = moved[nl].mean(axis=0)
                new_pts[vi] = moved[vi] + factor * (mean - moved[vi])
            moved = new_pts

        crown.points = moved.astype(crown.points.dtype)
        try: crown.GetPoints().Modified()
        except Exception: pass
        self._refresh_heatmap_if_on()
        try: self.app._mark_dirty()
        except Exception: pass
        self.app.plotter.render()
        self.app.set_status(f"Smoothed crown ({n_iter} iterations).")

    def _relieve_occlusion(self):
        """Push crown verts that are too close to the opposing arch INWARD
        along the crown's own surface normal, in small capped steps.

        Rationale: on an open opposing scan (jaw mesh isn't watertight),
        vtkImplicitPolyDataDistance's signed distance and closest-point
        direction are unreliable — a big shift = (target - d) can jump
        several mm and carve craters. Pushing along the crown's OWN inward
        normal by a tiny per-pass step (STEP_CAP) always shrinks the crown
        locally and never overshoots, regardless of opposing topology.
        """
        crown = self.app.state.final_crown
        opposing = self.app.state.opposing_jaw_mesh
        if crown is None:
            QMessageBox.warning(self, "No crown", "Solidify the crown first.")
            return
        if opposing is None or opposing.n_points == 0:
            QMessageBox.warning(
                self, "No opposing arch",
                "The opposing arch STL isn't loaded — nothing to relieve against.",
            )
            return

        target = float(self.spin_relief_target.value())
        impl = vtk.vtkImplicitPolyDataDistance()
        impl.SetInput(opposing)

        pts = np.asarray(crown.points, dtype=np.float64).copy()
        n = len(pts)

        # Crown outward normals — computed once. Small per-pass steps mean
        # slightly stale normals don't matter.
        try:
            normals_mesh = crown.compute_normals(
                point_normals=True, cell_normals=False,
                auto_orient_normals=True, non_manifold_traversal=False,
                inplace=False,
            )
            normals = np.asarray(normals_mesh["Normals"], dtype=np.float64)
        except Exception:
            centroid = pts.mean(axis=0)
            v = pts - centroid
            l = np.linalg.norm(v, axis=1, keepdims=True)
            normals = v / np.maximum(l, 1e-9)

        MAX_PASSES = 20
        STEP_CAP = 0.10   # mm per pass — small enough to avoid craters

        total_moved = np.zeros(n, dtype=bool)
        max_pen_initial = 0.0
        for pass_i in range(MAX_PASSES):
            moved_this_pass = 0
            worst = 0.0
            for i in range(n):
                d = float(impl.EvaluateFunction(
                    [float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])]
                ))
                # Use unsigned distance for the target test (opposing may be
                # open). Treat d < 0 as "definitely penetrating" and always
                # push then; treat 0 <= d < target as "too close".
                if d < 0:
                    d_gap = 0.0        # already touching → need full target
                    violating = True
                elif d < target:
                    d_gap = d
                    violating = True
                else:
                    d_gap = d
                    violating = False
                if d < 0 and abs(d) > worst:
                    worst = abs(d)
                if not violating:
                    continue
                shift = min(STEP_CAP, target - d_gap)
                if shift <= 0:
                    continue
                pts[i] -= normals[i] * shift
                total_moved[i] = True
                moved_this_pass += 1
            if pass_i == 0:
                max_pen_initial = worst
            if moved_this_pass == 0:
                break

        # Feather the dent into surrounding anatomy.
        if total_moved.any():
            pts = self._smooth_moved_region(crown, pts, total_moved,
                                            iterations=6, expand_rings=2)

        crown.points = pts.astype(crown.points.dtype)
        try: crown.GetPoints().Modified()
        except Exception: pass
        self._refresh_heatmap_if_on()
        try: self.app._mark_dirty()
        except Exception: pass
        self.app.plotter.render()

        n_moved = int(total_moved.sum())
        self.app.set_status(
            f"Relieved occlusion: {n_moved:,} verts pushed inward · "
            f"initial max penetration {max_pen_initial*1000:.0f} μm."
        )

    def _smooth_moved_region(self, mesh, pts, seed_mask,
                             iterations=6, expand_rings=2):
        """Laplacian-smooth vertices in `seed_mask` (plus `expand_rings` of
        neighbours) toward their neighbourhood mean, with a smoothstep taper
        so the smoothed region blends into the surrounding untouched anatomy.
        """
        n = len(pts)
        faces_arr = np.asarray(mesh.faces)
        tri = faces_arr.reshape(-1, 4)[:, 1:]
        neighbours = [[] for _ in range(n)]
        for a, b, c in tri:
            a, b, c = int(a), int(b), int(c)
            neighbours[a].extend((b, c))
            neighbours[b].extend((a, c))
            neighbours[c].extend((a, b))
        neighbours = [np.unique(np.asarray(nl, dtype=np.int64)) if nl
                      else np.empty(0, dtype=np.int64) for nl in neighbours]

        active = seed_mask.copy()
        for _ in range(int(expand_rings)):
            idxs = np.where(active)[0]
            for vi in idxs:
                active[neighbours[vi]] = True

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
                w = float(weight[vi]) * 0.5
                new_pts[vi] = moved[vi] * (1.0 - w) + mean * w
            moved = new_pts
        return moved

    def _hide_heatmap(self):
        if self._heatmap_actor is not None:
            try: self.app.plotter.remove_actor(self._heatmap_actor)
            except Exception: pass
            self._heatmap_actor = None
        try:
            self.app.plotter.remove_scalar_bar(self._heatmap_bar_name)
        except Exception:
            pass
        if self.final_actor is not None:
            self.final_actor.SetVisibility(True)
        self._heatmap_on = False
        self.btn_heatmap.setChecked(False)
        self.btn_heatmap.setText("Show Heatmap")
        self.app.plotter.render()

    # --- Export ---

    def export_stl(self):
        if self.app.state.final_crown is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export crown STL", "crown.stl", "STL (*.stl)"
        )
        if not path:
            return
        try:
            self.app.state.final_crown.save(path)
            self.app.set_status(f"Exported to {path}")
            QMessageBox.information(self, "Exported", f"Crown saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    # --- Geometry helpers ---

    def _extract_ordered_loops(self, boundary):
        """Walk boundary edges into ordered closed loops of point coords."""
        if boundary.n_cells == 0:
            return []
        lines = boundary.lines.reshape(-1, 3)[:, 1:]  # (n_edges, 2) of point indices
        adj = {}
        for p0, p1 in lines:
            adj.setdefault(int(p0), []).append(int(p1))
            adj.setdefault(int(p1), []).append(int(p0))

        loops = []
        visited_edges = set()

        def edge_key(a, b):
            return (a, b) if a < b else (b, a)

        for start_v in list(adj.keys()):
            for first_nxt in adj[start_v]:
                if edge_key(start_v, first_nxt) in visited_edges:
                    continue
                # Walk the loop
                loop_idx = [start_v]
                visited_edges.add(edge_key(start_v, first_nxt))
                prev, curr = start_v, first_nxt
                safety = 0
                while curr != start_v and safety < len(adj) * 2:
                    safety += 1
                    loop_idx.append(curr)
                    nxt = None
                    for n in adj[curr]:
                        if n != prev and edge_key(curr, n) not in visited_edges:
                            nxt = n
                            break
                    if nxt is None:
                        break
                    visited_edges.add(edge_key(curr, nxt))
                    prev, curr = curr, nxt
                if len(loop_idx) >= 3:
                    loops.append([np.asarray(boundary.points[v]) for v in loop_idx])
        return loops

    def _loop_perimeter(self, loop):
        n = len(loop)
        return float(sum(np.linalg.norm(loop[(i + 1) % n] - loop[i]) for i in range(n)))

    def _designed_border_band(self, outer_loop, inner_loop):
        """Build the crown-bottom band from the DESIGNED border profile in
        state (horizontal + angled + vertical + below_margin), instead of
        the greedy shortest-diagonal stitcher.

        The bottom rim of the band is snapped to `inner_loop` (welds cleanly
        to the trimmed inner shell) and the top rim is snapped to `outer_loop`
        (welds cleanly to the trimmed outer crown). Intermediate rings follow
        the shape of the border profile in the local (outward, +z) frame.
        Returns a PolyData or None if no border was designed.
        """
        from ..border_geometry import compute_border_profile_2d
        st = self.app.state
        profile = compute_border_profile_2d(
            horizontal=st.border_horizontal,
            angled=st.border_angled,
            angle_deg=st.border_angle_deg,
            vertical=st.border_vertical,
            below_margin=st.border_below_margin,
        )
        m = len(profile)
        if m < 2:
            return None  # user didn't design any border geometry

        inner = np.asarray(inner_loop, dtype=float)
        outer = np.asarray(outer_loop, dtype=float)
        n = len(inner)
        if n < 3 or len(outer) < 3:
            return None

        # For each inner vertex, find the closest outer vertex — that's the
        # "corresponding" outer point directly above. Using nearest-neighbour
        # in 3D handles arbitrary loop lengths and orderings.
        outer_tree = cKDTree(outer)
        _, oi = outer_tree.query(inner, k=1)
        outer_matched = outer[np.asarray(oi, dtype=int)]

        # Profile-space normalisation: parametrise each profile point by its
        # arc-length fraction so we can lerp between inner (t=0) and outer
        # (t=1) endpoints while still following the profile's shape in the
        # local outward + z plane.
        prof = np.asarray(profile, dtype=float)  # (m, 2) : x=outward, y=vertical
        pdx = prof[-1, 0] - prof[0, 0]
        pdy = prof[-1, 1] - prof[0, 1]
        p_span = float(np.hypot(pdx, pdy)) or 1e-9
        cum = np.zeros(m)
        for i in range(1, m):
            cum[i] = cum[i - 1] + float(np.hypot(prof[i, 0] - prof[i - 1, 0],
                                                 prof[i, 1] - prof[i - 1, 1]))
        t_along = cum / max(cum[-1], 1e-9)         # 0 → 1
        # Sideways ("bulge") offset in the outward direction at each profile
        # point relative to the straight inner→outer line, in profile-x mm.
        bulge_x = prof[:, 0] - (prof[0, 0] + t_along * pdx)
        bulge_y = prof[:, 1] - (prof[0, 1] + t_along * pdy)

        world_z = np.array([0.0, 0.0, 1.0])

        # Build (n × m) vertex grid: bottom row = inner, top row = outer,
        # intermediate rows = straight-line interpolation + local profile bulge.
        verts = np.empty((n, m, 3), dtype=float)
        for i in range(n):
            base = inner[i]
            tip = outer_matched[i]
            straight_dir = tip - base
            # Local outward direction in the horizontal plane (project out z).
            outward = straight_dir.copy(); outward[2] = 0.0
            onorm = float(np.linalg.norm(outward))
            if onorm > 1e-9:
                outward /= onorm
            else:
                # Fallback: radial-outward from the loop centroid.
                radial = base - inner.mean(axis=0); radial[2] = 0.0
                rnorm = float(np.linalg.norm(radial))
                outward = radial / rnorm if rnorm > 1e-9 else np.array([1.0, 0.0, 0.0])
            for j in range(m):
                if j == 0:
                    verts[i, j] = base
                elif j == m - 1:
                    verts[i, j] = tip
                else:
                    verts[i, j] = (base + t_along[j] * straight_dir
                                   + outward * bulge_x[j]
                                   + world_z * bulge_y[j])

        # Build triangle faces (two per quad).
        faces = []
        for i in range(n):
            i_next = (i + 1) % n
            for j in range(m - 1):
                a = i * m + j
                b = i * m + (j + 1)
                c = i_next * m + j
                d = i_next * m + (j + 1)
                faces.append([3, a, b, c])
                faces.append([3, b, d, c])
        faces_flat = np.asarray(faces, dtype=np.int64).ravel()

        return pv.PolyData(verts.reshape(-1, 3), faces_flat)

    def _stitch_band(self, outer, inner):
        """Triangulate the annular band between two closed loops on (approximately) the same plane.
        Greedy shortest-diagonal walk."""
        M, N = len(outer), len(inner)
        if M < 3 or N < 3:
            return None

        outer_arr = np.array(outer)
        inner_arr = np.array(inner)

        # Determine plane normal from PCA of all points
        all_pts = np.vstack([outer_arr, inner_arr])
        centroid = all_pts.mean(axis=0)
        centered = all_pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1] / np.linalg.norm(vh[-1])

        def signed_area(loop):
            s = 0.0
            for k in range(len(loop)):
                a = loop[k] - centroid
                b = loop[(k + 1) % len(loop)] - centroid
                s += float(np.dot(np.cross(a, b), normal))
            return s

        # Make winding consistent: both CCW around normal
        if signed_area(outer) < 0:
            outer = outer[::-1]; outer_arr = np.array(outer)
        if signed_area(inner) < 0:
            inner = inner[::-1]; inner_arr = np.array(inner)

        # Rotate inner so inner[0] is the closest vertex to outer[0]
        dists = np.linalg.norm(inner_arr - outer_arr[0], axis=1)
        start_j = int(np.argmin(dists))
        inner = inner[start_j:] + inner[:start_j]
        inner_arr = np.array(inner)

        all_band_pts = np.vstack([outer_arr, inner_arr])

        # Greedy shortest-diagonal walk: M+N triangles total
        triangles = []
        i = j = 0
        while i < M or j < N:
            if i >= M:
                triangles.append([0, M + ((j + 1) % N), M + (j % N)])
                j += 1
                continue
            if j >= N:
                triangles.append([i % M, (i + 1) % M, M])
                i += 1
                continue
            d_advance_i = float(np.linalg.norm(outer_arr[(i + 1) % M] - inner_arr[j % N]))
            d_advance_j = float(np.linalg.norm(outer_arr[i % M] - inner_arr[(j + 1) % N]))
            if d_advance_i <= d_advance_j:
                triangles.append([i % M, (i + 1) % M, M + (j % N)])
                i += 1
            else:
                triangles.append([i % M, M + ((j + 1) % N), M + (j % N)])
                j += 1

        if not triangles:
            return None
        faces = []
        for t in triangles:
            faces.extend([3, t[0], t[1], t[2]])
        return pv.PolyData(all_band_pts, np.asarray(faces, dtype=np.int_))

    def _smooth_loop_in_place(self, mesh, loop, passes=12):
        """Cyclic-Laplacian-smooth a closed rim loop AND push the smoothed
        positions back into `mesh` at the corresponding vertex indices.

        Vertex count and ordering are preserved so the smoothed loop can be
        fed straight to the band stitcher and welded with `clean()` later.

        Returns (smoothed_loop_as_list, mesh) — the mesh is the same object,
        returned for call-site symmetry.
        """
        arr = np.asarray(loop, dtype=float)
        n = len(arr)
        if n < 4:
            return loop, mesh

        # Map each loop point to its vertex index in the merged trimmed mesh.
        mesh_pts = np.asarray(mesh.points)
        tree = cKDTree(mesh_pts)
        _, idxs = tree.query(arr, k=1)
        idxs = np.asarray(idxs, dtype=np.int64)

        # Cyclic Laplacian: each point ← weighted avg of itself + ring neighbours.
        smoothed = arr.copy()
        for _ in range(int(passes)):
            prev = np.roll(smoothed, 1, axis=0)
            nxt = np.roll(smoothed, -1, axis=0)
            smoothed = 0.5 * smoothed + 0.25 * prev + 0.25 * nxt

        # Write smoothed coords back into the mesh at the same indices, so the
        # crown's own rim is smooth and shares vertices with the band built
        # from `smoothed` below.
        new_pts = mesh_pts.copy()
        new_pts[idxs] = smoothed
        mesh.points = new_pts.astype(mesh.points.dtype)
        return [smoothed[i] for i in range(n)], mesh

    def _smooth_band_edges(self, mesh, outer_rim, inner_rim,
                           radius=0.6, iterations=8):
        """Laplacian-smooth only the vertices within `radius` mm of either
        stitch rim. Outer rim verts (at the fit_ring) and inner rim verts (at
        the margin) become the two crease lines after band stitching; this
        averages each affected vertex toward its neighbours so those creases
        round off without disturbing the occlusal anatomy or the seating face.
        """
        pts = np.asarray(mesh.points)
        n = len(pts)
        if n == 0:
            return mesh

        # Mark smoothable vertices: those near either rim.
        rim = np.vstack([outer_rim, inner_rim]) if len(outer_rim) and len(inner_rim) \
              else (outer_rim if len(outer_rim) else inner_rim)
        if rim is None or len(rim) == 0:
            return mesh
        tree = cKDTree(rim)
        d, _ = tree.query(pts, k=1)
        movable = d < float(radius)
        if not movable.any():
            return mesh
        # Falloff so verts right on the rim move most, outer fringe barely:
        # smoothstep(1 - d/radius). Keeps the smoothed region blending into
        # the untouched anatomy without a visible seam.
        t = np.clip(1.0 - d / max(radius, 1e-9), 0.0, 1.0)
        weight = t * t * (3.0 - 2.0 * t)

        # Build vertex adjacency from triangle faces.
        faces_arr = np.asarray(mesh.faces)
        # PyVista face stream: [3, a, b, c, 3, a, b, c, ...]
        tri = faces_arr.reshape(-1, 4)[:, 1:]
        neighbours = [[] for _ in range(n)]
        for a, b, c in tri:
            a, b, c = int(a), int(b), int(c)
            neighbours[a].extend((b, c))
            neighbours[b].extend((a, c))
            neighbours[c].extend((a, b))
        # Dedup neighbour lists once.
        neighbours = [np.unique(np.asarray(nl, dtype=np.int64)) if nl else None
                      for nl in neighbours]

        moved = pts.copy()
        movable_idx = np.where(movable)[0]
        for _ in range(int(iterations)):
            new_pts = moved.copy()
            for vi in movable_idx:
                nl = neighbours[vi]
                if nl is None or len(nl) == 0:
                    continue
                mean = moved[nl].mean(axis=0)
                # Lerp toward the neighbour mean by the per-vertex weight.
                w = float(weight[vi])
                new_pts[vi] = moved[vi] * (1.0 - w) + mean * w
            moved = new_pts

        smoothed = mesh.copy()
        smoothed.points = moved.astype(mesh.points.dtype)
        return smoothed

    def _trim_stage(self):
        for s in self.app.stages:
            if isinstance(s, TrimStage):
                return s
        return None

    # --- Persistence ---

    def serialize(self):
        return {}

    def restore(self, data):
        if self.final_actor is not None:
            try: self.app.plotter.remove_actor(self.final_actor)
            except Exception: pass
            self.final_actor = None

        final = self.app.state.final_crown
        if final is None:
            return

        trim = self._trim_stage()
        if trim is not None and trim.trimmed_actor is not None:
            trim.trimmed_actor.SetVisibility(False)

        self.final_actor = self.app.plotter.add_mesh(final, color="gold", show_edges=False)

        try: n_open = int(final.n_open_edges)
        except Exception: n_open = -1
        try: vol = float(final.volume)
        except Exception: vol = None

        msg = f"Crown: {final.n_points:,} verts, {final.n_cells:,} faces."
        if n_open == 0:
            msg += "\n✓ Watertight — ready for export."
        elif n_open > 0:
            msg += f"\n⚠ {n_open} open edges."
        if vol is not None:
            msg += f"\nVolume: {vol:.1f} mm³"
        self.lbl_stats.setText(msg)
        self.btn_export_stl.setEnabled(True)
        self.btn_relieve.setEnabled(True)
        self.btn_smooth.setEnabled(True)
