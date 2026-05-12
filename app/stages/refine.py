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
    QDoubleSpinBox,
)

from ..config import STAGES
from ..ui import section_label
from .base import Stage
from .trim import TrimStage


class RefineStage(Stage):
    name = "Refine"
    description = STAGES[4][1]

    def __init__(self, app):
        super().__init__(app)
        self.final_actor = None

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

        band = self._stitch_band(outer_loop, inner_loop)
        if band is None or band.n_cells == 0:
            QMessageBox.warning(self, "Stitching failed",
                                "Could not build a triangulated band between the trim edges.")
            return

        merged = trimmed.merge(band).clean()
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

        msg = (
            f"Crown: {merged.n_points:,} verts, {merged.n_cells:,} faces.\n"
            f"Band added: {band.n_cells:,} triangles."
        )
        if n_open == 0:
            msg += "\n✓ Watertight — ready for export."
        elif n_open > 0:
            msg += f"\n⚠ {n_open} open edges remain — STL may not be perfectly watertight."
        if vol is not None:
            msg += f"\nVolume: {vol:.1f} mm³"
        self.lbl_stats.setText(msg)

        self.btn_export_stl.setEnabled(True)
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
        if not enabled:
            self.btn_sculpt.setChecked(False)
            self.btn_sculpt.setText("Enable Sculpting")
            self.btn_undo_sculpt.setEnabled(False)
            self.btn_redo_sculpt.setEnabled(False)

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
