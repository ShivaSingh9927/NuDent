"""Stage 1 — mark the margin curve by clicking on the prep tooth.

The polyline between consecutive clicks always follows the mesh surface (a
shortest path on the vertex graph) so the curve cannot cut through the tooth
interior, regardless of how sparse the clicks are. Two modes differ only in
the edge weights used by Dijkstra:

- Plain: pure edge-length geodesic — the most direct surface path.
- Smart Trace: clicks snap to the highest-curvature nearby vertex and the
  path is biased toward curvature ridges, hugging the natural margin.
"""
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QVBoxLayout, QPushButton, QLabel, QFileDialog, QListWidget,
    QApplication, QMessageBox,
)

from ..config import STAGES
from ..ui import section_label
from ..segmentation import split_prep_from_context, isolate_tooth
from .base import Stage


# Curvature-weight bias: higher = path prefers ridge edges more strongly.
# At 0 → pure geodesic; at large values → path will detour along ridges even when far.
CURVATURE_BIAS = 8.0
# Number of nearest neighbours considered when snapping a click to a ridge.
SNAP_K = 30


class MarginStage(Stage):
    name = "Margin"
    description = STAGES[0][1]

    def __init__(self, app):
        super().__init__(app)
        self._point_actors = []
        self._line_actors = []

        # Source clicks (visible as red spheres). The dense traced curve goes into
        # state.margin_points.
        self._user_clicks = []
        self._smart_trace = False

        # Two-actor focus view: full-opacity prep + dimmed non-pickable context.
        # Built on the first margin click and torn down on stage exit.
        self._prep_actor = None
        self._context_actor = None
        self._focus_active = False
        self._cached_context = None  # context sub-mesh recovered alongside prep_mesh

        # Lazily-computed mesh-derived data, keyed off whichever mesh
        # (jaw or isolated prep) margin tracing is running over.
        # _adj_geo: plain edge-length weights → pure surface geodesic.
        # _adj_ridge: curvature-biased weights → Smart Trace ridge path.
        self._cached_mesh_id = None
        self._curvature = None
        self._kdtree = None
        self._adj_geo = None
        self._adj_ridge = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # --- SMART TRACE ---
        layout.addWidget(section_label("SMART TRACE"))
        hint = QLabel(
            "Place sparse clicks along the margin; the path between them "
            "snaps to the prep's curvature ridge."
        )
        hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 4px 0;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.btn_smart = QPushButton("Enable Ridge Tracing")
        self.btn_smart.setCheckable(True)
        self.btn_smart.setStyleSheet(
            "QPushButton:checked { background-color: #0071e3; color: white; border-color: #0071e3; }"
        )
        self.btn_smart.clicked.connect(self._toggle_smart_trace)
        layout.addWidget(self.btn_smart)

        # --- MARGIN POINTS ---
        layout.addWidget(section_label("MARGIN CLICKS"))
        self.point_list = QListWidget()
        self.point_list.setMaximumHeight(140)
        layout.addWidget(self.point_list)

        self.status_label = QLabel("Loop: open (0 clicks)")
        self.status_label.setStyleSheet("color: #6e6e73; font-size: 12px; padding: 4px 0;")
        layout.addWidget(self.status_label)

        # --- TOOLS ---
        layout.addWidget(section_label("TOOLS"))
        self.btn_close = QPushButton("Close Loop  (F)")
        self.btn_undo  = QPushButton("Undo  (Z)")
        self.btn_clear = QPushButton("Clear  (C)")
        self.btn_close.clicked.connect(self.close_loop)
        self.btn_undo.clicked.connect(self.undo)
        self.btn_clear.clicked.connect(self.clear)
        layout.addWidget(self.btn_close)
        layout.addWidget(self.btn_undo)
        layout.addWidget(self.btn_clear)

        # --- SAVE / LOAD ---
        layout.addWidget(section_label("SAVE / LOAD"))
        self.btn_save = QPushButton("Save Margin (.npy)")
        self.btn_load = QPushButton("Load Margin (.npy)")
        self.btn_save.clicked.connect(self.save_margin)
        self.btn_load.clicked.connect(self.load_margin)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_load)

        layout.addStretch()
        self._update_buttons()

    # --- Stage lifecycle ---

    def is_complete(self):
        return self.app.state.margin_loop_closed

    def on_enter(self):
        if self.app.state.jaw_mesh is None:
            return
        # If a prep was already isolated (returning from a later stage or after
        # project restore), re-enter the focus view so picking is constrained.
        if self.app.state.prep_mesh is not None:
            self._enter_focus_view()
        self.app.plotter.enable_surface_point_picking(
            callback=self._on_pick,
            left_clicking=True,
            show_point=False,
            show_message=False,
        )
        self.app.set_status(self.description)

    def on_exit(self):
        try:
            self.app.plotter.disable_picking()
        except Exception:
            pass
        self._exit_focus_view()

    # --- Pick + tools ---

    def _on_pick(self, point):
        if point is None or self.app.state.margin_loop_closed:
            return
        # First margin click also seeds prep isolation. If segmentation
        # succeeds, subsequent picks land on the isolated prep actor only
        # (context is non-pickable), making it physically impossible to
        # extend the margin line onto a neighbour tooth.
        if self.app.state.prep_mesh is None and not self._user_clicks:
            self._try_isolate_prep(point)
            if self.app.state.prep_mesh is not None:
                self._enter_focus_view()
        self._user_clicks.append(np.asarray(point))
        self._rebuild_dense_margin()
        self._redraw_visualization()
        self._update_buttons()

    # --- Prep isolation + focus view ---

    def _try_isolate_prep(self, seed_point):
        """Segment the prep tooth from the first click and swap to focus view.

        Tries pure topological connectivity first (the scanner usually delivers
        the prep as a disconnected component — visible black gaps around it).
        Falls back to the dihedral region-grow used by View > Isolate when the
        prep happens to be topologically attached to the rest of the arch.
        """
        mesh = self.app.state.jaw_mesh
        if mesh is None:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            prep, context = split_prep_from_context(mesh, seed_point)
            if prep is None:
                # Topological fallback: dihedral region-grow, then derive
                # context by extracting the complement face set.
                prep = isolate_tooth(mesh, seed_point)
                if prep is None or prep.n_cells == 0:
                    return
                context = self._complement_mesh(mesh, prep)
            self.app.state.prep_mesh = prep
            self._cached_context = context
            # Invalidate Smart Trace caches — they'll rebuild over prep_mesh.
            self._cached_mesh_id = None
            self._adj_geo = None
            self._adj_ridge = None
        except Exception as e:
            # Non-fatal: user keeps marking on the full jaw if isolation fails.
            self.app.set_status(f"Prep isolation skipped: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _complement_mesh(self, full, sub):
        """Best-effort complement of `sub` within `full` for the context view."""
        try:
            # Match sub points back to full by nearest-neighbour. Adequate
            # because isolate_tooth returns extract_cells of the same mesh.
            tree = cKDTree(np.asarray(full.points))
            _, idxs = tree.query(np.asarray(sub.points), k=1)
            keep_pts = np.ones(full.n_points, dtype=bool)
            keep_pts[idxs] = False
            faces = np.asarray(full.faces).reshape(-1, 4)[:, 1:]
            face_keep = keep_pts[faces].all(axis=1)
            return full.extract_cells(np.where(face_keep)[0]).extract_surface()
        except Exception:
            return None

    def _enter_focus_view(self, context_override=None):
        """Hide the main jaw actor; add bright prep + dimmed non-pickable context."""
        if self._focus_active:
            return
        prep = self.app.state.prep_mesh
        if prep is None:
            return
        context = context_override or self._cached_context
        if context is None:
            context = self._complement_mesh(self.app.state.jaw_mesh, prep)
        self._cached_context = context

        if getattr(self.app, "jaw_actor", None) is not None:
            self.app.jaw_actor.SetVisibility(False)

        if context is not None and context.n_cells > 0:
            self._context_actor = self.app.plotter.add_mesh(
                context, color="lightsteelblue", opacity=0.22,
                pickable=False, reset_camera=False,
            )
        self._prep_actor = self.app.plotter.add_mesh(
            prep, color="white", opacity=1.0,
            pickable=True, reset_camera=False,
        )
        self._focus_active = True
        self.app.plotter.render()

    def _exit_focus_view(self):
        """Remove prep/context actors and restore the full-jaw view."""
        if not self._focus_active:
            return
        for a in (self._prep_actor, self._context_actor):
            if a is not None:
                try: self.app.plotter.remove_actor(a)
                except Exception: pass
        self._prep_actor = None
        self._context_actor = None
        if getattr(self.app, "jaw_actor", None) is not None:
            self.app.jaw_actor.SetVisibility(True)
        self._focus_active = False
        self.app.plotter.render()

    def close_loop(self):
        if len(self._user_clicks) < 3 or self.app.state.margin_loop_closed:
            return
        # Always trace the closing segment along the mesh surface — straight
        # 3D chord would cut through the tooth.
        try:
            self._ensure_mesh_data()
            snapped_last  = self._snap(self._user_clicks[-1], self._smart_trace)
            snapped_first = self._snap(self._user_clicks[0],  self._smart_trace)
            closing_path = self._shortest_path(
                snapped_last, snapped_first, use_ridge=self._smart_trace
            )
            if closing_path is not None and len(closing_path) > 2:
                pts = (self.app.state.prep_mesh or self.app.state.jaw_mesh).points
                # Append the interior of the closing path (endpoints are dups).
                self.app.state.margin_points.extend(
                    [np.asarray(pts[i]) for i in closing_path[1:-1]]
                )
        except Exception as e:
            QMessageBox.warning(self, "Trace failed",
                                f"Could not trace closing segment: {e}\n"
                                "Closing with a straight line instead.")
        self.app.state.margin_loop_closed = True
        self._refresh_list()
        self._redraw_visualization()
        self._update_buttons()
        self.completion_changed.emit()
        n = len(self._user_clicks)
        self.app.set_status(f"Loop closed with {n} click{'s' if n != 1 else ''}. "
                            "Stage 'Place' is now available.")

    def undo(self):
        if self.app.state.margin_loop_closed:
            # Undo just the closing action, keep all clicks
            self.app.state.margin_loop_closed = False
            self._rebuild_dense_margin()
            self._redraw_visualization()
            self._update_buttons()
            self.completion_changed.emit()
            return
        if not self._user_clicks:
            return
        self._user_clicks.pop()
        self._rebuild_dense_margin()
        self._redraw_visualization()
        self._update_buttons()

    def clear(self):
        was_closed = self.app.state.margin_loop_closed
        self._user_clicks = []
        self.app.state.margin_points = []
        self.app.state.margin_loop_closed = False
        # Drop the prep isolation so the next click re-seeds from the full jaw —
        # gives the user an escape hatch if the first click landed wrong.
        self._exit_focus_view()
        self.app.state.prep_mesh = None
        self._cached_context = None
        self._cached_mesh_id = None
        self._adj_geo = None
        self._adj_ridge = None
        self._redraw_visualization()
        self._refresh_list()
        self._update_buttons()
        if was_closed:
            self.completion_changed.emit()

    # --- Save / load ---

    def save_margin(self):
        if not self._user_clicks:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save margin points", "margin.npy", "NumPy (*.npy)"
        )
        if path:
            # Save the dense curve (what downstream actually uses)
            np.save(path, np.array(self.app.state.margin_points))
            self.app.set_status(
                f"Saved {len(self.app.state.margin_points)} margin points to {path}"
            )

    def load_margin(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load margin points", "", "NumPy (*.npy)"
        )
        if not path:
            return
        self.clear()
        loaded = np.load(path)
        # Treat loaded points as user clicks. If smart_trace is off, that's the
        # dense curve too; if on, we'll re-trace between them.
        self._user_clicks = [np.asarray(p) for p in loaded]
        self._rebuild_dense_margin()
        self._refresh_list()
        self._redraw_visualization()
        self._update_buttons()
        self.app.set_status(f"Loaded {len(loaded)} margin points from {path}")

    # --- Smart Trace toggle + lazy mesh data ---

    def _toggle_smart_trace(self):
        wanted = self.btn_smart.isChecked()
        if wanted and self.app.state.jaw_mesh is None:
            QMessageBox.information(self, "No mesh", "Open a prep STL first.")
            self.btn_smart.setChecked(False)
            return
        if wanted:
            try:
                self._ensure_mesh_data()
            except Exception as e:
                QMessageBox.warning(self, "Smart Trace failed",
                                    f"Mesh feature computation failed: {e}")
                self.btn_smart.setChecked(False)
                return
        self._smart_trace = wanted
        self.btn_smart.setText("Disable Ridge Tracing" if wanted else "Enable Ridge Tracing")
        self._rebuild_dense_margin()
        self._redraw_visualization()
        if wanted:
            self.app.set_status("Smart Trace on — sparse clicks auto-follow the ridge.")
        else:
            self.app.set_status(self.description)

    def _ensure_mesh_data(self):
        """One-time per-mesh computation: curvature, KDTree, weighted adjacency.

        Runs over the isolated prep when available — this also guarantees the
        ridge-traced path cannot leak across an interproximal valley onto a
        neighbour tooth.
        """
        mesh = self.app.state.prep_mesh or self.app.state.jaw_mesh
        if mesh is None:
            raise RuntimeError("No jaw mesh loaded.")
        if self._cached_mesh_id == id(mesh) and self._adj_geo is not None:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.app.set_status("Computing mesh features (one-time, may take a few seconds)...")
        QApplication.processEvents()
        try:
            # Per-vertex mean curvature (absolute value — sign isn't meaningful for ridges)
            curv = np.abs(np.asarray(mesh.curvature(curv_type="mean")))
            self._curvature = curv

            # Nearest-neighbour lookup for snapping clicks → mesh vertices
            self._kdtree = cKDTree(np.asarray(mesh.points))

            # Build edge list from triangle faces, dedupe
            faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
            edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
            edges = np.sort(edges, axis=1)
            edges = np.unique(edges, axis=0)

            pts = np.asarray(mesh.points)
            dists = np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=1)

            # Higher curvature → cheaper edge weight, so Dijkstra prefers ridges.
            cnorm = curv / (curv.max() + 1e-9)
            edge_curv = (cnorm[edges[:, 0]] + cnorm[edges[:, 1]]) * 0.5
            ridge_w = dists * (1.0 + CURVATURE_BIAS * (1.0 - edge_curv))

            n = mesh.n_points
            rows = np.concatenate([edges[:, 0], edges[:, 1]])
            cols = np.concatenate([edges[:, 1], edges[:, 0]])
            geo_data = np.concatenate([dists, dists])
            ridge_data = np.concatenate([ridge_w, ridge_w])
            self._adj_geo = csr_matrix((geo_data, (rows, cols)), shape=(n, n))
            self._adj_ridge = csr_matrix((ridge_data, (rows, cols)), shape=(n, n))

            self._cached_mesh_id = id(mesh)
        finally:
            QApplication.restoreOverrideCursor()
            self.app.set_status("Mesh features ready.")

    def _snap(self, point, prefer_ridge):
        """Snap a 3D click to a mesh vertex.

        Smart Trace (prefer_ridge=True) picks the highest-curvature vertex
        among the K nearest neighbours; plain mode picks the single closest.
        """
        if not prefer_ridge:
            _, idx = self._kdtree.query(np.asarray(point), k=1)
            return int(idx)
        k = min(SNAP_K, len(self._curvature))
        _, idxs = self._kdtree.query(np.asarray(point), k=k)
        idxs = np.atleast_1d(idxs)
        return int(idxs[np.argmax(self._curvature[idxs])])

    def _shortest_path(self, start_idx, end_idx, use_ridge):
        """Surface shortest path between two mesh vertex indices.

        use_ridge=True: curvature-biased weights (prefers ridges).
        use_ridge=False: pure edge-length weights (true surface geodesic).
        """
        if start_idx == end_idx:
            return [start_idx]
        adj = self._adj_ridge if use_ridge else self._adj_geo
        _, pred = dijkstra(adj, indices=start_idx,
                           return_predecessors=True)
        if pred[end_idx] < 0:
            return None
        path = [end_idx]
        guard = adj.shape[0]
        while path[-1] != start_idx and guard > 0:
            nxt = int(pred[path[-1]])
            if nxt < 0:
                return None
            path.append(nxt)
            guard -= 1
        if path[-1] != start_idx:
            return None
        return list(reversed(path))

    # --- Curve assembly + drawing ---

    def _rebuild_dense_margin(self):
        """Recompute state.margin_points from _user_clicks.

        The polyline between consecutive clicks always follows a mesh-surface
        shortest path so it cannot cut through the tooth interior. Smart Trace
        biases that path toward curvature ridges; plain mode runs a pure
        edge-length geodesic.
        """
        if not self._user_clicks:
            self.app.state.margin_points = []
            self._refresh_list()
            return

        if len(self._user_clicks) < 2:
            self.app.state.margin_points = [np.asarray(c) for c in self._user_clicks]
            self._refresh_list()
            return

        try:
            self._ensure_mesh_data()
        except Exception:
            # No mesh data → straight chords (best effort).
            self.app.state.margin_points = [np.asarray(c) for c in self._user_clicks]
            self._refresh_list()
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            pts = np.asarray(
                (self.app.state.prep_mesh or self.app.state.jaw_mesh).points
            )
            use_ridge = self._smart_trace
            snapped = [self._snap(c, use_ridge) for c in self._user_clicks]
            dense_idxs = [snapped[0]]
            for i in range(len(snapped) - 1):
                path = self._shortest_path(snapped[i], snapped[i + 1], use_ridge)
                if path is None:
                    # Disconnected — fall back to a direct edge to the next click
                    dense_idxs.append(snapped[i + 1])
                else:
                    dense_idxs.extend(path[1:])  # skip first to avoid dup
            self.app.state.margin_points = [np.asarray(pts[i]) for i in dense_idxs]
        finally:
            QApplication.restoreOverrideCursor()
        self._refresh_list()

    def _redraw_visualization(self):
        # Wipe prior actors
        for a in self._line_actors:
            try: self.app.plotter.remove_actor(a)
            except Exception: pass
        for a in self._point_actors:
            try: self.app.plotter.remove_actor(a)
            except Exception: pass
        self._line_actors.clear()
        self._point_actors.clear()

        # User clicks as larger spheres
        for pt in self._user_clicks:
            sphere = pv.Sphere(radius=0.5, center=pt)
            a = self.app.plotter.add_mesh(sphere, color='red', pickable=False, reset_camera=False)
            self._point_actors.append(a)

        # Dense margin as a single polyline tube (one actor, many segments)
        dense = self.app.state.margin_points
        if len(dense) >= 2:
            arr = np.asarray(dense)
            if self.app.state.margin_loop_closed:
                arr = np.vstack([arr, arr[0]])
            n = len(arr)
            poly = pv.PolyData(arr)
            poly.lines = np.hstack([[n], np.arange(n)])
            tube = poly.tube(radius=0.15)
            a = self.app.plotter.add_mesh(tube, color='red', pickable=False, reset_camera=False)
            self._line_actors.append(a)

        self.app.plotter.render()

    def _refresh_list(self):
        self.point_list.clear()
        for i, pt in enumerate(self._user_clicks):
            self.point_list.addItem(f"{i+1:3d}  ({pt[0]:7.2f}, {pt[1]:7.2f}, {pt[2]:7.2f})")
        n_clicks = len(self._user_clicks)
        n_dense = len(self.app.state.margin_points)
        if self.app.state.margin_loop_closed:
            self.status_label.setText(
                f"Loop: closed ({n_clicks} clicks → {n_dense} points)"
            )
        else:
            self.status_label.setText(
                f"Loop: open ({n_clicks} click{'s' if n_clicks != 1 else ''}"
                + (f" → {n_dense} points)" if self._smart_trace and n_dense != n_clicks else ")")
            )

    def _update_buttons(self):
        n = len(self._user_clicks)
        closed = self.app.state.margin_loop_closed
        self.btn_close.setEnabled(n >= 3 and not closed)
        self.btn_undo.setEnabled(n > 0 or closed)
        self.btn_clear.setEnabled(n > 0)
        self.btn_save.setEnabled(n > 0)

    # --- Persistence ---

    def serialize(self):
        return {
            "smart_trace": bool(self._smart_trace),
            "user_clicks": [c.tolist() for c in self._user_clicks],
        }

    def restore(self, data):
        """Rebuild visuals from app.state.margin_points + saved user_clicks."""
        # Invalidate any cached mesh features — jaw may have changed
        self._cached_mesh_id = None
        self._curvature = None
        self._kdtree = None
        self._adj_geo = None
        self._adj_ridge = None

        # Prefer the saved user_clicks; fall back to treating the dense curve as clicks
        # (for legacy projects + .npy imports that don't track click sources)
        if "user_clicks" in data and data["user_clicks"]:
            self._user_clicks = [np.asarray(c) for c in data["user_clicks"]]
        elif self.app.state.margin_points:
            self._user_clicks = [np.asarray(p) for p in self.app.state.margin_points]
        else:
            self._user_clicks = []

        self._smart_trace = bool(data.get("smart_trace", False))
        self.btn_smart.setChecked(self._smart_trace)
        self.btn_smart.setText(
            "Disable Ridge Tracing" if self._smart_trace else "Enable Ridge Tracing"
        )

        # Re-derive the prep isolation from the first saved click — deterministic,
        # avoids serialising the sub-mesh into the project file.
        self._exit_focus_view()
        self.app.state.prep_mesh = None
        if self._user_clicks and self.app.state.jaw_mesh is not None:
            self._try_isolate_prep(self._user_clicks[0])

        self._refresh_list()
        self._redraw_visualization()
        self._update_buttons()
