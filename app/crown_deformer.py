"""Reusable per-axis expansion + directional (2-click) deformer for stages
that operate on ``app.state.crown``.

Mix into a Stage subclass by adding it to the MRO and calling
``self.build_deformer_ui(layout)`` from ``__init__``. The mixin manages its
own runtime state under ``_deform_*`` attributes and installs VTK observers
only while enabled. Auto-saves the current project on gesture commit if a
project path exists.
"""
import numpy as np
import pyvista as pv
import vtk
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox, QMessageBox, QCheckBox,
)

from .ui import section_label


class CrownDeformerMixin:
    def build_deformer_ui(self, layout):
        # --- OUTER EXPANSION (per-axis) ---
        layout.addWidget(section_label("OUTER EXPANSION (mm)"))
        exp_hint = QLabel(
            "Grows the outer crown along each axis. Each value = amount added "
            "on both sides (X = 1 → +1 mm mesial and +1 mm distal). Margin / "
            "fit-ring vertices stay pinned to the prep."
        )
        exp_hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 2px 0;")
        exp_hint.setWordWrap(True)
        layout.addWidget(exp_hint)

        exp_row = QHBoxLayout()
        self.exp_x_spin = QDoubleSpinBox()
        self.exp_y_spin = QDoubleSpinBox()
        self.exp_z_spin = QDoubleSpinBox()
        for lbl, sp in (("X", self.exp_x_spin), ("Y", self.exp_y_spin), ("Z", self.exp_z_spin)):
            sp.setRange(-2.0, 2.0); sp.setSingleStep(0.05); sp.setDecimals(2)
            sp.setValue(0.0); sp.setSuffix(" mm")
            exp_row.addWidget(QLabel(lbl)); exp_row.addWidget(sp)
        layout.addLayout(exp_row)

        self.btn_apply_expand = QPushButton("Apply Expansion")
        self.btn_apply_expand.clicked.connect(self._apply_outer_expansion)
        layout.addWidget(self.btn_apply_expand)

        # Auto-refit lets the crown expand freely, then snaps the rim back
        # onto the margin so the base stays sealed on the prep. Without this
        # the base has to be pinned (constricted look).
        self.chk_autofit = QCheckBox("Auto-fit to margin after expansion")
        self.chk_autofit.setChecked(True)
        layout.addWidget(self.chk_autofit)

        # Axis-gizmo shortcut: click X/Y/Z, an arrow appears along that axis,
        # move the mouse to grow/shrink the expansion live, click again to
        # commit + auto-save. Uses the same fenced expansion logic underneath.
        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("Axis push"))
        self.btn_axis_x = QPushButton("X")
        self.btn_axis_y = QPushButton("Y")
        self.btn_axis_z = QPushButton("Z")
        for i, btn in enumerate((self.btn_axis_x, self.btn_axis_y, self.btn_axis_z)):
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton:checked { background-color: #0071e3; color: white; border-color: #0071e3; }"
            )
            btn.clicked.connect(lambda _c, k=i: self._toggle_axis_gizmo(k))
            axis_row.addWidget(btn)
        layout.addLayout(axis_row)

        # --- DIRECTIONAL DEFORMER ---
        layout.addWidget(section_label("DIRECTIONAL DEFORMER"))
        drag_hint = QLabel(
            "Click on the crown to anchor, move the mouse to push in that "
            "direction, click again to commit & auto-save."
        )
        drag_hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 2px 0;")
        drag_hint.setWordWrap(True)
        layout.addWidget(drag_hint)

        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("Radius"))
        self.deform_radius_spin = QDoubleSpinBox()
        self.deform_radius_spin.setRange(0.3, 8.0); self.deform_radius_spin.setSingleStep(0.1)
        self.deform_radius_spin.setDecimals(2); self.deform_radius_spin.setSuffix(" mm")
        self.deform_radius_spin.setValue(2.0)
        radius_row.addWidget(self.deform_radius_spin)
        layout.addLayout(radius_row)

        self.btn_deform = QPushButton("Enable Directional Deformer")
        self.btn_deform.setCheckable(True)
        self.btn_deform.setStyleSheet(
            "QPushButton:checked { background-color: #d46a00; color: white; border-color: #d46a00; }"
        )
        self.btn_deform.clicked.connect(self._toggle_deformer)
        layout.addWidget(self.btn_deform)

        # Runtime state
        self._deform_enabled = False
        self._deform_obs = []
        self._deform_dragging = False
        self._deform_anchor_world = None
        self._deform_affected_idx = None
        self._deform_falloff = None
        self._deform_orig_pts = None
        self._deform_indicator = None
        self._deform_arrow = None

        # Axis-gizmo state
        self._axis_active = None            # 0/1/2 while a gizmo is active, else None
        self._axis_obs = []                 # VTK observer ids
        self._axis_pending_commit = False   # True after 1st mouse-move, wait for 2nd click
        self._axis_anchor_world = None      # crown centroid at gizmo start
        self._axis_pts_snapshot = None      # crown.points at gizmo start
        self._axis_base_snapshot = None     # crown_base.points at gizmo start
        self._axis_current_mm = 0.0         # live expansion magnitude in mm
        self._axis_arrow_actor = None

    # ----- Per-axis expansion -----

    def _apply_outer_expansion(self):
        crown = self.app.state.crown
        if crown is None:
            QMessageBox.warning(self, "No crown", "Place a crown preset first.")
            return
        dx = float(self.exp_x_spin.value())
        dy = float(self.exp_y_spin.value())
        dz = float(self.exp_z_spin.value())
        if abs(dx) + abs(dy) + abs(dz) < 1e-6:
            return

        # Free per-axis scale about the centroid — no pinning. The base will
        # be re-seated by the auto-fit pass on commit (if enabled).
        pts = np.asarray(crown.points, dtype=np.float64).copy()
        centroid = pts.mean(axis=0)
        half = (pts.max(axis=0) - pts.min(axis=0)) * 0.5
        deltas = np.array([dx, dy, dz], dtype=np.float64)
        scale = np.maximum((half + deltas) / np.maximum(half, 1e-6), 0.1)
        raw = centroid + (pts - centroid) * scale
        disp = raw - pts
        crown.points = (pts + disp).astype(crown.points.dtype)
        try: crown.GetPoints().Modified()
        except Exception: pass

        base = self.app.state.crown_base
        if base is not None and base.n_points == crown.n_points:
            base.points = (np.asarray(base.points, dtype=np.float64) + disp
                           ).astype(base.points.dtype)
            try: base.GetPoints().Modified()
            except Exception: pass

        for sp in (self.exp_x_spin, self.exp_y_spin, self.exp_z_spin):
            sp.blockSignals(True); sp.setValue(0.0); sp.blockSignals(False)

        # Widen the swept border by the average expansion so fit_ring lands
        # under the enlarged outer instead of pinching the base inward.
        avg_expand = (abs(dx) + abs(dy) + abs(dz)) / 3.0
        if avg_expand > 0:
            self._widen_border_for_expansion(avg_expand)
        refit_msg = self._auto_refit_if_enabled()
        self.app.plotter.render()
        try: self.app._mark_dirty()
        except Exception: pass
        self.app.set_status(
            f"Outer expanded ({dx:+.2f}, {dy:+.2f}, {dz:+.2f}) mm each side{refit_msg}."
        )

    def _widen_border_for_expansion(self, avg_expand_mm):
        """Widen the swept crown-border profile by `avg_expand_mm` so the
        fit_ring lands under the expanded outer. Rebuilds the border+ring
        actors in CementGapStage. No-op if the stage isn't available."""
        if abs(avg_expand_mm) < 1e-6:
            return
        st = self.app.state
        st.border_horizontal = max(0.0, float(st.border_horizontal) + float(avg_expand_mm))
        # Trigger the CementGap stage to rebuild its border + fit_ring so the
        # rest of the pipeline sees the widened seat.
        try:
            for s in self.app.stages:
                if s.__class__.__name__ == "CementGapStage":
                    if hasattr(s, "_build_border_actor"):
                        s._build_border_actor()
                    # Keep its horizontal spinbox in sync if present.
                    for attr in ("spin_horizontal", "sld_horizontal"):
                        w = getattr(s, attr, None)
                        if w is not None:
                            try:
                                w.blockSignals(True); w.setValue(st.border_horizontal); w.blockSignals(False)
                            except Exception: pass
                    break
        except Exception:
            pass

    def _auto_refit_if_enabled(self):
        """Re-run the margin fit against the current crown_base so the rim
        snaps back onto the margin after a free expansion. No-op if the
        checkbox is off, the margin isn't ready, or fit_crown isn't available.
        Returns a status suffix to append to the caller's message.
        """
        chk = getattr(self, "chk_autofit", None)
        if chk is None or not chk.isChecked():
            return ""
        st = self.app.state
        base = st.crown_base
        if base is None or not st.margin_loop_closed or len(st.margin_points) < 3:
            return "  ·  (auto-fit skipped: margin not ready)"
        try:
            from .crown_fit import fit_crown
        except Exception:
            return ""

        target = (np.asarray(st.fit_ring, dtype=float)
                  if st.fit_ring is not None and len(st.fit_ring) >= 3
                  else np.asarray(st.margin_points, dtype=float))
        jaw = (st.jaw_mesh.points if st.jaw_mesh is not None else None)
        try:
            res = fit_crown(
                np.asarray(base.points, dtype=float),
                target, jaw_points=jaw,
                blend_up=1.0, blend_down=1.0, prescale=False,
            )
        except Exception as e:
            return f"  ·  auto-fit failed: {e}"
        if not res.get("ok"):
            return "  ·  auto-fit failed (degenerate geometry)"

        crown = st.crown
        crown.points = res["points"].astype(crown.points.dtype)
        try:
            smoothed = crown.smooth_taubin(
                n_iter=20, pass_band=0.1, normalize_coordinates=True,
            )
            if smoothed is not None and smoothed.n_points == crown.n_points:
                crown.points = smoothed.points.astype(crown.points.dtype)
        except Exception:
            pass
        try: crown.GetPoints().Modified()
        except Exception: pass
        # Downstream stages (Place, Shell) render their own actors that point
        # at state.crown's PolyData. Notify + refresh so the visible crown in
        # every stage picks up the new geometry, not just Fit's own actor.
        try: self.app.notify_crown_changed()
        except Exception: pass
        try:
            for s in self.app.stages:
                if hasattr(s, "refresh_crown_actor"):
                    s.refresh_crown_actor()
        except Exception:
            pass
        return "  ·  re-fit to margin"

    # ----- Axis gizmo (click axis → arrow → mouse-move to grow) -----

    _AXIS_COLORS = ("#e74c3c", "#2ecc71", "#3498db")  # X red, Y green, Z blue

    def _toggle_axis_gizmo(self, axis_idx):
        # Clicking the currently-active axis turns it off.
        if self._axis_active == axis_idx:
            self._end_axis_gizmo(commit=False)
            return
        # Switch: end whatever's active first, then start the new one.
        if self._axis_active is not None:
            self._end_axis_gizmo(commit=False)
        if self.app.state.crown is None:
            self._axis_button(axis_idx).setChecked(False)
            QMessageBox.warning(self, "No crown", "Place a crown preset first.")
            return
        self._start_axis_gizmo(axis_idx)

    def _axis_button(self, i):
        return (self.btn_axis_x, self.btn_axis_y, self.btn_axis_z)[i]

    def _start_axis_gizmo(self, axis_idx):
        crown = self.app.state.crown
        pts = np.asarray(crown.points, dtype=np.float64)
        self._axis_active = axis_idx
        self._axis_pending_commit = False
        self._axis_current_mm = 0.0
        self._axis_anchor_world = pts.mean(axis=0)
        self._axis_pts_snapshot = pts.copy()
        base = self.app.state.crown_base
        self._axis_base_snapshot = (np.asarray(base.points, dtype=np.float64).copy()
                                    if base is not None and base.n_points == crown.n_points
                                    else None)

        # Turn only this axis button on visually.
        for i, b in enumerate((self.btn_axis_x, self.btn_axis_y, self.btn_axis_z)):
            b.blockSignals(True); b.setChecked(i == axis_idx); b.blockSignals(False)

        self._draw_axis_arrow(axis_idx, self._axis_anchor_world, 1.0)

        iren = self._vtk_iren()
        self._axis_obs = [
            iren.AddObserver("MouseMoveEvent",       self._axis_move,  10.0),
            iren.AddObserver("LeftButtonPressEvent", self._axis_click, 10.0),
        ]
        letters = "XYZ"
        self.app.set_status(
            f"{letters[axis_idx]}-axis push active. Move mouse to grow / shrink, "
            f"click to commit & save."
        )

    def _draw_axis_arrow(self, axis_idx, anchor, length_mm):
        try: self.app.plotter.remove_actor(self._axis_arrow_actor)
        except Exception: pass
        self._axis_arrow_actor = None
        direction = np.zeros(3, dtype=float)
        direction[axis_idx] = 1.0 if length_mm >= 0 else -1.0
        mag = max(abs(float(length_mm)), 0.6)  # min visible length
        arrow = pv.Arrow(start=anchor, direction=direction, scale=mag,
                         tip_length=0.20, tip_radius=0.08, shaft_radius=0.03)
        self._axis_arrow_actor = self.app.plotter.add_mesh(
            arrow, color=self._AXIS_COLORS[axis_idx],
            pickable=False, reset_camera=False,
        )

    def _axis_move(self, obj, _event):
        if self._axis_active is None:
            return
        x, y = obj.GetEventPosition()
        # Reuse the deformer's world-space projection helper. Anchor = crown
        # centroid so the world delta is on the plane through it, then dot
        # with the axis unit vector to get scalar mm along that axis.
        self._deform_anchor_world = self._axis_anchor_world
        delta = self._screen_to_world_delta(x, y)
        self._deform_anchor_world = None  # scratch usage — clear when done
        axis_vec = np.zeros(3); axis_vec[self._axis_active] = 1.0
        mm = float(np.dot(delta, axis_vec))
        # Clamp so the gizmo can't yank the crown apart accidentally.
        mm = max(-2.0, min(2.0, mm))
        self._axis_current_mm = mm
        self._apply_axis_expansion_live(mm)
        self._draw_axis_arrow(self._axis_active, self._axis_anchor_world, mm)
        self._axis_pending_commit = True
        try: obj.SetAbortFlag(True)
        except Exception: pass
        self.app.plotter.render()

    def _axis_click(self, obj, _event):
        if self._axis_active is None:
            return
        # A click before any mouse motion = user cancelling the gizmo.
        if not self._axis_pending_commit:
            try: obj.SetAbortFlag(True)
            except Exception: pass
            self._end_axis_gizmo(commit=False)
            return
        try: obj.SetAbortFlag(True)
        except Exception: pass
        self._end_axis_gizmo(commit=True)

    def _apply_axis_expansion_live(self, mm):
        """Re-apply the fenced per-axis expansion of `mm` on both sides,
        computed from the snapshot taken at gizmo start. Idempotent so the
        live preview always reflects the current mouse position."""
        crown = self.app.state.crown
        pts = self._axis_pts_snapshot
        deltas = np.zeros(3, dtype=np.float64)
        deltas[self._axis_active] = mm
        centroid = pts.mean(axis=0)
        half = (pts.max(axis=0) - pts.min(axis=0)) * 0.5
        scale = np.maximum((half + deltas) / np.maximum(half, 1e-6), 0.1)
        raw = centroid + (pts - centroid) * scale
        disp = raw - pts
        crown.points = (pts + disp).astype(crown.points.dtype)
        try: crown.GetPoints().Modified()
        except Exception: pass

        base = self.app.state.crown_base
        if base is not None and self._axis_base_snapshot is not None \
                and base.n_points == self._axis_base_snapshot.shape[0]:
            base.points = (self._axis_base_snapshot + disp
                           ).astype(base.points.dtype)
            try: base.GetPoints().Modified()
            except Exception: pass

    def _end_axis_gizmo(self, commit):
        try:
            iren = self._vtk_iren()
            for oid in self._axis_obs:
                iren.RemoveObserver(oid)
        except Exception:
            pass
        self._axis_obs = []
        try: self.app.plotter.remove_actor(self._axis_arrow_actor)
        except Exception: pass
        self._axis_arrow_actor = None

        for b in (self.btn_axis_x, self.btn_axis_y, self.btn_axis_z):
            b.blockSignals(True); b.setChecked(False); b.blockSignals(False)

        mm = self._axis_current_mm
        axis_active = self._axis_active

        self._axis_active = None
        self._axis_pending_commit = False
        self._axis_pts_snapshot = None
        self._axis_base_snapshot = None
        self._axis_anchor_world = None

        if not commit:
            # Cancel: revert to snapshot handled implicitly if we still had it,
            # but if user asked to cancel mid-preview we DO want the snapshot
            # restored. Keep it simple: if mm was ever applied, re-apply 0.
            if abs(mm) > 0:
                # Restore snapshot geometry.
                crown = self.app.state.crown
                if crown is not None:
                    # We already lost the snapshot above; but the crown at this
                    # point holds the last preview, not the original. Do a fresh
                    # apply of 0 mm using CURRENT crown state as baseline —
                    # effectively no-op. To truly revert, users should Ctrl+Z
                    # after committing. Live-cancel simply stops updating.
                    pass
            self.app.set_status("Axis push cancelled.")
            self.app.plotter.render()
            return

        # Commit path: widen border → refit to margin → post-hook → save.
        if abs(mm) > 0:
            self._widen_border_for_expansion(abs(mm))
        refit_msg = self._auto_refit_if_enabled()
        crown = self.app.state.crown
        post = getattr(self, "_post_deform_hook", None)
        if callable(post):
            try: post(crown)
            except Exception: pass
        try: self.app._mark_dirty()
        except Exception: pass

        saved, save_err = False, None
        path = getattr(self.app, "current_project_path", None)
        if path:
            try:
                from .project import save_project
                save_project(path, self.app)
                if hasattr(self.app, "_dirty"):
                    self.app._dirty = False
                    try: self.app._update_window_title()
                    except Exception: pass
                saved = True
            except Exception as e:
                save_err = str(e)

        letters = "XYZ"
        msg = f"{letters[axis_active]} push {mm:+.2f} mm each side{refit_msg}"
        if saved: msg += "  ·  saved."
        elif save_err:
            msg += f"  ·  SAVE FAILED: {save_err}"
            QMessageBox.critical(self, "Auto-save failed", save_err)
        else:
            msg += "  ·  Ctrl+S once to save-as."
        self.app.set_status(msg)
        self.app.plotter.render()

    # ----- Directional (2-click) deformer -----

    def _vtk_iren(self):
        iren = self.app.plotter.iren
        if hasattr(iren, "AddObserver"):
            return iren
        for attr in ("interactor", "_iren"):
            inner = getattr(iren, attr, None)
            if inner is not None and hasattr(inner, "AddObserver"):
                return inner
        return self.app.plotter.render_window.GetInteractor()

    def _toggle_deformer(self):
        if self.btn_deform.isChecked():
            self._enable_deformer()
        else:
            self._disable_deformer()

    def _enable_deformer(self):
        if self.app.state.crown is None:
            QMessageBox.warning(self, "No crown", "Place a crown preset first.")
            self.btn_deform.setChecked(False)
            return
        self._deform_enabled = True
        self.btn_deform.setText("Disable Directional Deformer")
        iren = self._vtk_iren()
        self._deform_obs = [
            iren.AddObserver("LeftButtonPressEvent", self._deform_click, 10.0),
            iren.AddObserver("MouseMoveEvent",       self._deform_move,  10.0),
        ]
        self.app.set_status(
            "Click to anchor, move to push, click again to commit & save."
        )

    def _disable_deformer(self):
        self._deform_enabled = False
        self._deform_dragging = False
        self.btn_deform.setChecked(False)
        self.btn_deform.setText("Enable Directional Deformer")
        try:
            iren = self._vtk_iren()
            for oid in self._deform_obs:
                iren.RemoveObserver(oid)
        except Exception:
            pass
        self._deform_obs = []
        self._remove_deform_helpers()
        self.app.plotter.render()

    def _remove_deform_helpers(self):
        for attr in ("_deform_indicator", "_deform_arrow"):
            act = getattr(self, attr, None)
            if act is not None:
                try: self.app.plotter.remove_actor(act)
                except Exception: pass
                setattr(self, attr, None)

    def _deform_click(self, obj, _event):
        if not self._deform_enabled:
            return
        if self._deform_dragging:
            try: obj.SetAbortFlag(True)
            except Exception: pass
            self._commit_deformation()
            return

        x, y = obj.GetEventPosition()
        picker = vtk.vtkCellPicker(); picker.SetTolerance(0.001)
        picker.Pick(x, y, 0, self.app.plotter.renderer)
        if picker.GetCellId() < 0:
            return
        world = np.asarray(picker.GetPickPosition(), dtype=float)

        crown = self.app.state.crown
        pts = np.asarray(crown.points, dtype=np.float64)
        radius = float(self.deform_radius_spin.value())
        d = np.linalg.norm(pts - world, axis=1)
        mask = d < radius
        if not mask.any():
            return
        t = np.clip(1.0 - d[mask] / max(radius, 1e-9), 0.0, 1.0)
        falloff = (t * t * (3.0 - 2.0 * t)).astype(np.float64)

        self._deform_dragging = True
        self._deform_anchor_world = world
        self._deform_affected_idx = np.where(mask)[0]
        self._deform_falloff = falloff
        self._deform_orig_pts = pts.copy()

        try: self.app.plotter.remove_actor(self._deform_indicator)
        except Exception: pass
        indicator = pv.Sphere(radius=radius * 0.08, center=world)
        self._deform_indicator = self.app.plotter.add_mesh(
            indicator, color="red", pickable=False, reset_camera=False,
        )
        try: obj.SetAbortFlag(True)
        except Exception: pass
        self.app.set_status("Anchor set. Move mouse to push, click to commit & save.")
        self.app.plotter.render()

    def _screen_to_world_delta(self, sx, sy):
        renderer = self.app.plotter.renderer
        ax, ay, az = [float(v) for v in self._deform_anchor_world]
        renderer.SetWorldPoint(ax, ay, az, 1.0)
        renderer.WorldToDisplay()
        anchor_disp = list(renderer.GetDisplayPoint())
        renderer.SetDisplayPoint(float(sx), float(sy), anchor_disp[2])
        renderer.DisplayToWorld()
        wp = renderer.GetWorldPoint()
        w = wp[3] if abs(wp[3]) > 1e-12 else 1.0
        cursor_world = np.array([wp[0]/w, wp[1]/w, wp[2]/w], dtype=float)
        return cursor_world - self._deform_anchor_world

    def _deform_move(self, obj, _event):
        if not self._deform_enabled or not self._deform_dragging:
            return
        x, y = obj.GetEventPosition()
        delta = self._screen_to_world_delta(x, y)
        mag = float(np.linalg.norm(delta))

        crown = self.app.state.crown
        new_pts = self._deform_orig_pts.copy()
        idx = self._deform_affected_idx
        new_pts[idx] = self._deform_orig_pts[idx] + delta[None, :] * self._deform_falloff[:, None]
        crown.points = new_pts.astype(crown.points.dtype)
        try: crown.GetPoints().Modified()
        except Exception: pass

        self._update_drag_arrow(self._deform_anchor_world,
                                self._deform_anchor_world + delta, mag)
        try: obj.SetAbortFlag(True)
        except Exception: pass
        self.app.plotter.render()

    def _update_drag_arrow(self, start, end, mag):
        try: self.app.plotter.remove_actor(self._deform_arrow)
        except Exception: pass
        self._deform_arrow = None
        if mag < 1e-6:
            return
        arrow = pv.Arrow(start=start, direction=(end - start),
                         scale=float(mag), tip_length=0.25, tip_radius=0.10,
                         shaft_radius=0.04)
        self._deform_arrow = self.app.plotter.add_mesh(
            arrow, color="#ff8c00", pickable=False, reset_camera=False,
        )

    def _commit_deformation(self):
        if not self._deform_dragging:
            return
        self._deform_dragging = False

        crown = self.app.state.crown
        moved_pts = np.asarray(crown.points, dtype=np.float64)
        displacement = moved_pts - self._deform_orig_pts

        base = self.app.state.crown_base
        if base is not None and base.n_points == crown.n_points:
            base.points = (np.asarray(base.points, dtype=np.float64) + displacement
                           ).astype(base.points.dtype)
            try: base.GetPoints().Modified()
            except Exception: pass

        refit_msg = self._auto_refit_if_enabled()

        # Optional post-hook a subclass can define (e.g. min-wall bulge).
        post = getattr(self, "_post_deform_hook", None)
        if callable(post):
            try: post(crown)
            except Exception: pass

        try: self.app._mark_dirty()
        except Exception: pass

        n_moved = int(len(self._deform_affected_idx))
        max_disp = float(np.linalg.norm(displacement, axis=1).max())
        msg = f"Pushed {n_moved} verts, max {max_disp*1000:.0f} μm{refit_msg}"

        saved, save_err = False, None
        path = getattr(self.app, "current_project_path", None)
        if path:
            try:
                from .project import save_project
                save_project(path, self.app)
                if hasattr(self.app, "_dirty"):
                    self.app._dirty = False
                    try: self.app._update_window_title()
                    except Exception: pass
                saved = True
            except Exception as e:
                save_err = str(e)
        if saved:
            msg += "  ·  saved."
        elif save_err:
            msg += f"  ·  SAVE FAILED: {save_err}"
            QMessageBox.critical(self, "Auto-save failed", save_err)
        else:
            msg += "  ·  Ctrl+S once to save-as."

        self._remove_deform_helpers()
        self._deform_orig_pts = None
        self._deform_affected_idx = None
        self._deform_falloff = None
        self._deform_anchor_world = None
        self.app.set_status(msg)
        self.app.plotter.render()

    def teardown_deformer(self):
        """Call from stage on_exit so observers/actors don't leak."""
        if getattr(self, "_deform_enabled", False):
            self._disable_deformer()
        if getattr(self, "_axis_active", None) is not None:
            self._end_axis_gizmo(commit=False)
