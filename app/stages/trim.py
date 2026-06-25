"""Stage 4 — clip the shell against the 3D margin curve.

Real finish lines are wavy 3D curves (dipping interproximally, rising on the
facial/lingual), so a single flat cutting plane through the margin centroid
is wrong: the crown either over-cuts or under-cuts depending on where each
point on the curve sits relative to the plane.

Instead, we use the margin curve itself as a curved cutting surface. For
each crown vertex, compute its signed distance to the nearest margin point
along the insertion axis: positive = above the margin (keep), negative =
below (drop). Feeding that field to `clip_scalar` produces a clean
interpolated cut that follows the margin curve vertex-for-vertex.
"""
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from PyQt5.QtWidgets import (
    QVBoxLayout, QPushButton, QLabel, QMessageBox,
)

from ..config import STAGES
from ..ui import section_label
from .base import Stage
from .place import PlaceStage
from .shell import ShellStage


class TrimStage(Stage):
    name = "Trim"
    description = STAGES[5][1]

    def __init__(self, app):
        super().__init__(app)
        self.trimmed_actor = None
        self.plane_actor = None
        self.axis_actor = None
        self._flip_direction = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # --- VISUALIZE ---
        layout.addWidget(section_label("VISUALIZE"))
        self.btn_show_axis = QPushButton("Show Cut Direction (arrow)")
        self.btn_show_axis.setCheckable(True)
        self.btn_show_axis.clicked.connect(self._toggle_axis)
        layout.addWidget(self.btn_show_axis)

        self.btn_show_plane = QPushButton("Show Cut Plane")
        self.btn_show_plane.setCheckable(True)
        self.btn_show_plane.clicked.connect(self._toggle_plane)
        layout.addWidget(self.btn_show_plane)

        # --- ACTION ---
        layout.addWidget(section_label("ACTION"))
        hint = QLabel(
            "Arrow points toward the side that will be KEPT. "
            "If it points the wrong way (toward the gum instead of the crown), toggle Flip Cut Direction."
        )
        hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 4px 0;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.btn_flip = QPushButton("Flip Cut Direction")
        self.btn_flip.setCheckable(True)
        self.btn_flip.setStyleSheet(
            "QPushButton:checked { background-color: #0071e3; color: white; border-color: #0071e3; }"
        )
        self.btn_flip.clicked.connect(self._on_flip_change)
        layout.addWidget(self.btn_flip)

        self.btn_apply = QPushButton("Apply Trim")
        self.btn_apply.setObjectName("primary")
        self.btn_apply.clicked.connect(self.apply_trim)
        layout.addWidget(self.btn_apply)

        self.btn_revert = QPushButton("Revert Trim")
        self.btn_revert.clicked.connect(self.revert_trim)
        self.btn_revert.setEnabled(False)
        layout.addWidget(self.btn_revert)

        # --- STATUS ---
        layout.addWidget(section_label("STATUS"))
        self.lbl_status = QLabel("Trim not yet applied.")
        self.lbl_status.setStyleSheet("color: #6e6e73; font-size: 12px;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        layout.addStretch()

    # --- Stage lifecycle ---

    def is_complete(self):
        return self.app.state.trimmed_crown is not None

    def on_enter(self):
        # If a trim already exists (returning from Refine, or after Shell peek),
        # restore the trim scene: hide shell layers, show the trimmed crown.
        if self.app.state.trimmed_crown is not None and self.trimmed_actor is not None:
            self._restore_shell_visibility(False)
            self.trimmed_actor.SetVisibility(True)
            # Also hide any downstream solidified-crown actor.
            for s in self.app.stages:
                if hasattr(s, "final_actor") and s.final_actor is not None:
                    s.final_actor.SetVisibility(False)
            self.app.plotter.render()
        self.app.set_status(self.description)

    def on_exit(self):
        # Tidy up visualization helpers so they don't leak into other stages
        if self.btn_show_axis.isChecked():
            self.btn_show_axis.setChecked(False)
            self._toggle_axis()
        if self.btn_show_plane.isChecked():
            self.btn_show_plane.setChecked(False)
            self._toggle_plane()

    def reset_trim(self):
        if self.trimmed_actor is not None:
            try: self.app.plotter.remove_actor(self.trimmed_actor)
            except Exception: pass
        self.trimmed_actor = None
        self.app.state.trimmed_crown = None
        self._restore_shell_visibility(True)
        self.lbl_status.setText("Trim not yet applied.")
        self.btn_revert.setEnabled(False)
        self.completion_changed.emit()
        self.app.plotter.render()

    def revert_trim(self):
        self.reset_trim()
        self.app.set_status("Trim reverted.")

    # --- Geometry ---

    def _compute_axis(self):
        pts = np.array(self.app.state.margin_points)
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[-1] / np.linalg.norm(vh[-1])

        # Orient the axis toward the placed crown's centre. The outer crown
        # sits on the keep side by construction (above the margin), so this is
        # a reliable "up" reference. Falls back to a nearby-jaw-mass heuristic
        # if no crown is available — but in practice this stage is gated on
        # the Place stage being complete.
        crown = self.app.state.crown
        if crown is not None:
            crown_center = np.asarray(crown.center)
            if (crown_center - centroid) @ axis < 0:
                axis = -axis
        else:
            jaw_pts = self.app.state.jaw_mesh.points
            margin_extent = max(np.linalg.norm(pts - centroid, axis=1).max(), 1e-3)
            distances = np.linalg.norm(jaw_pts - centroid, axis=1)
            nearby_mask = distances < margin_extent * 3
            nearby = jaw_pts[nearby_mask] if nearby_mask.any() else jaw_pts
            margin_proj = float(centroid @ axis)
            proj = nearby @ axis
            if int(np.sum(proj > margin_proj)) > int(np.sum(proj < margin_proj)):
                axis = -axis

        # User override: when the heuristic guesses wrong they toggle Flip Cut Direction.
        if self._flip_direction:
            axis = -axis
        return centroid, axis

    def _on_flip_change(self):
        self._flip_direction = self.btn_flip.isChecked()
        # Re-render any visible helpers with the new direction
        if self.btn_show_axis.isChecked():
            self.btn_show_axis.setChecked(False)
            self._toggle_axis()
            self.btn_show_axis.setChecked(True)
            self._toggle_axis()
        if self.btn_show_plane.isChecked():
            self.btn_show_plane.setChecked(False)
            self._toggle_plane()
            self.btn_show_plane.setChecked(True)
            self._toggle_plane()

    def _toggle_axis(self):
        if self.btn_show_axis.isChecked():
            centroid, axis = self._compute_axis()
            # Arrow tip points toward the side that will be KEPT.
            arrow = pv.Arrow(
                start=centroid, direction=axis, tip_length=0.25,
                tip_radius=0.6, shaft_radius=0.2, scale=18.0,
            )
            self.axis_actor = self.app.plotter.add_mesh(
                arrow, color='magenta', pickable=False
            )
        else:
            if self.axis_actor is not None:
                try: self.app.plotter.remove_actor(self.axis_actor)
                except Exception: pass
                self.axis_actor = None
        self.app.plotter.render()

    def _toggle_plane(self):
        if self.btn_show_plane.isChecked():
            centroid, axis = self._compute_axis()
            plane = pv.Plane(center=centroid, direction=axis, i_size=40, j_size=40)
            self.plane_actor = self.app.plotter.add_mesh(
                plane, color='cyan', opacity=0.3, pickable=False
            )
        else:
            if self.plane_actor is not None:
                try: self.app.plotter.remove_actor(self.plane_actor)
                except Exception: pass
                self.plane_actor = None
        self.app.plotter.render()

    # --- Trim ---

    def apply_trim(self):
        outer = self.app.state.shell_outer
        inner = self.app.state.shell_inner
        if outer is None or inner is None:
            QMessageBox.warning(self, "No shell", "Generate the shell in Stage 3 first.")
            return

        # Invalidate any solidified crown built from a previous trim
        self.app.notify_trim_changed()

        if self.trimmed_actor is not None:
            try: self.app.plotter.remove_actor(self.trimmed_actor)
            except Exception: pass
            self.trimmed_actor = None

        centroid, axis = self._compute_axis()
        margin_pts = np.asarray(self.app.state.margin_points)
        if len(margin_pts) < 3:
            QMessageBox.warning(self, "No margin", "Need a closed margin curve to trim.")
            return
        tree = cKDTree(margin_pts)

        def clip_at_margin(mesh):
            """Drop the part of `mesh` that sits below the 3D margin curve.

            For each mesh vertex, the signed distance along the insertion axis
            to its nearest margin point gives the cut field. clip_scalar
            interpolates the cut along triangle edges where the field crosses
            zero — so the cut edge follows the margin curve, not a plane.
            """
            pts = np.asarray(mesh.points)
            _, nearest = tree.query(pts, k=1)
            signed = (pts - margin_pts[nearest]) @ axis
            tmp = mesh.copy()
            tmp["_margin_signed"] = signed
            # invert=True → keep where scalar > value (above the margin).
            clipped = tmp.clip_scalar("_margin_signed", invert=True, value=0.0)
            return clipped.extract_surface()

        outer_trim = clip_at_margin(outer)
        inner_trim = clip_at_margin(inner)

        outer_dropped = outer.n_points - outer_trim.n_points
        inner_dropped = inner.n_points - inner_trim.n_points

        # Merge into one PolyData. Trim edges are still open (no band yet);
        # Stage 5 will close it into a watertight solid.
        combined = outer_trim.merge(inner_trim)
        self.app.state.trimmed_crown = combined

        self._restore_shell_visibility(False)

        self.trimmed_actor = self.app.plotter.add_mesh(
            combined, color='gold', show_edges=False
        )
        self.app.plotter.render()

        if outer_dropped == 0 and inner_dropped == 0:
            self.lbl_status.setText(
                "Trim applied, but no crown geometry sat below the margin plane. "
                "Lower the crown in Stage 2 (Place) so its bottom dips into the prep, then re-apply."
            )
        else:
            self.lbl_status.setText(
                f"Trim applied.\n"
                f"Outer: removed {outer_dropped:,} verts below margin.\n"
                f"Inner: removed {inner_dropped:,} verts below margin.\n"
                f"Trim edges are open — Stage 5 will stitch them into a watertight crown."
            )
        self.btn_revert.setEnabled(True)
        self.completion_changed.emit()
        self.app.set_status("Trim applied. Crown cut at the margin plane.")

    # --- Helpers ---

    def _restore_shell_visibility(self, visible):
        place = self._place_stage()
        if place is not None:
            place.set_outer_visible(visible)
        shell = self._shell_stage()
        if shell is not None:
            if shell.inner_actor is not None:
                shell.inner_actor.SetVisibility(bool(visible))
            shell.btn_show_inner.setChecked(bool(visible))
            shell.btn_show_inner.setText(
                "Hide Inner Surface" if visible else "Show Inner Surface"
            )
            shell.btn_show_outer.setChecked(bool(visible))
            shell.btn_show_outer.setText(
                "Hide Outer Crown" if visible else "Show Outer Crown"
            )

    def _place_stage(self):
        for s in self.app.stages:
            if isinstance(s, PlaceStage):
                return s
        return None

    def _shell_stage(self):
        for s in self.app.stages:
            if isinstance(s, ShellStage):
                return s
        return None

    # --- Persistence ---

    def serialize(self):
        return {"flip_direction": bool(self._flip_direction)}

    def restore(self, data):
        self._flip_direction = bool(data.get("flip_direction", False))
        self.btn_flip.setChecked(self._flip_direction)

        if self.trimmed_actor is not None:
            try: self.app.plotter.remove_actor(self.trimmed_actor)
            except Exception: pass
            self.trimmed_actor = None

        trimmed = self.app.state.trimmed_crown
        if trimmed is not None:
            self._restore_shell_visibility(False)
            self.trimmed_actor = self.app.plotter.add_mesh(
                trimmed, color="gold", show_edges=False
            )
            self.btn_revert.setEnabled(True)
            self.lbl_status.setText(
                f"Trim restored.\n"
                f"{trimmed.n_points:,} verts, {trimmed.n_cells:,} faces.\n"
                f"Trim edges are open — Stage 5 will stitch them into a watertight crown."
            )
