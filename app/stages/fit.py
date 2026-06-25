"""Stage 4 — fit the crown's outer surface onto the margin line.

The Place stage drops a library crown roughly over the prep. This stage takes
that posed crown and fits it to the margin in two tiers:

  1. Rigid seat  — position / orient / uniformly scale the whole crown so its
     cervical edge best matches the margin loop. (Step 2 of the build.)
  2. Drape       — non-rigidly warp the cervical region so the rim lands exactly
     on the margin, fading the deformation to zero higher up so the occlusal
     anatomy is preserved. (Step 3 of the build.)

To stay idempotent (no compounding warp artifacts) the fit always works from an
undeformed snapshot of the posed crown — `self._base`. Manual nudges modify the
base; "Fit to Margin" re-derives the fitted mesh from scratch and writes it back
into `app.state.crown` so the downstream Shell/Trim/Refine stages consume it.

This file is currently the **scaffold** (build step 1): it shows the crown,
supports manual nudging, and has the Fit button wired to a placeholder. The
rigid-seat and drape algorithms land in the next steps.
"""
import os
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QSlider,
    QCheckBox,
)

from ..config import STAGES
from ..ui import section_label
from ..crown_fit import fit_crown
from .base import Stage


class FitStage(Stage):
    name = "Fit"
    description = STAGES[3][1]

    def __init__(self, app):
        super().__init__(app)
        self.crown_actor = None
        self._fitted = False       # has a fit been applied at the current pose?
        self.move_step = 0.5
        self.rotate_step = 5.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # --- FIT ---
        layout.addWidget(section_label("FIT TO MARGIN"))
        hint = QLabel(
            "Seat the crown on the margin, then drape its cervical edge onto "
            "the red margin loop. Nudge the crown first if the gross pose is off."
        )
        hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 4px 0;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.btn_fit = QPushButton("Fit to Margin")
        self.btn_fit.setObjectName("primary")
        self.btn_fit.clicked.connect(self._fit)
        layout.addWidget(self.btn_fit)

        self.chk_prescale = QCheckBox("Pre-scale to ring size")
        self.chk_prescale.setChecked(True)
        layout.addWidget(self.chk_prescale)

        blend_row = QHBoxLayout()
        self.lbl_blend = QLabel("Blend up: 2.5 mm")
        self.lbl_blend.setStyleSheet("color: #424245; font-size: 12px;")
        blend_row.addWidget(self.lbl_blend)
        blend_row.addStretch()
        layout.addLayout(blend_row)
        self.sld_blend = QSlider(Qt.Horizontal)
        self.sld_blend.setRange(5, 80)   # 0.5 .. 8.0 mm, /10
        self.sld_blend.setValue(25)
        self.sld_blend.valueChanged.connect(
            lambda v: self.lbl_blend.setText(f"Blend up: {v/10:.1f} mm"))
        self.sld_blend.sliderReleased.connect(self._on_blend_released)
        layout.addWidget(self.sld_blend)

        self.btn_reset = QPushButton("Reset (undo fit)")
        self.btn_reset.clicked.connect(self._reset_fit)
        layout.addWidget(self.btn_reset)

        # --- MANUAL NUDGE (feeds the base pose) ---
        layout.addWidget(section_label("POSITION (X / Y)"))
        grid = QGridLayout()
        btn_fwd = QPushButton("Fwd")
        btn_back = QPushButton("Back")
        btn_left = QPushButton("Left")
        btn_right = QPushButton("Right")
        btn_fwd.clicked.connect(lambda: self.translate(0, 1, 0))
        btn_back.clicked.connect(lambda: self.translate(0, -1, 0))
        btn_left.clicked.connect(lambda: self.translate(-1, 0, 0))
        btn_right.clicked.connect(lambda: self.translate(1, 0, 0))
        grid.addWidget(btn_fwd, 0, 1)
        grid.addWidget(btn_left, 1, 0)
        grid.addWidget(btn_right, 1, 2)
        grid.addWidget(btn_back, 2, 1)
        layout.addLayout(grid)

        layout.addWidget(section_label("HEIGHT (Z)"))
        zh = QHBoxLayout()
        btn_up = QPushButton("Up")
        btn_down = QPushButton("Down")
        btn_up.clicked.connect(lambda: self.translate(0, 0, 1))
        btn_down.clicked.connect(lambda: self.translate(0, 0, -1))
        zh.addWidget(btn_up)
        zh.addWidget(btn_down)
        layout.addLayout(zh)

        layout.addWidget(section_label("ROTATION"))
        for axis_name, axis_key in [("Pitch (X)", 'x'), ("Roll (Y)", 'y'), ("Yaw (Z)", 'z')]:
            row = QHBoxLayout()
            lbl = QLabel(axis_name)
            lbl.setStyleSheet("color: #424245; font-size: 12px;")
            row.addWidget(lbl)
            row.addStretch()
            btn_neg = QPushButton("−")
            btn_pos = QPushButton("+")
            btn_neg.setFixedWidth(40)
            btn_pos.setFixedWidth(40)
            btn_neg.clicked.connect(lambda _, a=axis_key: self.rotate(a, -1))
            btn_pos.clicked.connect(lambda _, a=axis_key: self.rotate(a, 1))
            row.addWidget(btn_neg)
            row.addWidget(btn_pos)
            layout.addLayout(row)

        # --- STATUS ---
        self.status = QLabel("")
        self.status.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 4px 0;")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        layout.addStretch()

    # ----- Stage lifecycle -----

    def is_complete(self):
        # For now, "complete" once there is a crown to carry forward. Tightens
        # to "fit applied" once the drape lands.
        return self.app.state.crown is not None

    def on_enter(self):
        crown = self.app.state.crown
        if crown is None:
            self.status.setText(
                "No crown to fit. Go back to the Place step and load a crown first."
            )
            self._clear_actor()
            return
        # Hide Place's undeformed crown so it doesn't mask our fitted one.
        self._set_place_crown_visible(False)
        # The undeformed posed crown lives in app.state.crown_base, kept in sync
        # with the placement by the Place stage. Fall back to the current crown
        # if it somehow wasn't set (e.g. a restored project).
        if self.app.state.crown_base is None:
            self.app.state.crown_base = crown.copy()
        if self._margin_ready():
            # Conform immediately from the latest placement so the stage always
            # shows the crown fitted at the current Place pose.
            self._fit()
        else:
            self.status.setText("Close the margin loop (step 1) before fitting.")
            self._redraw()
        self.app.set_status(self.description)

    @property
    def _base(self):
        return self.app.state.crown_base

    def on_exit(self):
        self._clear_actor()
        # Hand the fitted crown to Place's actor so downstream stages (Shell,
        # Trim) display the deformed crown, not Place's original undeformed mesh.
        ps = self._place_stage()
        if ps is not None:
            try: ps.refresh_crown_actor()
            except Exception: pass
        self._set_place_crown_visible(True)

    def _place_stage(self):
        """The PlaceStage instance, which owns its own (undeformed) crown actor."""
        from .place import PlaceStage
        for s in self.app.stages:
            if isinstance(s, PlaceStage):
                return s
        return None

    def _set_place_crown_visible(self, visible):
        """Hide/show Place's crown actor. Place doesn't remove it on exit, so
        without this its undeformed crown overlaps (and visually masks) our
        fitted crown."""
        ps = self._place_stage()
        if ps is not None:
            try: ps.set_outer_visible(visible)
            except Exception: pass

    # ----- Manual nudge (modifies the base pose) -----

    def translate(self, x, y, z):
        if self._base is None:
            return
        d = self.move_step
        self._base.translate([x * d, y * d, z * d], inplace=True)
        self._commit_base()

    def rotate(self, axis, sign):
        if self._base is None:
            return
        a = self.rotate_step * sign
        c = self._base.center
        if axis == 'x':
            self._base.rotate_x(a, point=c, inplace=True)
        elif axis == 'y':
            self._base.rotate_y(a, point=c, inplace=True)
        else:
            self._base.rotate_z(a, point=c, inplace=True)
        self._commit_base()

    def _commit_base(self):
        """A manual nudge changed the base pose. If a fit is already applied,
        re-run it from the new (undeformed) base so the preview stays conformed;
        otherwise just preview the undeformed base."""
        if self._fitted:
            self._fit()
        else:
            self.app.state.crown = self._base.copy()
            self._redraw()
            self.app.notify_crown_changed()

    # ----- Fit: conform the crown cross-section onto the margin ring -----

    def _margin_ready(self):
        return (self.app.state.margin_loop_closed
                and len(self.app.state.margin_points) >= 3)

    def _fit(self):
        """Deform a copy of the undeformed base so its cross-section at the
        margin ring matches the ring exactly, then write the result to
        app.state.crown for the downstream stages."""
        if self._base is None:
            return
        if not self._margin_ready():
            self.status.setText("Close the margin loop (step 1) before fitting.")
            return

        margin = np.asarray(self.app.state.margin_points, dtype=float)
        jaw = (self.app.state.jaw_mesh.points
               if self.app.state.jaw_mesh is not None else None)
        blend_up = self.sld_blend.value() / 10.0

        res = fit_crown(
            np.asarray(self._base.points, dtype=float),
            margin, jaw_points=jaw,
            blend_up=blend_up, blend_down=1.0,
            prescale=self.chk_prescale.isChecked(),
        )
        if not res["ok"]:
            self.status.setText("Fit failed — crown or margin geometry is degenerate.")
            return

        fitted = self._base.copy()
        fitted.points = res["points"].astype(self._base.points.dtype)
        self.app.state.crown = fitted
        self._fitted = True

        # Debug dump (set NUDENT_FIT_DEBUG=1) — saves the exact geometry this fit
        # ran on so the failing case can be reproduced offline.
        if os.environ.get("NUDENT_FIT_DEBUG"):
            try:
                d = os.environ.get("NUDENT_FIT_DEBUG_DIR", "/tmp/nudent_fit_debug")
                os.makedirs(d, exist_ok=True)
                np.save(f"{d}/margin.npy", margin)
                self._base.save(f"{d}/base.stl")
                fitted.save(f"{d}/fitted.stl")
                if self.app.state.jaw_mesh is not None:
                    self.app.state.jaw_mesh.save(f"{d}/jaw.stl")
                self.status.setText(f"[debug] dumped fit geometry to {d}")
            except Exception as e:
                print("fit debug dump failed:", e)
        self._redraw()
        self.app.notify_crown_changed()
        self.completion_changed.emit()
        self.status.setText(
            f"Fitted to margin (pre-scale ×{res['scale']:.2f}, blend {blend_up:.1f} mm). "
            "Nudge to adjust — it re-fits automatically."
        )

    def _on_blend_released(self):
        if self._fitted:
            self._fit()

    def _reset_fit(self):
        """Discard the deformation and return to the undeformed posed crown."""
        if self._base is None:
            return
        self._fitted = False
        self.app.state.crown = self._base.copy()
        self._redraw()
        self.app.notify_crown_changed()
        self.status.setText("Fit reset. Crown is back to its preset shape.")

    # ----- Visualization -----

    def _clear_actor(self):
        if self.crown_actor is not None:
            try: self.app.plotter.remove_actor(self.crown_actor)
            except Exception: pass
        self.crown_actor = None

    def _redraw(self):
        self._clear_actor()
        crown = self.app.state.crown
        if crown is None:
            return
        self.crown_actor = self.app.plotter.add_mesh(
            crown, color="gold", reset_camera=False,
        )
        self.app.plotter.render()

    # ----- Persistence -----

    def serialize(self):
        return {
            "move_step": float(self.move_step),
            "rotate_step": float(self.rotate_step),
        }

    def restore(self, data):
        self.move_step = float(data.get("move_step", 0.5))
        self.rotate_step = float(data.get("rotate_step", 5.0))
        self._clear_actor()
        if self.app.state.crown is not None:
            if self.app.state.crown_base is None:
                self.app.state.crown_base = self.app.state.crown.copy()
            self.crown_actor = self.app.plotter.add_mesh(
                self.app.state.crown, color="gold", reset_camera=False,
            )
