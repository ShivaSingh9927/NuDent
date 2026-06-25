"""Stage 2 — place a crown preset over the margin and transform it into position."""
import os
import numpy as np
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QSpinBox,
)

from ..config import STAGES
from ..ui import section_label
from ..teeth import resolve as resolve_fdi
from ..mesh_io import read_mesh
from .base import Stage


class PlaceStage(Stage):
    name = "Place"
    description = STAGES[2][1]

    def __init__(self, app, library_dir):
        super().__init__(app)
        self.library_dir = library_dir
        self.available_teeth = sorted(
            [f for f in os.listdir(library_dir) if f.endswith('.stl')]
        )
        self.current_index = 0
        self.crown_actor = None
        self.move_step = 0.5
        self.rotate_step = 5.0
        self._last_fdi = None  # last FDI the user resolved (for persistence)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # --- AUTO-PICK BY FDI ---
        layout.addWidget(section_label("AUTO-PICK BY FDI"))
        fdi_hint = QLabel(
            "Type the tooth number (FDI). 35 = lower-left 2nd premolar; "
            "11–18 upper-right, 21–28 upper-left, 31–38 lower-left, 41–48 lower-right."
        )
        fdi_hint.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 2px 0;")
        fdi_hint.setWordWrap(True)
        layout.addWidget(fdi_hint)

        fdi_row = QHBoxLayout()
        self.fdi_spin = QSpinBox()
        self.fdi_spin.setRange(11, 48)
        self.fdi_spin.setValue(35)
        fdi_row.addWidget(self.fdi_spin)
        btn_fdi = QPushButton("Load anatomy")
        btn_fdi.clicked.connect(self._on_fdi_pick)
        fdi_row.addWidget(btn_fdi)
        layout.addLayout(fdi_row)

        self.lbl_fdi_status = QLabel("")
        self.lbl_fdi_status.setStyleSheet("color: #6e6e73; font-size: 11px; padding: 2px 0;")
        self.lbl_fdi_status.setWordWrap(True)
        layout.addWidget(self.lbl_fdi_status)

        # --- LIBRARY ---
        layout.addWidget(section_label("LIBRARY"))
        nav = QHBoxLayout()
        btn_prev = QPushButton("<<  Prev")
        btn_next = QPushButton("Next  >>")
        btn_prev.clicked.connect(self.prev_tooth)
        btn_next.clicked.connect(self.next_tooth)
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        layout.addLayout(nav)

        self.lbl_current = QLabel("(no tooth loaded)")
        self.lbl_current.setStyleSheet("color: #424245; font-size: 12px; padding: 4px 0;")
        self.lbl_current.setWordWrap(True)
        layout.addWidget(self.lbl_current)

        # --- STEP SIZE ---
        layout.addWidget(section_label("STEP SIZE"))
        step_h = QHBoxLayout()
        self.btn_fine   = QPushButton("Fine")
        self.btn_normal = QPushButton("Normal")
        self.btn_coarse = QPushButton("Coarse")
        for b in (self.btn_fine, self.btn_normal, self.btn_coarse):
            b.setCheckable(True)
        self.btn_normal.setChecked(True)
        self.btn_fine.clicked.connect(lambda: self._set_step(0.1, 1.0))
        self.btn_normal.clicked.connect(lambda: self._set_step(0.5, 5.0))
        self.btn_coarse.clicked.connect(lambda: self._set_step(2.0, 15.0))
        step_h.addWidget(self.btn_fine)
        step_h.addWidget(self.btn_normal)
        step_h.addWidget(self.btn_coarse)
        layout.addLayout(step_h)

        # --- POSITION (X / Y) ---
        layout.addWidget(section_label("POSITION (X / Y)"))
        grid = QGridLayout()
        btn_fwd   = QPushButton("Fwd")
        btn_back  = QPushButton("Back")
        btn_left  = QPushButton("Left")
        btn_right = QPushButton("Right")
        btn_fwd.clicked.connect(lambda:   self.translate(0,  1, 0))
        btn_back.clicked.connect(lambda:  self.translate(0, -1, 0))
        btn_left.clicked.connect(lambda:  self.translate(-1, 0, 0))
        btn_right.clicked.connect(lambda: self.translate( 1, 0, 0))
        grid.addWidget(btn_fwd,   0, 1)
        grid.addWidget(btn_left,  1, 0)
        grid.addWidget(btn_right, 1, 2)
        grid.addWidget(btn_back,  2, 1)
        layout.addLayout(grid)

        # --- HEIGHT (Z) ---
        layout.addWidget(section_label("HEIGHT (Z)"))
        zh = QHBoxLayout()
        btn_up   = QPushButton("Up")
        btn_down = QPushButton("Down")
        btn_up.clicked.connect(lambda:   self.translate(0, 0,  1))
        btn_down.clicked.connect(lambda: self.translate(0, 0, -1))
        zh.addWidget(btn_up)
        zh.addWidget(btn_down)
        layout.addLayout(zh)

        # --- ROTATION ---
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
            btn_pos.clicked.connect(lambda _, a=axis_key: self.rotate(a,  1))
            row.addWidget(btn_neg)
            row.addWidget(btn_pos)
            layout.addLayout(row)

        # --- SHAPE ---
        layout.addWidget(section_label("SHAPE"))
        sh = QHBoxLayout()
        btn_sup = QPushButton("Scale +5%")
        btn_sdn = QPushButton("Scale −5%")
        btn_sup.clicked.connect(lambda: self.scale(1.05))
        btn_sdn.clicked.connect(lambda: self.scale(0.95))
        sh.addWidget(btn_sup)
        sh.addWidget(btn_sdn)
        layout.addLayout(sh)

        btn_mirror = QPushButton("Mirror Anatomy")
        btn_mirror.clicked.connect(self.mirror)
        layout.addWidget(btn_mirror)

        # --- MOUSE EDIT ---
        layout.addWidget(section_label("MOUSE EDIT"))
        self.btn_edit = QPushButton("Edit with Mouse (Drag)")
        self.btn_edit.setCheckable(True)
        self.btn_edit.setStyleSheet(
            "QPushButton:checked { background-color: #0071e3; color: white; border-color: #0071e3; }"
        )
        self.btn_edit.clicked.connect(self.toggle_edit_mode)
        layout.addWidget(self.btn_edit)
        self._end_obs_id = None

        btn_reseed = QPushButton("Re-seed at Margin")
        btn_reseed.clicked.connect(self.reseed)
        layout.addWidget(btn_reseed)

        layout.addStretch()

    # --- Stage lifecycle ---

    def is_complete(self):
        return self.app.state.crown is not None

    def on_enter(self):
        if self.app.state.crown is None and self.available_teeth:
            self.load_tooth(self.current_index)
        else:
            # Re-sync our actor to the current crown — another stage (e.g. Fit)
            # may have replaced state.crown with a new (deformed) mesh, and our
            # actor may be hidden or still pointing at the old geometry.
            if self.crown_actor is not None:
                try: self.app.plotter.remove_actor(self.crown_actor)
                except Exception: pass
                self.crown_actor = None
            if self.app.state.crown is not None:
                self.crown_actor = self.app.plotter.add_mesh(
                    self.app.state.crown, color="gold", reset_camera=False)
                self.app.plotter.render()
        self.app.set_status(self.description)

    def on_exit(self):
        if self.btn_edit.isChecked():
            self.exit_edit_mode()

    def reset_crown(self):
        if self.btn_edit.isChecked():
            self.exit_edit_mode()
        if self.crown_actor is not None:
            try: self.app.plotter.remove_actor(self.crown_actor)
            except Exception: pass
        self.crown_actor = None
        self.app.state.crown = None
        self.current_index = 0
        self._last_fdi = None
        self.lbl_current.setText("(no tooth loaded)")
        if hasattr(self, "lbl_fdi_status"):
            self.lbl_fdi_status.setText("")
        self.completion_changed.emit()

    # --- Library navigation ---

    def prev_tooth(self):
        if not self.available_teeth: return
        self.current_index = (self.current_index - 1) % len(self.available_teeth)
        self.load_tooth(self.current_index)

    def next_tooth(self):
        if not self.available_teeth: return
        self.current_index = (self.current_index + 1) % len(self.available_teeth)
        self.load_tooth(self.current_index)

    def set_fdi(self, fdi):
        """Programmatically pre-fill the FDI spinbox and trigger an anatomy
        load — used by the case-folder importer after parsing scanInfo."""
        if not (11 <= int(fdi) <= 48):
            return
        self.fdi_spin.setValue(int(fdi))
        self._on_fdi_pick()

    def _on_fdi_pick(self):
        """Look the FDI number up in app/teeth.py, load the matching library
        file, mirror if it's a right-quadrant tooth (library is left-side)."""
        fdi = int(self.fdi_spin.value())
        info = resolve_fdi(fdi, self.available_teeth)
        if info["file"] is None:
            self.lbl_fdi_status.setText(
                f"No library entry for FDI {fdi} ({info['name']}). Pick manually."
            )
            return
        try:
            index = self.available_teeth.index(info["file"])
        except ValueError:
            self.lbl_fdi_status.setText(f"{info['file']} not found in library.")
            return
        self.current_index = index
        self._last_fdi = fdi
        self.load_tooth(index, mirror=info["mirror"])
        mirror_note = "  (mirrored)" if info["mirror"] else ""
        self.lbl_fdi_status.setText(
            f"FDI {fdi} — {info['name']}\n→ {info['file']}{mirror_note}"
        )

    def load_tooth(self, index, mirror=False):
        """Load library tooth `index`. If `mirror` is True, flip across the
        mid-sagittal plane (negate X) so a left-side library file fits a
        right-side prep — used when auto-picking by FDI."""
        if self.crown_actor is not None:
            try: self.app.plotter.remove_actor(self.crown_actor)
            except Exception: pass

        fn = self.available_teeth[index]
        path = os.path.join(self.library_dir, fn)
        crown = read_mesh(path)

        if mirror:
            # Sagittal flip — assumes library STLs are authored in dental
            # convention with patient X = left-right. Triangle winding must
            # also flip so outward normals stay outward after the reflection.
            pts = np.asarray(crown.points)
            pts[:, 0] = -pts[:, 0]
            crown.points = pts
            if hasattr(crown, "flip_faces"):
                crown.flip_faces(inplace=True)
            else:
                try: crown.flip_normals()
                except Exception: pass

        # Auto-orient: rotate the library tooth so its principal (longest) axis
        # points along +Z. Robust to library files authored with the long axis
        # along Y, X, or some oblique direction — the user gets a tooth standing
        # roughly upright every time, then fine-tunes with the rotation buttons.
        self._orient_long_axis_up(crown)

        # Auto-normalize mesio-distal width to ~10mm. Uses the larger of the two
        # horizontal extents (X or Y) — after rotation the mesiodistal direction
        # is whichever in-plane axis happens to be longer.
        x_len = crown.bounds[1] - crown.bounds[0]
        y_len = crown.bounds[3] - crown.bounds[2]
        horiz = max(x_len, y_len)
        if horiz > 0 and (horiz < 6.0 or horiz > 15.0):
            crown.points *= (10.0 / horiz)

        # Bisect placement: align the library tooth's centroid with the margin
        # centroid, so the margin plane cuts the tooth roughly in half. This is
        # direction-agnostic — works for both upper and lower jaws regardless of
        # whether the library's +Z is the cervical or occlusal end. User then
        # uses the rotate/translate buttons (or mouse-drag) to finalise.
        margin_centroid = np.array(self.app.state.margin_points).mean(axis=0)
        crown.translate(margin_centroid - np.array(crown.center), inplace=True)

        self.app.state.crown = crown
        # A fresh preset is undeformed, so the base starts identical to it.
        self.app.state.crown_base = crown.copy()
        self.crown_actor = self.app.plotter.add_mesh(crown, color="gold")
        self.app.plotter.render()

        self.lbl_current.setText(f"{index+1}/{len(self.available_teeth)}: {fn}")
        self.app.notify_crown_changed()
        self.completion_changed.emit()
        self.app.set_status(
            f"Loaded {fn}. Use rotation buttons or 'Edit with Mouse' to orient onto the prep."
        )

    # --- Geometry helpers ---

    def _orient_long_axis_up(self, crown):
        """Rotate `crown` in place so its principal axis (longest spread
        direction, found via SVD) points along patient +Z.

        Library STLs are authored in different conventions — some have the
        long axis along Y, some along Z, some oblique. This gives a sensible
        upright starting orientation regardless. The user's rotation buttons
        still handle any remaining 180° / cusps-up-vs-down flip.
        """
        pts = np.asarray(crown.points)
        centered = pts - pts.mean(axis=0)
        # Right singular vectors = principal axes; first row = longest direction.
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        long_axis = vh[0]
        n = float(np.linalg.norm(long_axis))
        if n < 1e-9:
            return
        long_axis = long_axis / n
        # Disambiguate sign so the axis points toward +Z (rather than -Z).
        if long_axis[2] < 0:
            long_axis = -long_axis

        target = np.array([0.0, 0.0, 1.0])
        c = float(np.clip(np.dot(long_axis, target), -1.0, 1.0))
        if c > 1.0 - 1e-9:
            return  # already aligned with +Z
        rot_axis = np.cross(long_axis, target)
        s = float(np.linalg.norm(rot_axis))
        if s < 1e-9:
            # Antiparallel — rotate 180° around any horizontal axis.
            rot_axis = np.array([1.0, 0.0, 0.0])
            angle = 180.0
        else:
            rot_axis = rot_axis / s
            angle = float(np.degrees(np.arccos(c)))
        crown.rotate_vector(
            tuple(rot_axis), angle, point=tuple(crown.center), inplace=True
        )

    def _compute_insertion_axis(self):
        pts = np.array(self.app.state.margin_points)
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[-1]
        axis = axis / np.linalg.norm(axis)

        # Disambiguate sign using only LOCAL jaw geometry around the prep — the rest of
        # the dental arch (tall neighbouring teeth) can otherwise outvote the local prep.
        jaw_pts = self.app.state.jaw_mesh.points
        margin_extent = max(np.linalg.norm(pts - centroid, axis=1).max(), 1e-3)
        distances = np.linalg.norm(jaw_pts - centroid, axis=1)
        nearby_mask = distances < margin_extent * 3
        nearby = jaw_pts[nearby_mask] if nearby_mask.any() else jaw_pts
        margin_proj = float(centroid @ axis)
        proj = nearby @ axis
        if int(np.sum(proj > margin_proj)) > int(np.sum(proj < margin_proj)):
            axis = -axis
        return centroid, axis

    # --- Edit operations ---

    def _set_step(self, move, rotate):
        self.move_step = move
        self.rotate_step = rotate
        self.btn_fine.setChecked(move == 0.1)
        self.btn_normal.setChecked(move == 0.5)
        self.btn_coarse.setChecked(move == 2.0)

    def translate(self, x, y, z):
        if self.app.state.crown is None: return
        d = self.move_step
        vec = [x*d, y*d, z*d]
        self.app.state.crown.translate(vec, inplace=True)
        if self.app.state.crown_base is not None:
            self.app.state.crown_base.translate(vec, inplace=True)
        self.app.notify_crown_changed()
        self.app.plotter.render()

    def rotate(self, axis, sign):
        if self.app.state.crown is None: return
        a = self.rotate_step * sign
        # Use the SAME world pivot for crown and base so they stay aligned even
        # if the displayed crown has been deformed (its center may differ).
        c = self.app.state.crown.center
        for m in (self.app.state.crown, self.app.state.crown_base):
            if m is None: continue
            if axis == 'x':   m.rotate_x(a, point=c, inplace=True)
            elif axis == 'y': m.rotate_y(a, point=c, inplace=True)
            else:             m.rotate_z(a, point=c, inplace=True)
        self.app.notify_crown_changed()
        self.app.plotter.render()

    def scale(self, factor):
        if self.app.state.crown is None: return
        c = np.array(self.app.state.crown.center)
        for m in (self.app.state.crown, self.app.state.crown_base):
            if m is None: continue
            m.translate(-c, inplace=True)
            m.points *= factor
            m.translate(c, inplace=True)
        self.app.notify_crown_changed()
        self.app.plotter.render()

    def mirror(self):
        if self.app.state.crown is None: return
        cx = self.app.state.crown.center[0]
        for crown in (self.app.state.crown, self.app.state.crown_base):
            if crown is None: continue
            crown.points[:, 0] = 2 * cx - crown.points[:, 0]
            if hasattr(crown, 'flip_faces'):
                crown.flip_faces(inplace=True)
            else:
                crown.flip_normals()
        self.app.notify_crown_changed()
        self.app.plotter.render()

    def reseed(self):
        """Re-place the current preset at the margin (useful if margin changed)."""
        if not self.available_teeth: return
        if self.btn_edit.isChecked():
            self.exit_edit_mode()
        self.load_tooth(self.current_index)

    def refresh_crown_actor(self):
        """Rebuild this stage's crown actor from the current app.state.crown.
        Another stage (Fit) may have replaced state.crown with a deformed mesh;
        downstream stages (Shell/Trim) use this actor as the visible outer crown,
        so it must point at the latest geometry."""
        if self.crown_actor is not None:
            try: self.app.plotter.remove_actor(self.crown_actor)
            except Exception: pass
            self.crown_actor = None
        if self.app.state.crown is not None:
            self.crown_actor = self.app.plotter.add_mesh(
                self.app.state.crown, color="gold", reset_camera=False)
            self.app.plotter.render()

    def set_outer_opacity(self, opacity):
        """Used by ShellStage to ghost the outer crown so the inner is visible."""
        if self.crown_actor is None: return
        self.crown_actor.GetProperty().SetOpacity(float(opacity))
        self.app.plotter.render()

    def set_outer_visible(self, visible):
        if self.crown_actor is None: return
        self.crown_actor.SetVisibility(bool(visible))
        self.app.plotter.render()

    # --- Mouse-drag edit mode ---

    def toggle_edit_mode(self):
        if self.btn_edit.isChecked():
            self.enter_edit_mode()
        else:
            self.exit_edit_mode()

    def enter_edit_mode(self):
        if self.app.state.crown is None or self.crown_actor is None:
            self.btn_edit.setChecked(False)
            return
        self.app.plotter.enable_trackball_actor_style()
        if self.app.jaw_actor is not None:
            self.app.jaw_actor.SetPickable(False)
        style = self.app.plotter.iren.interactor.GetInteractorStyle()
        self._end_obs_id = style.AddObserver('EndInteractionEvent', self._bake_transform)
        self.app.set_status(
            "EDIT MODE: drag the gold crown — Left=rotate, Middle=move, Right=scale. "
            "Click 'Edit with Mouse' again to exit."
        )

    def exit_edit_mode(self):
        if self._end_obs_id is not None:
            try:
                style = self.app.plotter.iren.interactor.GetInteractorStyle()
                style.RemoveObserver(self._end_obs_id)
            except Exception:
                pass
            self._end_obs_id = None
        self.app.plotter.enable_trackball_style()
        if self.app.jaw_actor is not None:
            self.app.jaw_actor.SetPickable(True)
        self.btn_edit.setChecked(False)
        self.app.set_status(self.description)

    def _bake_transform(self, _style, _event):
        """After every mouse-drag, fold the actor's transform into the mesh's vertices
        so subsequent stages see the new geometry directly."""
        if self.crown_actor is None or self.app.state.crown is None:
            return
        m = self.crown_actor.GetMatrix()
        M = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
        if np.allclose(M, np.eye(4)):
            return
        pts = self.app.state.crown.points
        H = np.hstack([pts, np.ones((pts.shape[0], 1))])
        self.app.state.crown.points = ((H @ M.T)[:, :3]).astype(pts.dtype)
        # Apply the same rigid transform to the undeformed base so Fit deforms
        # from the latest mouse-placed pose.
        base = self.app.state.crown_base
        if base is not None:
            bp = base.points
            Hb = np.hstack([bp, np.ones((bp.shape[0], 1))])
            base.points = ((Hb @ M.T)[:, :3]).astype(bp.dtype)
        self.crown_actor.SetPosition(0.0, 0.0, 0.0)
        self.crown_actor.SetOrientation(0.0, 0.0, 0.0)
        self.crown_actor.SetScale(1.0, 1.0, 1.0)
        self.app.notify_crown_changed()
        self.app.plotter.render()

    # --- Persistence ---

    def serialize(self):
        return {
            "current_index": int(self.current_index),
            "move_step": float(self.move_step),
            "rotate_step": float(self.rotate_step),
            "last_fdi": int(self._last_fdi) if self._last_fdi is not None else None,
        }

    def restore(self, data):
        # UI fields
        self.current_index = int(data.get("current_index", 0))
        last_fdi = data.get("last_fdi")
        if isinstance(last_fdi, int) and 11 <= last_fdi <= 48:
            self._last_fdi = last_fdi
            self.fdi_spin.setValue(last_fdi)
        move = float(data.get("move_step", 0.5))
        rot  = float(data.get("rotate_step", 5.0))
        if move == 0.1:
            self._set_step(0.1, rot)
        elif move == 2.0:
            self._set_step(2.0, rot)
        else:
            self._set_step(0.5, rot)

        # Rebuild crown_actor from the loaded mesh
        if self.crown_actor is not None:
            try: self.app.plotter.remove_actor(self.crown_actor)
            except Exception: pass
            self.crown_actor = None

        if self.app.state.crown is not None:
            self.crown_actor = self.app.plotter.add_mesh(self.app.state.crown, color="gold")
            if 0 <= self.current_index < len(self.available_teeth):
                fn = self.available_teeth[self.current_index]
                self.lbl_current.setText(f"{self.current_index+1}/{len(self.available_teeth)}: {fn}")
            else:
                self.lbl_current.setText("(restored crown)")
        else:
            self.lbl_current.setText("(no tooth loaded)")
