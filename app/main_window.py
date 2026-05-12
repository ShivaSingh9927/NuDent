"""Top-level window: stage rail, swappable left panel, central 3D viewer, details panel."""
import os
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QStackedWidget, QFileDialog, QShortcut, QMessageBox, QAction,
    QApplication,
)

from .config import LIBRARY_DIR, LIGHT_QSS, STAGES
from .state import AppState
from .ui import section_label
from .project import save_project, load_project, PROJECT_EXT, PROJECT_FILTER
from .settings import record_recent, forget, get_recent, get_last_project
from .segmentation import isolate_tooth
from .stages import (
    MarginStage, PlaceStage, ShellStage, TrimStage, RefineStage,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NuDent CAD")
        self.resize(1400, 900)
        self.setStyleSheet(LIGHT_QSS)

        self.state = AppState()
        self.jaw_actor = None
        self.current_stage_idx = 0
        self.current_project_path = None
        self._dirty = False
        self._slice_widget = None
        self._slice_actor = None
        self._isolated_actor = None

        # Header
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(50)
        h = QHBoxLayout(header)
        h.setContentsMargins(16, 0, 16, 0)
        title = QLabel("NuDent CAD")
        title.setObjectName("appTitle")
        h.addWidget(title)
        h.addStretch()
        self.file_label = QLabel("No file loaded")
        self.file_label.setObjectName("fileName")
        h.addWidget(self.file_label)
        h.addStretch()
        btn_open = QPushButton("Open STL...")
        btn_open.setObjectName("primary")
        btn_open.clicked.connect(self.open_file)
        h.addWidget(btn_open)

        # Left rail: stage selector
        rail = QFrame()
        rail.setObjectName("leftRail")
        rail.setFixedWidth(64)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(0, 8, 0, 8)
        rail_layout.setSpacing(0)
        self.stage_buttons = []
        for i, (name, _) in enumerate(STAGES):
            btn = QPushButton(f"{i+1}\n{name}")
            btn.setObjectName("stageButton")
            btn.setCheckable(True)
            btn.setFixedHeight(60)
            btn.clicked.connect(lambda _checked, idx=i: self.go_to_stage(idx))
            rail_layout.addWidget(btn)
            self.stage_buttons.append(btn)
        rail_layout.addStretch()

        # Left panel (per-stage, swappable)
        self.left_panel = QStackedWidget()
        self.left_panel.setObjectName("leftPanel")
        self.left_panel.setFixedWidth(260)

        self.stages = []
        margin = MarginStage(self)
        margin.completion_changed.connect(self._on_completion_changed)
        self.stages.append(margin)
        self.left_panel.addWidget(margin)

        place = PlaceStage(self, LIBRARY_DIR)
        place.completion_changed.connect(self._on_completion_changed)
        self.stages.append(place)
        self.left_panel.addWidget(place)

        shell = ShellStage(self)
        shell.completion_changed.connect(self._on_completion_changed)
        self.stages.append(shell)
        self.left_panel.addWidget(shell)

        trim = TrimStage(self)
        trim.completion_changed.connect(self._on_completion_changed)
        self.stages.append(trim)
        self.left_panel.addWidget(trim)

        refine = RefineStage(self)
        refine.completion_changed.connect(self._on_completion_changed)
        self.stages.append(refine)
        self.left_panel.addWidget(refine)

        # Center: 3D viewer
        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        # Small XYZ orientation gizmo in the top-right corner so the user can
        # always tell which way is patient +X / +Y / +Z while orbiting.
        try:
            self.plotter.add_axes(
                viewport=(0.82, 0.82, 1.0, 1.0),
                xlabel="X", ylabel="Y", zlabel="Z",
                line_width=2,
                labels_off=False,
                interactive=False,
            )
        except Exception:
            # Older PyVista versions may not accept all kwargs — fall back to defaults.
            try:
                self.plotter.add_axes()
            except Exception:
                pass

        # Right panel: details
        right = QFrame()
        right.setObjectName("rightPanel")
        right.setFixedWidth(280)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(16, 16, 16, 16)
        rl.setSpacing(6)
        rl.addWidget(section_label("VIEW"))
        self.btn_hide_jaw = QPushButton("Hide Jaw")
        self.btn_hide_jaw.setCheckable(True)
        self.btn_hide_jaw.setStyleSheet(
            "QPushButton:checked { background-color: #0071e3; color: white; border-color: #0071e3; }"
        )
        self.btn_hide_jaw.setEnabled(False)
        self.btn_hide_jaw.clicked.connect(self._toggle_jaw_visibility)
        rl.addWidget(self.btn_hide_jaw)

        rl.addWidget(section_label("DETAILS"))
        self.details_label = QLabel("Open an STL to begin.")
        self.details_label.setStyleSheet("color: #424245; font-size: 13px;")
        self.details_label.setWordWrap(True)
        self.details_label.setTextFormat(Qt.RichText)
        rl.addWidget(self.details_label)
        rl.addStretch()

        # Assemble body
        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.addWidget(rail)
        bl.addWidget(self.left_panel)
        bl.addWidget(self.plotter.interactor, stretch=1)
        bl.addWidget(right)

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(header)
        cl.addWidget(body, stretch=1)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready. Open an STL file to begin.")

        self._build_menu_bar()
        self._wire_shortcuts()
        self._refresh_gating()
        self._update_window_title()
        self.stage_buttons[0].setChecked(True)

        # Auto-reopen the last project (if any) after the window is shown
        QTimer.singleShot(0, self._auto_reopen_last)

    def _wire_shortcuts(self):
        def margin_active():
            return isinstance(self.stages[self.current_stage_idx], MarginStage)
        QShortcut(QKeySequence("F"), self, activated=lambda: margin_active() and self.stages[self.current_stage_idx].close_loop())
        QShortcut(QKeySequence("Z"), self, activated=lambda: margin_active() and self.stages[self.current_stage_idx].undo())
        QShortcut(QKeySequence("C"), self, activated=lambda: margin_active() and self.stages[self.current_stage_idx].clear())
        # Standard Ctrl+Z / Ctrl+Y dispatch to the active stage's undo/redo.
        QShortcut(QKeySequence("Ctrl+Z"),       self, activated=self._global_undo)
        QShortcut(QKeySequence("Ctrl+Y"),       self, activated=self._global_redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self._global_redo)

    def _global_undo(self):
        """Route Ctrl+Z to whichever undo the active stage exposes."""
        stage = self.stages[self.current_stage_idx]
        if hasattr(stage, "stage_undo"):
            stage.stage_undo()
        elif hasattr(stage, "undo"):
            stage.undo()

    def _global_redo(self):
        """Route Ctrl+Y / Ctrl+Shift+Z to the active stage's redo (if any)."""
        stage = self.stages[self.current_stage_idx]
        if hasattr(stage, "stage_redo"):
            stage.stage_redo()
        elif hasattr(stage, "redo"):
            stage.redo()

    def open_file(self):
        """Import a fresh prep STL — replaces the current jaw."""
        if not self._confirm_discard_if_dirty("Open a new prep STL?"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open prep tooth", "", "Mesh (*.stl *.obj *.ply)"
        )
        if not path:
            return
        try:
            self._load_jaw(path)
        except Exception as e:
            QMessageBox.critical(self, "Failed to load", str(e))

    def _load_jaw(self, path):
        """Wipe everything and load a fresh prep STL as the new jaw."""
        if hasattr(self, "act_isolate") and self.act_isolate.isChecked():
            self._exit_isolate_mode()
        if hasattr(self, "act_slice") and self.act_slice.isChecked():
            self._disable_slice()
        # Remove jaw actor surgically (don't use plotter.clear() — it also removes lights)
        if self.jaw_actor is not None:
            try: self.plotter.remove_actor(self.jaw_actor)
            except Exception: pass
            self.jaw_actor = None

        # Reset shared state, then let each stage clean up its own actors
        self._reset_state()
        for stage in self.stages:
            stage.restore({})

        # Load
        self.state.jaw_path = path
        self.state.jaw_mesh = pv.read(path)
        self.jaw_actor = self.plotter.add_mesh(
            self.state.jaw_mesh, color="white", opacity=1.0, pickable=True
        )
        self.plotter.reset_camera()
        self.file_label.setText(os.path.basename(path))
        self._update_details()
        self._reset_jaw_toggle()

        self.current_project_path = None
        self._dirty = True  # imported a mesh, not saved to a project yet
        self._refresh_gating()
        self.go_to_stage(0)
        self._update_window_title()

    def go_to_stage(self, idx):
        if not self._can_enter_stage(idx):
            # Re-sync the checked state in case a disabled button got toggled
            for i, btn in enumerate(self.stage_buttons):
                btn.setChecked(i == self.current_stage_idx)
            return
        prev = self.current_stage_idx
        self.stages[self.current_stage_idx].on_exit()
        self.current_stage_idx = idx
        self.left_panel.setCurrentIndex(idx)
        for i, btn in enumerate(self.stage_buttons):
            btn.setChecked(i == idx)
        self.stages[idx].on_enter()
        # Autosave when advancing forward, but only to an already-named project
        if idx > prev and self.current_project_path is not None and self._dirty:
            self._autosave()

    def _can_enter_stage(self, idx):
        if self.state.jaw_mesh is None:
            return False
        if idx == 0:
            return True
        for i in range(idx):
            if not self.stages[i].is_complete():
                return False
        return True

    def _refresh_gating(self):
        for i, btn in enumerate(self.stage_buttons):
            btn.setEnabled(self._can_enter_stage(i))

    def _on_completion_changed(self):
        self._refresh_gating()

    def notify_crown_changed(self):
        """Invalidate any downstream artifacts that depended on the prior crown geometry."""
        self._mark_dirty()
        for s in self.stages:
            if isinstance(s, ShellStage) and s.app.state.shell_inner is not None:
                s.reset_shell()
            if isinstance(s, TrimStage) and s.app.state.trimmed_crown is not None:
                s.reset_trim()
            if isinstance(s, RefineStage) and s.app.state.final_crown is not None:
                s.reset_final()

    def notify_shell_changed(self):
        """Called when the shell is regenerated; invalidates trim and refine."""
        self._mark_dirty()
        for s in self.stages:
            if isinstance(s, TrimStage) and s.app.state.trimmed_crown is not None:
                s.reset_trim()
            if isinstance(s, RefineStage) and s.app.state.final_crown is not None:
                s.reset_final()

    def notify_trim_changed(self):
        """Called when trim is re-applied; invalidates the solidified crown."""
        self._mark_dirty()
        for s in self.stages:
            if isinstance(s, RefineStage) and s.app.state.final_crown is not None:
                s.reset_final()

    def set_status(self, text):
        self.statusBar().showMessage(text)

    # ----- Project persistence + menu actions -----

    def _build_menu_bar(self):
        mb = self.menuBar()
        # --- File ---
        file_menu = mb.addMenu("&File")

        act_new = QAction("&New Project", self)
        act_new.setShortcut(QKeySequence("Ctrl+N"))
        act_new.triggered.connect(self._new_project)
        file_menu.addAction(act_new)

        act_open = QAction("&Open Project...", self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self._open_project_dialog)
        file_menu.addAction(act_open)

        self.recent_menu = file_menu.addMenu("Recent &Projects")
        self._refresh_recent_menu()

        file_menu.addSeparator()

        act_save = QAction("&Save Project", self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        act_save_as = QAction("Save Project &As...", self)
        act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act_save_as.triggered.connect(self._save_project_as)
        file_menu.addAction(act_save_as)

        file_menu.addSeparator()

        act_import = QAction("&Import Prep STL...", self)
        act_import.triggered.connect(self.open_file)
        file_menu.addAction(act_import)

        act_export = QAction("&Export Final STL...", self)
        act_export.setShortcut(QKeySequence("Ctrl+E"))
        act_export.triggered.connect(self._export_final_stl)
        file_menu.addAction(act_export)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # --- View ---
        view_menu = mb.addMenu("&View")
        act_reset = QAction("&Reset Camera", self)
        act_reset.setShortcut(QKeySequence("Home"))
        act_reset.triggered.connect(lambda: (self.plotter.reset_camera(), self.plotter.render()))
        view_menu.addAction(act_reset)

        self.act_hide_jaw = QAction("&Hide Jaw", self)
        self.act_hide_jaw.setCheckable(True)
        self.act_hide_jaw.setShortcut(QKeySequence("Ctrl+J"))
        self.act_hide_jaw.triggered.connect(self._on_hide_jaw_menu)
        view_menu.addAction(self.act_hide_jaw)

        self.act_isolate = QAction("&Isolate Prep Tooth (click)", self)
        self.act_isolate.setCheckable(True)
        self.act_isolate.setShortcut(QKeySequence("Ctrl+I"))
        self.act_isolate.triggered.connect(self._toggle_isolate)
        view_menu.addAction(self.act_isolate)

        self.act_slice = QAction("&Slice View (clip plane)", self)
        self.act_slice.setCheckable(True)
        self.act_slice.setShortcut(QKeySequence("Ctrl+L"))
        self.act_slice.triggered.connect(self._toggle_slice)
        view_menu.addAction(self.act_slice)

        # --- Help ---
        help_menu = mb.addMenu("&Help")
        act_about = QAction("&About NuDent CAD", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _refresh_recent_menu(self):
        self.recent_menu.clear()
        recent = get_recent()
        if not recent:
            a = self.recent_menu.addAction("(none)")
            a.setEnabled(False)
            return
        for path in recent:
            a = self.recent_menu.addAction(os.path.basename(path))
            a.setToolTip(path)
            a.triggered.connect(lambda _checked, p=path: self._open_project(p))
        self.recent_menu.addSeparator()
        clr = self.recent_menu.addAction("Clear Recent")
        clr.triggered.connect(self._clear_recent)

    def _clear_recent(self):
        for p in list(get_recent()):
            forget(p)
        self._refresh_recent_menu()

    def _new_project(self):
        if not self._confirm_discard_if_dirty("Start a new project?"):
            return
        if hasattr(self, "act_isolate") and self.act_isolate.isChecked():
            self._exit_isolate_mode()
        if hasattr(self, "act_slice") and self.act_slice.isChecked():
            self._disable_slice()
        if self.jaw_actor is not None:
            try: self.plotter.remove_actor(self.jaw_actor)
            except Exception: pass
            self.jaw_actor = None
        self._reset_state()
        for stage in self.stages:
            stage.restore({})
        self.current_project_path = None
        self._dirty = False
        self.file_label.setText("No file loaded")
        self.details_label.setText("Open an STL to begin.")
        self._reset_jaw_toggle()
        self._refresh_gating()
        self.go_to_stage(0)
        self._update_window_title()
        self.set_status("New project. Import a prep STL to begin.")

    def _open_project_dialog(self):
        if not self._confirm_discard_if_dirty("Open another project?"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open NuDent Project", "", PROJECT_FILTER
        )
        if path:
            self._open_project(path)

    def _open_project(self, path):
        if not os.path.exists(path):
            QMessageBox.warning(self, "Not found",
                                f"Project file no longer exists:\n{path}")
            forget(path)
            self._refresh_recent_menu()
            return False
        try:
            meta = load_project(path, self)
        except Exception as e:
            QMessageBox.critical(self, "Failed to open",
                                 f"Could not load {os.path.basename(path)}:\n{e}")
            return False

        # Rebuild visualization scaffolding — surgical actor removal preserves
        # the default lights, otherwise the white mesh blends into the white bg.
        if hasattr(self, "act_isolate") and self.act_isolate.isChecked():
            self._exit_isolate_mode()
        if hasattr(self, "act_slice") and self.act_slice.isChecked():
            self._disable_slice()
        if self.jaw_actor is not None:
            try: self.plotter.remove_actor(self.jaw_actor)
            except Exception: pass
            self.jaw_actor = None
        # Each stage clears its own actors during restore()
        if self.state.jaw_mesh is not None:
            self.jaw_actor = self.plotter.add_mesh(
                self.state.jaw_mesh, color="white", opacity=1.0, pickable=True
            )
            self.plotter.reset_camera()
            self.file_label.setText(self.state.jaw_path or "(no jaw filename)")
            self._update_details()
            self._reset_jaw_toggle()
        else:
            self.file_label.setText("No file loaded")
            self.details_label.setText("Open an STL to begin.")
            self._reset_jaw_toggle()

        # Restore each stage
        stages_data = meta.get("stages", {})
        for stage in self.stages:
            stage.restore(stages_data.get(stage.name.lower(), {}))

        self.current_project_path = path
        self._dirty = False
        self._refresh_gating()

        # Jump to the saved stage if reachable, else the highest reachable one
        target = int(meta.get("current_stage", 0))
        if not self._can_enter_stage(target):
            target = 0
            for i in range(len(self.stages) - 1, -1, -1):
                if self._can_enter_stage(i):
                    target = i
                    break
        self.go_to_stage(target)

        record_recent(path)
        self._refresh_recent_menu()
        self._update_window_title()
        self.set_status(f"Opened {os.path.basename(path)}")
        return True

    def _save_project(self):
        """Save to the current project path, or prompt if none."""
        if self.state.jaw_mesh is None:
            QMessageBox.information(self, "Nothing to save",
                                    "Import a prep STL first.")
            return False
        if self.current_project_path is None:
            return self._save_project_as()
        return self._write_project_to(self.current_project_path)

    def _save_project_as(self):
        if self.state.jaw_mesh is None:
            QMessageBox.information(self, "Nothing to save",
                                    "Import a prep STL first.")
            return False
        default = self.current_project_path or os.path.expanduser("~/untitled" + PROJECT_EXT)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save NuDent Project", default, PROJECT_FILTER
        )
        if not path:
            return False
        if not path.endswith(PROJECT_EXT):
            path += PROJECT_EXT
        return self._write_project_to(path)

    def _write_project_to(self, path):
        try:
            save_project(path, self)
        except Exception as e:
            QMessageBox.critical(self, "Failed to save", str(e))
            return False
        self.current_project_path = path
        self._dirty = False
        record_recent(path)
        self._refresh_recent_menu()
        self._update_window_title()
        self.set_status(f"Saved {os.path.basename(path)}")
        return True

    def _autosave(self):
        """Silent save to the current project path. Triggered on stage advance."""
        if self.current_project_path is None:
            return
        try:
            save_project(self.current_project_path, self)
            self._dirty = False
            self._update_window_title()
            self.set_status(f"Autosaved {os.path.basename(self.current_project_path)}")
        except Exception as e:
            self.set_status(f"Autosave failed: {e}")

    def _export_final_stl(self):
        """Forward to the Refine stage's export — only enabled if final exists."""
        if self.state.final_crown is None:
            QMessageBox.information(self, "Nothing to export",
                                    "Solidify the crown in Stage 5 first.")
            return
        for s in self.stages:
            if isinstance(s, RefineStage):
                s.export_stl()
                return

    def _show_about(self):
        QMessageBox.about(self, "About NuDent CAD",
                          "NuDent CAD\n\nDental crown design pipeline.\n"
                          "5-stage workflow: Margin → Place → Shell → Trim → Refine.")

    # ----- Hide Jaw toggle -----

    def _toggle_jaw_visibility(self):
        """Right-panel button — apply current checked state to the jaw actor."""
        self._apply_jaw_visibility()
        # Keep the menu item in sync with the button.
        if hasattr(self, "act_hide_jaw"):
            self.act_hide_jaw.setChecked(self.btn_hide_jaw.isChecked())

    def _on_hide_jaw_menu(self, checked):
        """View-menu action — mirror to the right-panel button + apply."""
        self.btn_hide_jaw.setChecked(bool(checked))
        self._apply_jaw_visibility()

    def _apply_jaw_visibility(self):
        if self.jaw_actor is None:
            return
        visible = not self.btn_hide_jaw.isChecked()
        self.jaw_actor.SetVisibility(visible)
        self.btn_hide_jaw.setText("Show Jaw" if not visible else "Hide Jaw")
        self.plotter.render()

    def _reset_jaw_toggle(self):
        """Sync the toggle to the current jaw state — enabled iff a jaw is loaded,
        unchecked (jaw visible) on every load/new/open."""
        has_jaw = self.jaw_actor is not None
        self.btn_hide_jaw.setEnabled(has_jaw)
        self.btn_hide_jaw.setChecked(False)
        self.btn_hide_jaw.setText("Hide Jaw")
        if hasattr(self, "act_hide_jaw"):
            self.act_hide_jaw.setChecked(False)

    # ----- Slice View (clip plane to expose obscured surfaces) -----

    def _toggle_slice(self):
        if self.act_slice.isChecked():
            self._enable_slice()
        else:
            self._disable_slice()

    def _enable_slice(self):
        if self.state.jaw_mesh is None:
            QMessageBox.information(self, "No mesh", "Open a prep STL first.")
            self.act_slice.setChecked(False)
            return
        if self.jaw_actor is not None:
            self.jaw_actor.SetVisibility(False)

        mesh = self.state.jaw_mesh
        # PyVista's add_plane_widget hands us an interactive plane and fires the
        # callback with (normal, origin) on end-of-drag (we set the event to
        # EndInteractionEvent so re-clipping doesn't run on every mouse move).
        self._slice_widget = self.plotter.add_plane_widget(
            callback=self._on_slice_changed,
            normal='x',
            origin=mesh.center,
            bounds=mesh.bounds,
            factor=1.1,
            color='magenta',
            outline_translation=True,
            origin_translation=True,
            interaction_event=vtk.vtkCommand.EndInteractionEvent,
        )
        # Initial clip at the starting plane position
        self._on_slice_changed((1.0, 0.0, 0.0), tuple(mesh.center))
        self.set_status(
            "Slice View ON — drag the magenta plane to slice away occluding teeth. "
            "Ctrl+L (or View menu) to disable."
        )

    def _on_slice_changed(self, normal, origin):
        if self.state.jaw_mesh is None:
            return
        try:
            clipped = self.state.jaw_mesh.clip(
                normal=normal, origin=origin, invert=False
            )
        except Exception as e:
            self.set_status(f"Clip failed: {e}")
            return
        if self._slice_actor is not None:
            try: self.plotter.remove_actor(self._slice_actor)
            except Exception: pass
        self._slice_actor = self.plotter.add_mesh(
            clipped, color="white", opacity=1.0, pickable=True
        )
        self.plotter.render()

    def _disable_slice(self):
        if self._slice_widget is not None:
            try: self._slice_widget.SetEnabled(0)
            except Exception: pass
            self._slice_widget = None
        if self._slice_actor is not None:
            try: self.plotter.remove_actor(self._slice_actor)
            except Exception: pass
            self._slice_actor = None
        if self.jaw_actor is not None:
            self.jaw_actor.SetVisibility(True)
        self.plotter.render()
        if hasattr(self, "act_slice") and self.act_slice.isChecked():
            self.act_slice.setChecked(False)
        self.set_status("Slice View OFF.")

    # ----- Isolate Prep Tooth (region-grow from a click) -----

    def _toggle_isolate(self):
        if self.act_isolate.isChecked():
            self._enter_isolate_mode()
        else:
            self._exit_isolate_mode()

    def _enter_isolate_mode(self):
        if self.state.jaw_mesh is None:
            QMessageBox.information(self, "No mesh", "Open a prep STL first.")
            self.act_isolate.setChecked(False)
            return
        # If a previous isolation is still showing, drop it so the next click
        # starts from the full jaw.
        if self._isolated_actor is not None:
            try: self.plotter.remove_actor(self._isolated_actor)
            except Exception: pass
            self._isolated_actor = None
            if self.jaw_actor is not None:
                self.jaw_actor.SetVisibility(True)
        # Hijack the picker for one click (replaces any active stage picker).
        self.plotter.enable_surface_point_picking(
            callback=self._on_isolate_pick,
            left_clicking=True,
            show_point=False,
            show_message=False,
        )
        self.set_status(
            "Click anywhere on the prep tooth to isolate it. Ctrl+I again to cancel."
        )

    def _on_isolate_pick(self, point):
        if point is None:
            return
        try: self.plotter.disable_picking()
        except Exception: pass

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.set_status("Isolating tooth...")
        QApplication.processEvents()
        try:
            isolated = isolate_tooth(self.state.jaw_mesh, point)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.act_isolate.setChecked(False)
            self._restore_stage_picker()
            QMessageBox.warning(self, "Isolation failed", str(e))
            return
        QApplication.restoreOverrideCursor()

        if isolated is None or isolated.n_points == 0:
            self.act_isolate.setChecked(False)
            self._restore_stage_picker()
            QMessageBox.warning(self, "Isolation failed",
                                "Could not extract a tooth region from that click.")
            return

        # Swap visible mesh: hide full jaw, show isolated region.
        if self.jaw_actor is not None:
            self.jaw_actor.SetVisibility(False)
        self._isolated_actor = self.plotter.add_mesh(
            isolated, color="white", opacity=1.0, pickable=True
        )
        self.plotter.render()
        self._restore_stage_picker()
        n_orig = self.state.jaw_mesh.n_points
        n_new = isolated.n_points
        self.set_status(
            f"Isolated tooth ({n_new:,}/{n_orig:,} verts). "
            "Ctrl+I again to restore the full jaw."
        )

    def _exit_isolate_mode(self):
        if self._isolated_actor is not None:
            try: self.plotter.remove_actor(self._isolated_actor)
            except Exception: pass
            self._isolated_actor = None
        if self.jaw_actor is not None:
            self.jaw_actor.SetVisibility(True)
        try: self.plotter.disable_picking()
        except Exception: pass
        self._restore_stage_picker()
        self.plotter.render()
        if hasattr(self, "act_isolate") and self.act_isolate.isChecked():
            self.act_isolate.setChecked(False)
        self.set_status("Full jaw view restored.")

    def _restore_stage_picker(self):
        """Re-run the active stage's on_enter so its picker (margin point-picking,
        etc.) is reinstated after the isolate-mode picker took over."""
        try:
            self.stages[self.current_stage_idx].on_enter()
        except Exception:
            pass

    # ----- Helpers -----

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            self._update_window_title()

    def _update_window_title(self):
        base = "NuDent CAD"
        if self.current_project_path:
            name = os.path.basename(self.current_project_path)
            mark = " *" if self._dirty else ""
            self.setWindowTitle(f"{base} — {name}{mark}")
        elif self.state.jaw_mesh is not None:
            mark = " *" if self._dirty else ""
            self.setWindowTitle(f"{base} — (unsaved){mark}")
        else:
            self.setWindowTitle(base)

    def _update_details(self):
        m = self.state.jaw_mesh
        if m is None:
            return
        sx = m.bounds[1] - m.bounds[0]
        sy = m.bounds[3] - m.bounds[2]
        sz = m.bounds[5] - m.bounds[4]
        name = self.state.jaw_path or "(unknown)"
        self.details_label.setText(
            f"<b>File</b>: {os.path.basename(name)}<br>"
            f"<b>Vertices</b>: {m.n_points:,}<br>"
            f"<b>Triangles</b>: {m.n_cells:,}<br>"
            f"<b>Size</b>: {sx:.1f} × {sy:.1f} × {sz:.1f} mm"
        )

    def _reset_state(self):
        self.state.jaw_path = None
        self.state.jaw_mesh = None
        self.state.prep_mesh = None
        self.state.margin_points = []
        self.state.margin_loop_closed = False
        self.state.crown = None
        self.state.shell_outer = None
        self.state.shell_inner = None
        self.state.trimmed_crown = None
        self.state.final_crown = None

    def _confirm_discard_if_dirty(self, action_question):
        if not self._dirty:
            return True
        if self.current_project_path is None:
            # No file to save to — offer Save As / Discard / Cancel
            reply = QMessageBox.question(
                self, "Unsaved changes",
                f"You have unsaved changes. {action_question}",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return False
            if reply == QMessageBox.Save:
                return self._save_project_as()
            return True
        reply = QMessageBox.question(
            self, "Unsaved changes",
            f"You have unsaved changes. {action_question}",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save:
            return self._save_project()
        return True

    def _auto_reopen_last(self):
        last = get_last_project()
        if not last:
            return
        if not os.path.exists(last):
            forget(last)
            self._refresh_recent_menu()
            return
        self._open_project(last)

    def closeEvent(self, event):
        if self._confirm_discard_if_dirty("Quit anyway?"):
            event.accept()
        else:
            event.ignore()
