"""Top-level window: stage rail, swappable left panel, central 3D viewer, details panel."""
import os
import vtk
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QStackedWidget, QFileDialog, QShortcut, QMessageBox, QAction,
    QApplication, QInputDialog, QProgressDialog, QCheckBox,
)

from .config import LIBRARY_DIR, LIGHT_QSS, STAGES
from .state import AppState
from .ui import section_label, LayerRow
from .mesh_io import read_mesh, FILE_DIALOG_FILTER
from .scaninfo import find_scaninfo, parse_scaninfo, prep_arch
from .project import save_project, load_project, PROJECT_EXT, PROJECT_FILTER
from .settings import record_recent, forget, get_recent, get_last_project
from .segmentation import isolate_tooth
from .stages import (
    MarginStage, CementGapStage, PlaceStage, FitStage, ShellStage, TrimStage, RefineStage,
)


def _arch_from_path(path):
    """Infer 'Upper' / 'Lower' from a jaw mesh filename, or None."""
    if not path:
        return None
    base = os.path.basename(path).lower()
    if "upperjaw" in base:
        return "Upper"
    if "lowerjaw" in base:
        return "Lower"
    return None


def _find_arch_mesh(folder, arch_token):
    """Return the path to a `*-{arch_token}.{stl,obj,ply}` mesh in `folder`.

    `arch_token` is 'upperjaw' or 'lowerjaw'. Files containing '-situ' are
    skipped (those are pre-op reference scans, not the working geometry).
    Returns None if no match is found.
    """
    if not os.path.isdir(folder):
        return None
    for name in os.listdir(folder):
        lower = name.lower()
        if arch_token not in lower:
            continue
        if "-situ" in lower:
            continue
        if not lower.endswith((".stl", ".obj", ".ply")):
            continue
        return os.path.join(folder, name)
    return None


def _sibling_opposing_path(path):
    """Given a path like '*-upperjaw.stl', return the matching '*-lowerjaw.stl'
    in the same directory if it exists, and vice-versa. Returns None if no
    sibling can be inferred or found."""
    folder = os.path.dirname(path)
    base = os.path.basename(path)
    lower = base.lower()
    if "upperjaw" in lower:
        sibling_name = base[:lower.index("upperjaw")] + "lowerjaw" + base[lower.index("upperjaw") + len("upperjaw"):]
    elif "lowerjaw" in lower:
        sibling_name = base[:lower.index("lowerjaw")] + "upperjaw" + base[lower.index("lowerjaw") + len("lowerjaw"):]
    else:
        return None
    candidate = os.path.join(folder, sibling_name)
    return candidate if os.path.exists(candidate) else None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NuDent CAD")
        self.resize(1400, 900)
        self.setStyleSheet(LIGHT_QSS)
        # Window/taskbar icon
        _logo = "/home/shiva/Documents/NuDent/Nudent_logo-removebg-preview.png"
        if os.path.exists(_logo):
            self.setWindowIcon(QIcon(_logo))

        self.state = AppState()
        self.jaw_actor = None
        self.opposing_actor = None
        self._realistic_colors = False
        self._case_folder = None        # set when opened via "Open Case..."
        self._case_fdi = None           # FDI chosen for this case (single-prep or user-picked)
        self.current_stage_idx = 0
        self.current_project_path = None
        self._dirty = False
        self._slice_widget = None
        self._slice_actor = None
        self._isolated_actor = None

        # Header
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(56)
        h = QHBoxLayout(header)
        h.setContentsMargins(16, 0, 16, 0)
        # Header logo: image if the file exists, otherwise the original text.
        _logo_path = "/home/shiva/Documents/NuDent/Nudent_logo-removebg-preview.png"
        if os.path.exists(_logo_path):
            title = QLabel()
            pm = QPixmap(_logo_path)
            if not pm.isNull():
                pm = pm.scaledToHeight(40, Qt.SmoothTransformation)
                title.setPixmap(pm)
            title.setToolTip("NuDent CAD")
        else:
            title = QLabel("NuDent CAD")
            title.setObjectName("appTitle")
        h.addWidget(title)
        h.addStretch()

        # Two-line case identity (title + subtitle), centered between brand and button.
        case_box = QWidget()
        case_v = QVBoxLayout(case_box)
        case_v.setContentsMargins(0, 0, 0, 0)
        case_v.setSpacing(0)
        self.case_title = QLabel("No case loaded")
        self.case_title.setObjectName("caseTitle")
        self.case_title.setAlignment(Qt.AlignCenter)
        self.case_subtitle = QLabel("Open a case folder or import an STL to begin")
        self.case_subtitle.setObjectName("caseSubtitle")
        self.case_subtitle.setAlignment(Qt.AlignCenter)
        case_v.addWidget(self.case_title)
        case_v.addWidget(self.case_subtitle)
        h.addWidget(case_box)
        # Keep `file_label` as a hidden alias so legacy code paths still work.
        self.file_label = self.case_title

        h.addStretch()
        btn_open = QPushButton("Open Case...")
        btn_open.setObjectName("primary")
        btn_open.setToolTip("Open a scan folder containing upper/lower jaw meshes and scanInfo")
        btn_open.clicked.connect(self._open_case_folder_dialog)
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

        cement = CementGapStage(self)
        cement.completion_changed.connect(self._on_completion_changed)
        self.stages.append(cement)
        self.left_panel.addWidget(cement)

        place = PlaceStage(self, LIBRARY_DIR)
        place.completion_changed.connect(self._on_completion_changed)
        self.stages.append(place)
        self.left_panel.addWidget(place)

        fit = FitStage(self)
        fit.completion_changed.connect(self._on_completion_changed)
        self.stages.append(fit)
        self.left_panel.addWidget(fit)

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

        # Center: 3D viewer — subtle vertical gradient (cool top, warm bottom)
        # reads as a "studio" backdrop and makes the white prep mesh pop.
        self.plotter = QtInteractor(self)
        self.plotter.set_background('#b8c7d9', top='#eef3f9')
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
        rl.addWidget(section_label("LAYERS"))
        self.layer_prep = LayerRow("Prep Jaw", default_opacity=1.0)
        self.layer_prep.visibility_changed.connect(self._on_prep_visibility)
        self.layer_prep.opacity_changed.connect(self._on_prep_opacity)
        self.layer_prep.setLayerEnabled(False)
        rl.addWidget(self.layer_prep)

        self.layer_opposing = LayerRow("Opposing Jaw", default_opacity=0.35)
        self.layer_opposing.visibility_changed.connect(self._on_opposing_visibility)
        self.layer_opposing.opacity_changed.connect(self._on_opposing_opacity)
        self.layer_opposing.setLayerEnabled(False)
        rl.addWidget(self.layer_opposing)

        self.btn_load_opposing = QPushButton("Load Opposing STL...")
        self.btn_load_opposing.setEnabled(False)
        self.btn_load_opposing.clicked.connect(self._pick_opposing_file)
        rl.addWidget(self.btn_load_opposing)

        # Realistic per-vertex RGB rendering (OBJ scans only — most STL/PLY
        # files have no vertex colors so the checkbox stays disabled).
        self.chk_realistic = QCheckBox("Realistic colors")
        self.chk_realistic.setToolTip(
            "Render the prep + opposing jaws with their scanned per-vertex "
            "colors instead of flat white/blue. Available when the source "
            "file is an OBJ with embedded RGB."
        )
        self.chk_realistic.setEnabled(False)
        self.chk_realistic.toggled.connect(self._on_realistic_toggled)
        rl.addWidget(self.chk_realistic)

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
            self, "Open prep tooth", "", FILE_DIALOG_FILTER
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
        if self.opposing_actor is not None:
            try: self.plotter.remove_actor(self.opposing_actor)
            except Exception: pass
            self.opposing_actor = None

        # Reset shared state, then let each stage clean up its own actors
        self._reset_state()
        for stage in self.stages:
            stage.restore({})

        # Load
        self.state.jaw_path = path
        self.state.jaw_mesh = read_mesh(path)
        self._add_jaw_actor(opacity=1.0, visible=True)

        # Auto-detect sibling opposing-jaw STL in the same folder. The dataset
        # convention is *-upperjaw.stl / *-lowerjaw.stl, so flip those tokens.
        sibling = _sibling_opposing_path(path)
        if sibling is not None:
            try:
                self._load_opposing(sibling)
            except Exception as e:
                self.set_status(f"Couldn't load opposing jaw: {e}")

        self.plotter.reset_camera()
        # Single-file open (not via "Open Case...") clears any prior case context
        # so the header doesn't keep showing stale folder/FDI info.
        if self._case_folder is None or os.path.dirname(path) != self._case_folder:
            self._case_folder = None
            self._case_fdi = None
        self._refresh_case_header()
        self._update_details()
        self._sync_layer_panel()

        self.current_project_path = None
        self._dirty = True  # imported a mesh, not saved to a project yet
        self._refresh_gating()
        self.go_to_stage(0)
        self._update_window_title()

    # ----- Case-folder import (auto-load both jaws + scanInfo) -----

    def _open_case_folder_dialog(self):
        if not self._confirm_discard_if_dirty("Open a case folder?"):
            return
        folder = QFileDialog.getExistingDirectory(self, "Open case folder")
        if not folder:
            return
        try:
            self._load_case_folder(folder)
        except Exception as e:
            QMessageBox.critical(self, "Failed to load case", str(e))

    def _load_case_folder(self, folder):
        """Auto-load a scan folder: upper jaw, lower jaw, and scanInfo.

        Decides which arch is the prep from scanInfo's ReconstructionType. If
        multiple prep teeth are listed, ask the user which one to design.
        """
        self.set_status(f"Scanning folder: {os.path.basename(folder)}...")
        QApplication.processEvents()
        upper = _find_arch_mesh(folder, "upperjaw")
        lower = _find_arch_mesh(folder, "lowerjaw")
        if upper is None and lower is None:
            QMessageBox.warning(
                self, "No meshes found",
                f"No *-upperjaw or *-lowerjaw mesh files found in:\n{folder}",
            )
            return

        # Parse scanInfo (optional — we degrade gracefully if it's missing).
        scaninfo_path = find_scaninfo(folder)
        chosen_fdi = None
        arch = None
        if scaninfo_path is not None:
            try:
                info = parse_scaninfo(scaninfo_path)
            except Exception as e:
                self.set_status(f"scanInfo parse failed ({e}); falling back to manual.")
                info = {"preps": [], "antagonists": [], "healthy": [], "all": []}

            prep_fdis = [f for f, _ in info["preps"]]
            arch = prep_arch(prep_fdis)

            if len(prep_fdis) == 1:
                chosen_fdi = prep_fdis[0]
            elif len(prep_fdis) > 1:
                # Multi-prep case (e.g. anterior bridge) — ask which to design.
                items = [f"{f}  ({t})" for f, t in info["preps"]]
                pick, ok = QInputDialog.getItem(
                    self, "Multiple preps found",
                    f"This case has {len(prep_fdis)} prep teeth. Which one are you designing?",
                    items, 0, False,
                )
                if not ok:
                    return
                chosen_fdi = int(pick.split()[0])

        # Decide prep vs opposing arch.
        if arch == "upper":
            prep_path, opposing_path = upper, lower
        elif arch == "lower":
            prep_path, opposing_path = lower, upper
        else:
            # No scanInfo or ambiguous — fall back to whichever exists, or ask.
            if upper and not lower:
                prep_path, opposing_path = upper, None
            elif lower and not upper:
                prep_path, opposing_path = lower, None
            else:
                # Both exist but we can't tell — ask.
                pick, ok = QInputDialog.getItem(
                    self, "Which arch is the prep?",
                    "scanInfo didn't identify the prep arch. Pick the arch you're designing on:",
                    ["Upper", "Lower"], 0, False,
                )
                if not ok:
                    return
                if pick == "Upper":
                    prep_path, opposing_path = upper, lower
                else:
                    prep_path, opposing_path = lower, upper

        # Record case context BEFORE _load_jaw runs (which would otherwise
        # clear it because it can't tell the path comes from a case folder).
        self._case_folder = folder
        self._case_fdi = chosen_fdi

        # Hand off to the existing single-file loader, then attach the opposing.
        # Large OBJ scans (30+ MB, 400k+ verts) can take several seconds —
        # show a modal busy dialog so the user knows the app isn't hung.
        progress = QProgressDialog(
            f"Loading prep jaw\n{os.path.basename(prep_path)}",
            None,  # no cancel button — interrupting mid-read leaves bad state
            0, 0,  # range (0, 0) = indeterminate "busy" spinner
            self,
        )
        progress.setWindowTitle("Loading case")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.show()
        QApplication.processEvents()
        try:
            self._load_jaw(prep_path)
            if opposing_path is not None:
                progress.setLabelText(
                    f"Loading opposing jaw\n{os.path.basename(opposing_path)}"
                )
                QApplication.processEvents()
                try:
                    self._load_opposing(opposing_path)
                except Exception as e:
                    self.set_status(f"Couldn't load opposing jaw: {e}")
        finally:
            progress.close()

        # Pre-fill the FDI in Place stage so the user lands on the right tooth.
        if chosen_fdi is not None:
            from .stages import PlaceStage
            for s in self.stages:
                if isinstance(s, PlaceStage):
                    s.fdi_spin.setValue(chosen_fdi)
                    s._last_fdi = chosen_fdi
                    break
            self.set_status(
                f"Case loaded — prep FDI {chosen_fdi} pre-filled. "
                f"Complete Margin first; Place stage will auto-load the anatomy."
            )
        else:
            self.set_status(f"Case loaded from {os.path.basename(folder)}.")

        # Refresh header now that all case context is set.
        self._refresh_case_header()

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

        act_open_folder = QAction("Open &Case Folder...", self)
        act_open_folder.setShortcut(QKeySequence("Ctrl+Shift+O"))
        act_open_folder.triggered.connect(self._open_case_folder_dialog)
        file_menu.addAction(act_open_folder)

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

        self.act_hide_jaw = QAction("&Hide Prep Jaw", self)
        self.act_hide_jaw.setCheckable(True)
        self.act_hide_jaw.setShortcut(QKeySequence("Ctrl+J"))
        self.act_hide_jaw.triggered.connect(self._on_hide_jaw_menu)
        view_menu.addAction(self.act_hide_jaw)

        self.act_hide_opposing = QAction("Hide &Opposing Jaw", self)
        self.act_hide_opposing.setCheckable(True)
        self.act_hide_opposing.triggered.connect(self._on_hide_opposing_menu)
        view_menu.addAction(self.act_hide_opposing)

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

        # --- Nu Smile ---
        smile_menu = mb.addMenu("Nu &Smile")
        act_smile = QAction("Smile &Preview…", self)
        act_smile.setShortcut(QKeySequence("Ctrl+Shift+M"))
        act_smile.triggered.connect(self._open_nusmile)
        smile_menu.addAction(act_smile)

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
        if self.opposing_actor is not None:
            try: self.plotter.remove_actor(self.opposing_actor)
            except Exception: pass
            self.opposing_actor = None
        self._reset_state()
        for stage in self.stages:
            stage.restore({})
        self.current_project_path = None
        self._dirty = False
        self._case_folder = None
        self._case_fdi = None
        self._refresh_case_header()
        self.details_label.setText("Open an STL to begin.")
        self._sync_layer_panel()
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
        if self.opposing_actor is not None:
            try: self.plotter.remove_actor(self.opposing_actor)
            except Exception: pass
            self.opposing_actor = None
        # Each stage clears its own actors during restore()
        if self.state.jaw_mesh is not None:
            self._add_jaw_actor(opacity=1.0, visible=True)
            if self.state.opposing_jaw_mesh is not None:
                self._add_opposing_actor(opacity=0.35, visible=True)
            self.plotter.reset_camera()
            self._refresh_case_header()
            self._update_details()
            self._sync_layer_panel()
        else:
            self._refresh_case_header()
            self.details_label.setText("Open an STL to begin.")
            self._sync_layer_panel()

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

    def _open_nusmile(self):
        try:
            from .nusmile import NuSmileDialog
        except Exception as e:
            QMessageBox.critical(self, "Nu Smile",
                                 f"Nu Smile requires mediapipe + opencv.\n\n{e}")
            return
        dlg = NuSmileDialog(self)
        dlg.exec_()

    def _show_about(self):
        QMessageBox.about(self, "About NuDent CAD",
                          "NuDent CAD\n\nDental crown design pipeline.\n"
                          "5-stage workflow: Margin → Place → Shell → Trim → Refine.")

    # ----- Header (case identity) -----

    def _refresh_case_header(self):
        """Recompute the two-line case label from current state.

        Title:    case ID (folder name) or jaw filename
        Subtitle: arch + FDI (when known), or hints / fallback details.
        """
        if self.state.jaw_mesh is None:
            self.case_title.setText("No case loaded")
            self.case_subtitle.setText("Open a case folder or import an STL to begin")
            return

        # Title: prefer the case folder name, else the prep filename.
        if self._case_folder:
            title = os.path.basename(self._case_folder)
        elif self.state.jaw_path:
            title = os.path.basename(self.state.jaw_path)
        else:
            title = "(unsaved)"
        self.case_title.setText(title)

        # Subtitle: arch + FDI when we know them, else opposing-jaw indicator.
        parts = []
        arch = _arch_from_path(self.state.jaw_path)
        if arch:
            parts.append(arch)
        if self._case_fdi is not None:
            parts.append(f"FDI {self._case_fdi}")
        if self.state.opposing_jaw_mesh is not None:
            parts.append("antagonist loaded")
        self.case_subtitle.setText(" · ".join(parts) if parts else "Prep mesh loaded")

    # ----- Layers panel -----

    def _add_jaw_actor(self, opacity=1.0, visible=True):
        """(Re)create the prep-jaw actor honouring the current realistic-colors mode."""
        if self.state.jaw_mesh is None:
            return
        if self.jaw_actor is not None:
            try: self.plotter.remove_actor(self.jaw_actor)
            except Exception: pass
        if self._realistic_colors and "RGB" in self.state.jaw_mesh.point_data:
            self.jaw_actor = self.plotter.add_mesh(
                self.state.jaw_mesh, scalars="RGB", rgb=True,
                opacity=opacity, pickable=True,
            )
        else:
            self.jaw_actor = self.plotter.add_mesh(
                self.state.jaw_mesh, color="white",
                opacity=opacity, pickable=True,
            )
        self.jaw_actor.SetVisibility(bool(visible))

    def _add_opposing_actor(self, opacity=0.35, visible=True):
        """(Re)create the opposing-jaw actor honouring the current realistic-colors mode."""
        if self.state.opposing_jaw_mesh is None:
            return
        if self.opposing_actor is not None:
            try: self.plotter.remove_actor(self.opposing_actor)
            except Exception: pass
        if self._realistic_colors and "RGB" in self.state.opposing_jaw_mesh.point_data:
            self.opposing_actor = self.plotter.add_mesh(
                self.state.opposing_jaw_mesh, scalars="RGB", rgb=True,
                opacity=opacity, pickable=False,
            )
        else:
            self.opposing_actor = self.plotter.add_mesh(
                self.state.opposing_jaw_mesh, color=(0.55, 0.7, 0.95),
                opacity=opacity, pickable=False,
            )
        self.opposing_actor.SetVisibility(bool(visible))

    def _on_realistic_toggled(self, on):
        self._realistic_colors = bool(on)
        # Preserve the user's current opacity/visibility on each layer.
        prep_op = self.jaw_actor.GetProperty().GetOpacity() if self.jaw_actor else 1.0
        prep_vis = bool(self.jaw_actor.GetVisibility()) if self.jaw_actor else True
        opp_op = self.opposing_actor.GetProperty().GetOpacity() if self.opposing_actor else 0.35
        opp_vis = bool(self.opposing_actor.GetVisibility()) if self.opposing_actor else True
        self._add_jaw_actor(opacity=prep_op, visible=prep_vis)
        self._add_opposing_actor(opacity=opp_op, visible=opp_vis)
        # Tell the active stage to refresh its own actors — stages with
        # their own prep-mesh actors (Cement, Margin focus view) honour
        # this flag too, but only by rebuilding when asked.
        stage = self.stages[self.current_stage_idx]
        if hasattr(stage, "_redraw"):
            try: stage._redraw()
            except Exception: pass
        self.plotter.render()

    def _on_prep_visibility(self, visible):
        if self.jaw_actor is not None:
            self.jaw_actor.SetVisibility(bool(visible))
            self.plotter.render()
        if hasattr(self, "act_hide_jaw"):
            self.act_hide_jaw.setChecked(not visible)

    def _on_prep_opacity(self, opacity):
        if self.jaw_actor is not None:
            self.jaw_actor.GetProperty().SetOpacity(float(opacity))
            self.plotter.render()

    def _on_opposing_visibility(self, visible):
        if self.opposing_actor is not None:
            self.opposing_actor.SetVisibility(bool(visible))
            self.plotter.render()
        if hasattr(self, "act_hide_opposing"):
            self.act_hide_opposing.setChecked(not visible)

    def _on_opposing_opacity(self, opacity):
        if self.opposing_actor is not None:
            self.opposing_actor.GetProperty().SetOpacity(float(opacity))
            self.plotter.render()

    def _on_hide_jaw_menu(self, checked):
        self.layer_prep.setVisible_(not bool(checked))
        if self.jaw_actor is not None:
            self.jaw_actor.SetVisibility(not bool(checked))
            self.plotter.render()

    def _on_hide_opposing_menu(self, checked):
        self.layer_opposing.setVisible_(not bool(checked))
        if self.opposing_actor is not None:
            self.opposing_actor.SetVisibility(not bool(checked))
            self.plotter.render()

    def _sync_layer_panel(self):
        """Sync layer rows + menu to current actor state. Called after load/open/new."""
        has_prep = self.jaw_actor is not None
        self.layer_prep.setLayerEnabled(has_prep)
        self.layer_prep.setVisible_(True)
        self.layer_prep.setOpacity(1.0)
        if has_prep:
            self.jaw_actor.SetVisibility(True)
            self.jaw_actor.GetProperty().SetOpacity(1.0)
        if hasattr(self, "act_hide_jaw"):
            self.act_hide_jaw.setChecked(False)

        has_opp = self.opposing_actor is not None
        self.layer_opposing.setLayerEnabled(has_opp)
        self.layer_opposing.setVisible_(has_opp)
        self.layer_opposing.setOpacity(0.35)
        if has_opp:
            self.opposing_actor.SetVisibility(True)
            self.opposing_actor.GetProperty().SetOpacity(0.35)
        if hasattr(self, "act_hide_opposing"):
            self.act_hide_opposing.setChecked(False)

        self.btn_load_opposing.setEnabled(has_prep)
        self.btn_load_opposing.setText(
            "Replace Opposing STL..." if has_opp else "Load Opposing STL..."
        )

        # Realistic-colors checkbox: enabled iff at least one loaded mesh has
        # per-vertex RGB (i.e. came from an exocad-style OBJ).
        has_rgb = (
            (self.state.jaw_mesh is not None and "RGB" in self.state.jaw_mesh.point_data) or
            (self.state.opposing_jaw_mesh is not None and "RGB" in self.state.opposing_jaw_mesh.point_data)
        )
        self.chk_realistic.blockSignals(True)
        self.chk_realistic.setEnabled(has_rgb)
        if not has_rgb:
            self.chk_realistic.setChecked(False)
            self._realistic_colors = False
        else:
            self.chk_realistic.setChecked(self._realistic_colors)
        self.chk_realistic.blockSignals(False)

    # ----- Opposing jaw load -----

    def _load_opposing(self, path):
        """Load an opposing-arch STL. Assumes it shares the prep's coordinate frame
        (true for scans where upper and lower come from the same session)."""
        if self.opposing_actor is not None:
            try: self.plotter.remove_actor(self.opposing_actor)
            except Exception: pass
            self.opposing_actor = None
        mesh = read_mesh(path)
        self.state.opposing_jaw_path = path
        self.state.opposing_jaw_mesh = mesh
        self._add_opposing_actor(opacity=0.35, visible=True)
        self._sync_layer_panel()
        self.plotter.render()
        self.set_status(f"Loaded opposing jaw: {os.path.basename(path)}")

    def _pick_opposing_file(self):
        if self.state.jaw_mesh is None:
            return
        start_dir = (
            os.path.dirname(self.state.jaw_path) if self.state.jaw_path else ""
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Open opposing-jaw STL", start_dir, FILE_DIALOG_FILTER
        )
        if not path:
            return
        try:
            self._load_opposing(path)
            self._mark_dirty()
        except Exception as e:
            QMessageBox.critical(self, "Failed to load", str(e))

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
        self.state.opposing_jaw_path = None
        self.state.opposing_jaw_mesh = None
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
