import sys
import time
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog,
    QMessageBox, QFrame, QVBoxLayout, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QRadialGradient, QCursor, QPainterPath
)
import pyvista as pv
from pyvistaqt import QtInteractor
import trimesh
from vtkmodules.vtkRenderingCore import vtkCellPicker

from cross_section import get_cross_section


BUBBLE_SIZE = 200   # pixels — circular lens diameter
MARGIN = 22         # pixels inside circle before drawing profile
RADIUS_MM = 5.0     # crop radius in mesh units (mm)
OFFSET = 60         # pixels offset from cursor to bubble top-left


# ---------------------------------------------------------------------------
# Circular bubble overlay
# ---------------------------------------------------------------------------

class CrossSectionBubble(QWidget):
    """Circular lens showing the 2D cross-section under the cursor."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedSize(BUBBLE_SIZE, BUBBLE_SIZE)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._pts = []
        self.hide()

    def update_profile(self, pts):
        self._pts = list(pts)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cx = BUBBLE_SIZE // 2
        cy = BUBBLE_SIZE // 2
        r  = BUBBLE_SIZE // 2 - 2

        # Clip to circle so corners stay transparent
        clip = QPainterPath()
        clip.addEllipse(2, 2, BUBBLE_SIZE - 4, BUBBLE_SIZE - 4)
        painter.setClipPath(clip)

        # Radial gradient background
        grad = QRadialGradient(cx, cy, r)
        grad.setColorAt(0.0,  QColor(28, 18, 60, 235))
        grad.setColorAt(0.75, QColor(20, 12, 46, 220))
        grad.setColorAt(1.0,  QColor(12,  6, 28, 150))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, BUBBLE_SIZE - 4, BUBBLE_SIZE - 4)

        # Cross-section profile
        pts = self._pts
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            rng = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
            scale = (r - MARGIN) * 2 / rng

            path = QPainterPath()
            for i, (px, py) in enumerate(pts):
                sx = cx + px * scale
                sy = cy - py * scale    # flip y: world up → screen up
                if i == 0:
                    path.moveTo(sx, sy)
                else:
                    path.lineTo(sx, sy)
            painter.setPen(QPen(QColor(255, 255, 255), 2.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

        # Small dot at P (the centre)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 120, 30, 230))
        painter.drawEllipse(cx - 4, cy - 4, 8, 8)

        # Circle outline (outside clip so it stays crisp)
        painter.setClipping(False)
        painter.setPen(QPen(QColor(190, 170, 230, 160), 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(2, 2, BUBBLE_SIZE - 4, BUBBLE_SIZE - 4)

        painter.end()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dental Scan Viewer")
        self.resize(1100, 800)

        self._trimesh = None
        self._last_pick_time = 0.0

        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.005)

        self._frame = QFrame()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self._plotter = QtInteractor(self._frame)
        layout.addWidget(self._plotter)
        self._frame.setLayout(layout)
        self.setCentralWidget(self._frame)

        self._bubble = CrossSectionBubble(self._frame)

        self._build_menu()

        self.show()
        self._plotter.enable_trackball_style()
        self._plotter.set_background("dimgray")
        self._plotter.add_text(
            "File → Open STL to load a scan", position="upper_left",
            font_size=10, color="white"
        )
        self._plotter.render()

        self._plotter.AddObserver("LeftButtonPressEvent", self._on_click)
        self._plotter.AddObserver("MouseMoveEvent", self._on_mouse_move)

    # ------------------------------------------------------------------
    def _build_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open STL…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_stl)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    # ------------------------------------------------------------------
    def _open_stl(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open STL file", "", "STL files (*.stl);;All files (*)"
        )
        if not path:
            return

        try:
            pv_mesh = pv.read(path)
            loaded = trimesh.load(path)
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(
                    list(loaded.geometry.values())
                )
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        self._trimesh = None
        self._bubble.hide()
        self._plotter.clear()
        self._plotter.add_mesh(pv_mesh, color="lightgray",
                               show_edges=False, pickable=True)
        self._plotter.reset_camera()
        self._plotter.render()
        self._trimesh = loaded
        self.setWindowTitle(f"Dental Scan Viewer — {path}")

    # ------------------------------------------------------------------
    def _on_click(self, obj, event):
        x, y = self._plotter.GetEventPosition()
        self._picker.Pick(x, y, 0, self._plotter.renderer)
        if self._picker.GetCellId() != -1:
            p = self._picker.GetPickPosition()
            print(f"Picked:  x={p[0]:.4f}  y={p[1]:.4f}  z={p[2]:.4f}")

    # ------------------------------------------------------------------
    def _on_mouse_move(self, obj, event):
        now = time.monotonic()
        if now - self._last_pick_time < 1 / 60:
            return
        self._last_pick_time = now

        if self._trimesh is None:
            return

        x_vtk, y_vtk = self._plotter.GetEventPosition()
        self._picker.Pick(x_vtk, y_vtk, 0, self._plotter.renderer)

        if self._picker.GetCellId() == -1:
            self._bubble.hide()
            self._plotter.remove_actor('hover_point', render=True)
            return

        P = np.array(self._picker.GetPickPosition())

        # Camera vectors — read live every hover, never cached
        cam = self._plotter.renderer.GetActiveCamera()
        forward = np.array(cam.GetFocalPoint()) - np.array(cam.GetPosition())
        fl = np.linalg.norm(forward)
        if fl < 1e-9:
            self._bubble.hide()
            return
        forward /= fl
        up_unit = np.array(cam.GetViewUp())
        up_unit /= np.linalg.norm(up_unit)

        plane_normal = np.cross(forward, up_unit)
        pl = np.linalg.norm(plane_normal)
        if pl < 1e-9:
            self._bubble.hide()
            return
        plane_normal /= pl

        pts = get_cross_section(self._trimesh, P, plane_normal,
                                up_unit, RADIUS_MM)

        # Green dot at P on the mesh
        self._plotter.add_mesh(
            pv.Sphere(radius=0.6, center=P.tolist()),
            color='lime', name='hover_point', render=True
        )

        # Position bubble offset from cursor using Qt logical coords
        cur = self._frame.mapFromGlobal(QCursor.pos())
        bx = min(cur.x() + OFFSET, self._frame.width()  - BUBBLE_SIZE - 4)
        by = min(cur.y() + OFFSET, self._frame.height() - BUBBLE_SIZE - 4)
        bx = max(bx, 4)
        by = max(by, 4)

        self._bubble.move(bx, by)
        self._bubble.update_profile(pts)
        self._bubble.show()
        self._bubble.raise_()

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._plotter.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.close)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
