"""Run as: python -m app"""
import os
import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen

from .main_window import MainWindow


SPLASH_PATH = "/home/shiva/Documents/NuDent/Nudent_opening.png"
SPLASH_DURATION_MS = 2000  # total time the splash is visible


def main():
    app = QApplication(sys.argv)

    splash = None
    if os.path.exists(SPLASH_PATH):
        pix = QPixmap(SPLASH_PATH)
        if not pix.isNull():
            if pix.width() > 720 or pix.height() > 720:
                pix = pix.scaled(720, 720, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            splash = QSplashScreen(pix, Qt.WindowStaysOnTopHint)
            splash.show()
            splash.raise_()
            splash.activateWindow()
            # Pump the event loop a few times so the splash actually paints
            # before we hand control back. A single processEvents() is often
            # not enough on Wayland/XCB.
            for _ in range(5):
                app.processEvents()
        else:
            print(f"[splash] failed to load pixmap: {SPLASH_PATH}", file=sys.stderr)
    else:
        print(f"[splash] file not found: {SPLASH_PATH}", file=sys.stderr)

    # Defer MainWindow construction until AFTER the splash has been visible
    # for SPLASH_DURATION_MS. MainWindow's constructor blocks for a few
    # seconds (PyVista plotter init, OpenGL context), so building it before
    # the event loop runs is what was hiding the splash entirely.
    holder = {}

    def _build_and_show():
        holder["window"] = MainWindow()  # keep a reference so GC doesn't kill it
        holder["window"].show()
        if splash is not None:
            splash.finish(holder["window"])

    QTimer.singleShot(SPLASH_DURATION_MS, _build_and_show)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
