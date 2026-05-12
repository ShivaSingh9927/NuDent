"""Stage base class — every workflow step subclasses this."""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget


class Stage(QWidget):
    """Base class for stages. Subclasses build their panel UI in __init__
    and override the lifecycle hooks below."""
    completion_changed = pyqtSignal()

    name = "Stage"
    description = ""

    def __init__(self, app):
        super().__init__()
        self.app = app  # MainWindow reference

    def is_complete(self) -> bool:
        return False

    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def serialize(self) -> dict:
        """Return per-stage UI state for inclusion in project.json. Mesh data
        lives in AppState; only return what's NOT already in AppState."""
        return {}

    def restore(self, data: dict):
        """Rebuild this stage's actors + UI from app.state (already populated)
        and the per-stage dict from project.json."""
        pass
