"""Shared mesh/data state passed between stages."""
from PyQt5.QtCore import QObject


class AppState(QObject):
    def __init__(self):
        super().__init__()
        self.jaw_path = None
        self.jaw_mesh = None
        self.prep_mesh = None            # sub-mesh of jaw_mesh isolated as the prep tooth
        self.margin_points = []          # list of np.ndarray(3,)
        self.margin_loop_closed = False
        self.crown = None
        self.shell_outer = None
        self.shell_inner = None
        self.trimmed_crown = None
        self.final_crown = None
