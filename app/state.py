"""Shared mesh/data state passed between stages."""
from PyQt5.QtCore import QObject


class AppState(QObject):
    def __init__(self):
        super().__init__()
        self.jaw_path = None
        self.jaw_mesh = None
        self.opposing_jaw_path = None    # antagonist arch (shares the prep's coordinate frame)
        self.opposing_jaw_mesh = None
        self.prep_mesh = None            # sub-mesh of jaw_mesh isolated as the prep tooth
        self.margin_points = []          # list of np.ndarray(3,)
        self.margin_loop_closed = False
        # A known cap-side seed point: the first margin click (manual or AI).
        # Used by CementGapStage to flood the cap from a guaranteed-cap-side
        # vertex, bounded by the margin loop. Far more robust than guessing
        # an "up" axis when prep_mesh isn't truly isolated.
        self.cap_seed_point = None
        # Cement-gap stage output: cap = portion of prep above the margin line,
        # labelled per-vertex as 0 (cement gap zone) or 1 (no-cement band near
        # the margin). Parameters drive the labelling and the future shell offset.
        self.cap_mesh = None
        self.cap_zone_labels = None      # np.ndarray(int32) of length cap_mesh.n_points
        self.cement_gap_thickness = 0.08   # mm — orange zone lift
        self.no_cement_thickness = 0.0     # mm — blue zone lift (0 = crown seats hard against prep)
        self.no_cement_band_width = 1.0    # mm from margin
        self.crown = None
        self.shell_outer = None
        self.shell_inner = None
        self.trimmed_crown = None
        self.final_crown = None
