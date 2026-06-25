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
        # Crown-border ("Crown Bottoms") profile, swept along the margin loop.
        # All mm except border_angle_deg (degrees). 0 = segment skipped.
        self.border_horizontal = 0.2
        self.border_angled = 0.0
        self.border_angle_deg = 45.0
        self.border_vertical = 0.0
        self.border_below_margin = 0.0
        self.crown = None
        # Undeformed crown at the current pose. Place keeps this in lockstep with
        # `crown` through every rigid move; the Fit stage deforms FROM this and
        # writes the conformed result into `crown`, so re-fitting never compounds
        # and always reflects the latest placement.
        self.crown_base = None
        self.shell_outer = None
        self.shell_inner = None
        self.trimmed_crown = None
        self.final_crown = None
