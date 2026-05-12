"""App-wide constants: paths, stage definitions, and theme stylesheet."""

LIBRARY_DIR = "/home/shiva/Documents/NuDent/Tooth_Library/tooth_stl"


STAGES = [
    ("Margin", "Click on the prep tooth to mark margin points. Press F to close the loop."),
    ("Place",  "Position a crown preset over the margin."),
    ("Shell",  "Generate the inner surface from wall thickness."),
    ("Trim",   "Cut the crown bottom to match the margin."),
    ("Refine", "Sculpt, check collisions, and export."),
]


LIGHT_QSS = """
QMainWindow, QWidget { background-color: #f5f5f7; color: #1d1d1f; }
QFrame#header { background-color: #ffffff; border-bottom: 1px solid #d2d2d7; }
QLabel#appTitle { font-size: 16px; font-weight: 600; }
QLabel#fileName { font-size: 14px; color: #424245; }
QFrame#leftRail { background-color: #ffffff; border-right: 1px solid #d2d2d7; }
QFrame#leftPanel { background-color: #ffffff; border-right: 1px solid #d2d2d7; }
QFrame#rightPanel { background-color: #ffffff; border-left: 1px solid #d2d2d7; }

QPushButton#stageButton {
    background-color: transparent;
    border: none;
    border-left: 3px solid transparent;
    color: #6e6e73;
    font-size: 11px;
    font-weight: 600;
    padding: 10px 0;
    text-align: center;
}
QPushButton#stageButton:hover:!disabled {
    background-color: #f5f5f7;
    color: #0071e3;
}
QPushButton#stageButton:checked {
    background-color: #e8f0fe;
    color: #0071e3;
    border-left: 3px solid #0071e3;
}
QPushButton#stageButton:disabled { color: #c7c7cc; }

QPushButton {
    background-color: #f5f5f7;
    color: #1d1d1f;
    border: 1px solid #d2d2d7;
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 13px;
}
QPushButton:hover:!disabled { background-color: #e8e8ed; }
QPushButton:disabled { color: #c7c7cc; background-color: #fafafa; }
QPushButton#primary {
    background-color: #0071e3;
    color: white;
    border: none;
    font-weight: 600;
}
QPushButton#primary:hover { background-color: #0077ed; }

QLabel[role="sectionHeader"] {
    font-size: 11px;
    font-weight: 700;
    color: #6e6e73;
    padding-top: 12px;
    padding-bottom: 4px;
}
QLabel[role="hint"] { color: #6e6e73; font-size: 12px; }

QListWidget {
    background-color: #fafafa;
    border: 1px solid #e5e5ea;
    border-radius: 6px;
    font-family: monospace;
    font-size: 11px;
}
QStatusBar { background-color: #ffffff; border-top: 1px solid #d2d2d7; color: #424245; }
"""
