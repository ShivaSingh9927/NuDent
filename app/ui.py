"""Small UI helpers shared across stage panels."""
from PyQt5.QtWidgets import QLabel


def section_label(text):
    lbl = QLabel(text)
    lbl.setProperty("role", "sectionHeader")
    lbl.setStyleSheet(
        "font-size: 11px; font-weight: 700; color: #6e6e73; "
        "padding-top: 12px; padding-bottom: 4px;"
    )
    return lbl
