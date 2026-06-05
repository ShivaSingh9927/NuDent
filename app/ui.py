"""Small UI helpers shared across stage panels."""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QLabel, QWidget, QHBoxLayout, QVBoxLayout, QToolButton, QSlider,
)


def section_label(text):
    lbl = QLabel(text)
    lbl.setProperty("role", "sectionHeader")
    lbl.setStyleSheet(
        "font-size: 11px; font-weight: 700; color: #6e6e73; "
        "padding-top: 12px; padding-bottom: 4px;"
    )
    return lbl


class LayerRow(QWidget):
    """One row in the Layers panel: visibility toggle + label + opacity slider.

    Emits `visibility_changed(bool)` and `opacity_changed(float in [0, 1])`.
    Disabled state (no actor present yet) is set via `setLayerEnabled(bool)`.
    """
    visibility_changed = pyqtSignal(bool)
    opacity_changed = pyqtSignal(float)

    def __init__(self, label, default_opacity=1.0, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(2)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)

        self.btn_vis = QToolButton()
        self.btn_vis.setCheckable(True)
        self.btn_vis.setChecked(True)
        self.btn_vis.setText("◉")  # filled = visible
        self.btn_vis.setFixedWidth(24)
        self.btn_vis.setToolTip("Toggle visibility")
        self.btn_vis.clicked.connect(self._on_vis_clicked)
        top.addWidget(self.btn_vis)

        self.lbl = QLabel(label)
        self.lbl.setStyleSheet("font-size: 12px; color: #1d1d1f;")
        top.addWidget(self.lbl)
        top.addStretch()

        self.lbl_pct = QLabel(f"{int(default_opacity * 100)}%")
        self.lbl_pct.setStyleSheet("font-size: 11px; color: #6e6e73;")
        self.lbl_pct.setFixedWidth(34)
        self.lbl_pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self.lbl_pct)

        outer.addLayout(top)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(default_opacity * 100))
        self.slider.valueChanged.connect(self._on_slider)
        outer.addWidget(self.slider)

    def _on_vis_clicked(self):
        on = self.btn_vis.isChecked()
        self.btn_vis.setText("◉" if on else "◎")
        self.visibility_changed.emit(on)

    def _on_slider(self, v):
        self.lbl_pct.setText(f"{v}%")
        self.opacity_changed.emit(v / 100.0)

    def setOpacity(self, opacity):
        """Set slider without firing signals."""
        self.slider.blockSignals(True)
        self.slider.setValue(int(opacity * 100))
        self.slider.blockSignals(False)
        self.lbl_pct.setText(f"{int(opacity * 100)}%")

    def setVisible_(self, visible):
        """Set visibility toggle without firing signals."""
        self.btn_vis.blockSignals(True)
        self.btn_vis.setChecked(bool(visible))
        self.btn_vis.setText("◉" if visible else "◎")
        self.btn_vis.blockSignals(False)

    def setLayerEnabled(self, enabled):
        self.btn_vis.setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.lbl.setEnabled(enabled)
        self.lbl_pct.setEnabled(enabled)
