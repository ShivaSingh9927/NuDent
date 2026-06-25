"""Live cross-section diagram for the crown-border parameters.

PyQt5 port of the sibling `nudent` project's BorderProfileDiagram, restyled
for NuDent's light theme. Draws a tooth silhouette, the margin reference
line, and the red border profile (segments 1-5), redrawn from the same 5
parameters used by border_geometry.compute_border_profile_2d.

    1 = Horizontal   2 = Angled   3 = Angle
    4 = Vertical     5 = Below margin
"""
import numpy as np
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QPainterPath
from PyQt5.QtWidgets import QWidget


class BorderProfileDiagram(QWidget):
    # Minimum *visual* segment lengths (mm) used only for diagram layout, so
    # the 1-5 badges stay legible even when a slider value is 0. The real
    # geometry sent to the viewer uses the true slider values.
    _MIN_SEG = 0.6
    _MIN_BELOW = 0.4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setStyleSheet(
            "background:#fafafa;border:1px solid #d2d2d7;border-radius:6px;")
        self._params = dict(horizontal=0.2, angled=0.0, angle_deg=45.0,
                            vertical=0.0, below_margin=0.0)

    def set_params(self, horizontal, angled, angle_deg, vertical, below_margin):
        self._params = dict(horizontal=horizontal, angled=angled,
                            angle_deg=angle_deg, vertical=vertical,
                            below_margin=below_margin)
        self.update()

    def _profile_points(self):
        p = self._params
        h = max(p["horizontal"], self._MIN_SEG)
        a = max(p["angled"], self._MIN_SEG)
        v = max(p["vertical"], self._MIN_SEG)
        bm = max(p["below_margin"], self._MIN_BELOW)

        x0, y0 = 0.0, -bm
        x1, y1 = x0 + h, y0
        rad = np.radians(p["angle_deg"])
        x2 = x1 + a * np.cos(rad)
        y2 = y1 + a * np.sin(rad)
        x3, y3 = x2, y2 + v
        return [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#fafafa"))

        # Margin point — fixed anchor in pixel space. Everything to its right
        # is the border profile (1-5).
        M = QPointF(w * 0.40, h * 0.50)

        # Tooth silhouette: bulging crown above the margin, narrow neck at the
        # margin, tapering root below.
        crown_top_l = QPointF(w * 0.12, h * 0.06)
        crown_top_r = QPointF(w * 0.50, h * 0.03)
        crown_bulge = QPointF(w * 0.62, h * 0.18)
        neck_l = QPointF(w * 0.10, h * 0.50)
        root_bulge_r = QPointF(w * 0.46, h * 0.75)
        root_bot_r = QPointF(w * 0.34, h * 0.98)
        root_bot_l = QPointF(w * 0.16, h * 0.98)
        root_bulge_l = QPointF(w * 0.06, h * 0.75)
        crown_bulge_l = QPointF(w * 0.02, h * 0.22)

        tooth = QPainterPath()
        tooth.moveTo(crown_top_l)
        tooth.quadTo(crown_top_r, crown_top_r)
        tooth.quadTo(crown_bulge, M)
        tooth.quadTo(root_bulge_r, root_bot_r)
        tooth.quadTo(QPointF(w * 0.25, h * 1.0), root_bot_l)
        tooth.quadTo(root_bulge_l, neck_l)
        tooth.quadTo(crown_bulge_l, crown_top_l)
        tooth.closeSubpath()

        painter.setPen(QPen(QColor("#c7c7cc"), 1))
        painter.setBrush(QColor("#e5e5ea"))
        painter.drawPath(tooth)

        # Margin reference line (dashed).
        painter.setPen(QPen(QColor("#6e6e73"), 1, Qt.DashLine))
        painter.drawLine(QPointF(0, M.y()), QPointF(w, M.y()))

        # Border profile (red), anchored at M.
        pts = self._profile_points()
        xs = [0.0] + [p[0] for p in pts]
        ys = [0.0] + [p[1] for p in pts]
        span_x = max(max(xs) - min(xs), 0.4)
        span_y = max(max(ys) - min(ys), 0.4)

        pad = 14
        avail_w = max(w - M.x() - pad, 10)
        avail_h = max(M.y() - pad, 10)
        scale = min(avail_w / span_x, avail_h / span_y)

        def to_px(pt):
            x, y = pt
            return QPointF(M.x() + x * scale, M.y() - y * scale)

        all_pts = [(0.0, 0.0)] + pts
        poly_px = [to_px(p) for p in all_pts]
        painter.setPen(QPen(QColor("#e74c3c"), 2.5))
        for i in range(len(poly_px) - 1):
            painter.drawLine(poly_px[i], poly_px[i + 1])

        # Margin point marker + "Margin" callout.
        painter.setBrush(QColor("#34c759"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(poly_px[0], 3.5, 3.5)

        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)

        margin_lbl_pos = QPointF(4, M.y() - 6)
        painter.setPen(QColor("#34c759"))
        painter.drawText(margin_lbl_pos, "Margin")
        painter.setPen(QPen(QColor("#34c759"), 1, Qt.DotLine))
        painter.drawLine(QPointF(margin_lbl_pos.x() + 34, margin_lbl_pos.y() + 2),
                         poly_px[0])

        # Crown / Prep orientation labels.
        font.setPointSize(7)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor("#86868b"))
        painter.drawText(QPointF(w * 0.18, h * 0.22), "Crown")
        painter.drawText(QPointF(w * 0.13, h * 0.90), "Prep")

        # Numbered labels (1-5).
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        def midpoint(a, b):
            return QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)

        def label(num, anchor, text_offset):
            painter.setBrush(QColor("#e74c3c"))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(anchor, 2.2, 2.2)
            pos = anchor + text_offset
            # white halo for legibility, then the number on top
            painter.setPen(QColor("#ffffff"))
            for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                painter.drawText(pos + QPointF(dx, dy), str(num))
            painter.setPen(QColor("#1d1d1f"))
            painter.drawText(pos, str(num))

        p_margin, p0, p1, p2, p3 = poly_px
        label(5, midpoint(p_margin, p0), QPointF(-14, 4))   # Below margin
        label(1, midpoint(p0, p1), QPointF(-4, 14))         # Horizontal
        label(2, midpoint(p1, p2), QPointF(8, -4))          # Angled
        label(3, p1, QPointF(-4, -10))                      # Angle
        label(4, midpoint(p2, p3), QPointF(8, 4))           # Vertical

        painter.end()
