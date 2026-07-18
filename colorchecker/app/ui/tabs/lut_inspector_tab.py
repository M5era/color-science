"""LUT Inspector: load a .cube and see exactly what it does.

Three views, no editing and no smoothing (yet):
- Image: reference gradient or a loaded image, original vs with LUT
- Curves: R/G/B response along the neutral axis vs the identity
- 3D LUT: the output lattice as a rotatable point cloud
"""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core import image_io
from app.core.lut import (
    CubeLUT,
    apply_lut,
    lattice_points,
    neutral_curves,
    parse_cube,
    reference_gradient,
)
from app.core.preview import to_display_u8


class CurvesView(QWidget):
    """R/G/B response along the neutral axis, identity as dotted diagonal."""

    def __init__(self):
        super().__init__()
        self._curves: tuple[np.ndarray, np.ndarray] | None = None
        self.setMinimumSize(400, 300)

    def set_lut(self, lut: CubeLUT | None) -> None:
        self._curves = neutral_curves(lut) if lut is not None else None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        w, h = self.width(), self.height()
        margin = 30
        plot_w, plot_h = w - 2 * margin, h - 2 * margin

        grid_pen = QPen(QColor(70, 70, 70))
        painter.setPen(grid_pen)
        for i in range(11):
            x = margin + plot_w * i / 10
            y = margin + plot_h * i / 10
            painter.drawLine(int(x), margin, int(x), h - margin)
            painter.drawLine(margin, int(y), w - margin, int(y))

        identity_pen = QPen(QColor(150, 150, 150))
        identity_pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(identity_pen)
        painter.drawLine(margin, h - margin, w - margin, margin)

        if self._curves is None:
            painter.setPen(QColor(170, 170, 170))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Load a LUT to plot curves")
            return

        inputs, outputs = self._curves
        # Scale the y-axis to fit the output range (with identity visible).
        lo = min(0.0, float(outputs.min()))
        hi = max(1.0, float(outputs.max()))
        span = hi - lo or 1.0

        for channel, color in enumerate(
            (QColor(235, 80, 80), QColor(90, 200, 90), QColor(90, 130, 235))
        ):
            pen = QPen(color)
            pen.setWidth(2)
            painter.setPen(pen)
            points = [
                QPointF(
                    margin + plot_w * t,
                    h - margin - plot_h * (outputs[i, channel] - lo) / span,
                )
                for i, t in enumerate(inputs)
            ]
            painter.drawPolyline(points)

        painter.setPen(QColor(200, 200, 200))
        painter.drawText(margin + 4, margin + 14, "LUT response curves (neutral axis)")


class LatticeView(QWidget):
    """Output lattice as a colored point cloud; drag to orbit."""

    def __init__(self):
        super().__init__()
        self._points: tuple[np.ndarray, np.ndarray] | None = None
        self._yaw, self._pitch = 0.6, 0.4
        self._last_pos = None
        self.setMinimumSize(400, 300)

    def set_lut(self, lut: CubeLUT | None, resolution: int = 17) -> None:
        self._points = lattice_points(lut, resolution) if lut is not None else None
        self.update()

    def mousePressEvent(self, event):
        self._last_pos = event.position()

    def mouseMoveEvent(self, event):
        if self._last_pos is not None:
            delta = event.position() - self._last_pos
            self._yaw += delta.x() * 0.01
            self._pitch = float(np.clip(self._pitch + delta.y() * 0.01, -1.4, 1.4))
            self._last_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self._last_pos = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        if self._points is None:
            painter.setPen(QColor(170, 170, 170))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Load a LUT to view the lattice (drag to rotate)")
            return

        _, outputs = self._points
        centered = outputs - 0.5
        cy, sy = np.cos(self._yaw), np.sin(self._yaw)
        cp, sp = np.cos(self._pitch), np.sin(self._pitch)
        x = centered[:, 0] * cy + centered[:, 2] * sy
        z = -centered[:, 0] * sy + centered[:, 2] * cy
        y = centered[:, 1] * cp - z * sp
        depth = centered[:, 1] * sp + z * cp

        scale = min(self.width(), self.height()) * 0.62
        cx_screen, cy_screen = self.width() / 2, self.height() / 2
        order = np.argsort(depth)

        colors = np.clip(outputs, 0.0, 1.0)
        for i in order:
            painter.fillRect(
                int(cx_screen + x[i] * scale) - 2,
                int(cy_screen - y[i] * scale) - 2,
                4, 4,
                QColor(int(colors[i, 0] * 255), int(colors[i, 1] * 255), int(colors[i, 2] * 255)),
            )
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(10, 18, "Drag to rotate")


class LutInspectorTab(QWidget):
    def __init__(self):
        super().__init__()
        self._lut: CubeLUT | None = None
        self._preview: np.ndarray = reference_gradient()

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ---------------------------------------------------- controls
        controls = QVBoxLayout()
        load_btn = QPushButton("Load .cube…")
        load_btn.clicked.connect(self._load_lut)
        controls.addWidget(load_btn)
        self.lut_label = QLabel("No LUT loaded")
        self.lut_label.setWordWrap(True)
        controls.addWidget(self.lut_label)

        controls.addSpacing(12)
        image_btn = QPushButton("Load preview image…")
        image_btn.clicked.connect(self._load_image)
        controls.addWidget(image_btn)
        default_btn = QPushButton("Default reference")
        default_btn.clicked.connect(self._load_default)
        controls.addWidget(default_btn)
        controls.addStretch(1)

        controls_box = QWidget()
        controls_box.setLayout(controls)
        controls_box.setFixedWidth(200)
        root.addWidget(controls_box)

        # ------------------------------------------------------- views
        right = QVBoxLayout()
        switcher_row = QHBoxLayout()
        switcher_row.addStretch(1)
        self._view_buttons = QButtonGroup(self)
        self._view_buttons.setExclusive(True)
        for i, name in enumerate(("Image", "Curves", "3D LUT")):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, idx=i: self._stack.setCurrentIndex(idx))
            self._view_buttons.addButton(btn)
            switcher_row.addWidget(btn)
        switcher_row.addStretch(1)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Original", "With LUT"])
        self.mode_combo.currentIndexChanged.connect(self._refresh_image)
        switcher_row.addWidget(self.mode_combo)
        right.addLayout(switcher_row)

        self._stack = QStackedWidget()
        self.image_view = QLabel()
        self.image_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_view.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self._stack.addWidget(self.image_view)
        self.curves_view = CurvesView()
        self._stack.addWidget(self.curves_view)
        self.lattice_view = LatticeView()
        self._stack.addWidget(self.lattice_view)
        right.addWidget(self._stack, stretch=1)
        root.addLayout(right, stretch=1)

        self._view_buttons.buttons()[0].setChecked(True)
        self._refresh_image()

    # -------------------------------------------------------- loading

    def _load_lut(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load LUT", "", "Cube LUT (*.cube)"
        )
        if not path:
            return
        try:
            self._lut = parse_cube(path)
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot load LUT", str(exc))
            return
        domain = (
            f"domain {self._lut.domain_min.min():g}–{self._lut.domain_max.max():g}"
        )
        self.lut_label.setText(
            f"{Path(path).name}\n{self._lut.size}³ lattice, {domain}"
        )
        self.curves_view.set_lut(self._lut)
        self.lattice_view.set_lut(self._lut)
        self.mode_combo.setCurrentIndex(1)  # show the LUT's effect right away
        self._refresh_image()

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load preview image", "", "TIFF images (*.tif *.tiff)"
        )
        if not path:
            return
        try:
            self._preview = image_io.load_image(path).pixels
        except Exception as exc:
            QMessageBox.critical(self, "Cannot load image", str(exc))
            return
        self._refresh_image()

    def _load_default(self) -> None:
        self._preview = reference_gradient()
        self._refresh_image()

    # ------------------------------------------------------ rendering

    def _refresh_image(self) -> None:
        img = self._preview
        if self.mode_combo.currentIndex() == 1 and self._lut is not None:
            img = apply_lut(self._lut, img)
        display = np.ascontiguousarray(to_display_u8(img))
        h, w, _ = display.shape
        qimage = QImage(display.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimage)
        self._update_scaled()

    def _update_scaled(self) -> None:
        if getattr(self, "_pixmap", None) is None:
            return
        self.image_view.setPixmap(
            self._pixmap.scaled(
                self.image_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled()
