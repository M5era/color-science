"""Image canvas: QGraphicsView showing the display preview of the raw buffer.

Pan by dragging, zoom with the scroll wheel or the floating −/+/Fit
buttons (bottom-right, like the reference tool). The canvas only ever
receives the one-way 8-bit preview — never the raw float data.
"""

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

_ZOOM_STEP = 1.25
_ZOOM_MIN = 0.02
_ZOOM_MAX = 64.0


class ImageCanvas(QGraphicsView):
    #: emitted after a rect-select drag: x0, y0, x1, y1 in image coordinates
    rectSelected = Signal(float, float, float, float)

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._display_ref: np.ndarray | None = None  # keeps QImage memory alive

        self.setBackgroundBrush(QColor(28, 28, 28))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._zoom_controls = self._build_zoom_controls()
        self._tool = "pan"
        self._band_start: QPointF | None = None
        self._band_end: QPointF | None = None
        self.rubberBandChanged.connect(self._band_changed)

    # -------------------------------------------------------------- tools

    def set_tool(self, tool: str) -> None:
        """'pan' (drag to scroll) or 'select' (rubber-band a chart region)."""
        self._tool = tool
        if tool == "select":
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().unsetCursor()

    def _band_changed(self, viewport_rect, from_scene: QPointF, to_scene: QPointF):
        if not viewport_rect.isNull():
            self._band_start = from_scene
            self._band_end = to_scene

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if (
            self._tool == "select"
            and self._band_start is not None
            and self._band_end is not None
        ):
            start, end = self._band_start, self._band_end
            self._band_start = self._band_end = None
            self.rectSelected.emit(start.x(), start.y(), end.x(), end.y())

    # ------------------------------------------------------------- image

    def set_display_image(self, display_u8: np.ndarray) -> None:
        """Show an (H, W, 3) uint8 preview array."""
        buffer = np.ascontiguousarray(display_u8)
        self._display_ref = buffer
        height, width, _ = buffer.shape
        qimage = QImage(
            buffer.data, width, height, 3 * width, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage)

        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
        else:
            self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(0, 0, width, height)
        self.fit()

    def has_image(self) -> bool:
        return self._pixmap_item is not None

    # -------------------------------------------------------------- zoom

    def fit(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_in(self) -> None:
        self._apply_zoom(_ZOOM_STEP)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / _ZOOM_STEP)

    def _apply_zoom(self, factor: float) -> None:
        current = self.transform().m11()
        target = current * factor
        if target < _ZOOM_MIN or target > _ZOOM_MAX:
            return
        self.scale(factor, factor)

    def wheelEvent(self, event):
        # Scale with the actual wheel delta so trackpads (many small
        # events) and mouse wheels (few big ones) both feel gradual.
        delta = event.angleDelta().y()
        if delta:
            self._apply_zoom(1.0015 ** delta)
        event.accept()

    # ---------------------------------------------- floating controls

    def _build_zoom_controls(self) -> QWidget:
        controls = QWidget(self)
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for text, handler in (
            ("−", self.zoom_out),
            ("+", self.zoom_in),
            ("Fit", self.fit),
        ):
            btn = QPushButton(text)
            btn.setFixedHeight(24)
            btn.setMinimumWidth(28)
            btn.clicked.connect(handler)
            layout.addWidget(btn)
        controls.adjustSize()
        return controls

    def resizeEvent(self, event):
        super().resizeEvent(event)
        margin = 10
        self._zoom_controls.move(
            self.width() - self._zoom_controls.width() - margin,
            self.height() - self._zoom_controls.height() - margin,
        )
