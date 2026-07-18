"""Canvas graphics for an overlay: patch grid + four draggable corner handles."""

from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem

from app.core.homography import patch_quads_image
from app.core.overlay import Overlay

_GRID_PEN = QPen(QColor(255, 255, 255, 200))
_GRID_PEN.setCosmetic(True)  # constant 1px regardless of zoom
_OUTLINE_PEN = QPen(QColor(255, 255, 255, 140))
_OUTLINE_PEN.setCosmetic(True)
_HANDLE_RADIUS = 6.0


class CornerHandle(QGraphicsEllipseItem):
    """One draggable corner dot; reports moves back to the OverlayItem."""

    def __init__(self, index: int, on_moved: Callable[[int, QPointF], None]):
        r = _HANDLE_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r)
        self._index = index
        self._on_moved = on_moved
        self._silent = False
        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setZValue(10)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def set_position_silently(self, x: float, y: float) -> None:
        self._silent = True
        self.setPos(x, y)
        self._silent = False

    def itemChange(self, change, value):
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged
            and not self._silent
        ):
            self._on_moved(self._index, value)
        return super().itemChange(change, value)


class GridItem(QGraphicsItem):
    """Draws the overlay's outer quad and every patch sample quad."""

    def __init__(self, overlay: Overlay):
        super().__init__()
        self._overlay = overlay
        self._quads: list[np.ndarray] = []
        self.setZValue(5)
        self.refresh()

    def refresh(self) -> None:
        self.prepareGeometryChange()
        try:
            self._quads = [q for _, _, q in patch_quads_image(self._overlay)]
        except ValueError:  # degenerate corners mid-drag: draw outline only
            self._quads = []
        self.update()

    def boundingRect(self) -> QRectF:
        xs = [pt[0] for pt in self._overlay.corners]
        ys = [pt[1] for pt in self._overlay.corners]
        pad = 2.0
        return QRectF(
            min(xs) - pad, min(ys) - pad,
            (max(xs) - min(xs)) + 2 * pad, (max(ys) - min(ys)) + 2 * pad,
        )

    def paint(self, painter, option, widget=None):
        painter.setPen(_OUTLINE_PEN)
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in self._overlay.corners]))
        painter.setPen(_GRID_PEN)
        for quad in self._quads:
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in quad]))


class OverlayItem:
    """Bundles the grid and its handles; keeps them in sync with the model.

    `on_changed` fires after a corner drag updates the model, so the owner
    can refresh anything bound to the overlay (sidebar fields, etc.).
    """

    def __init__(self, scene, overlay: Overlay, on_changed: Callable[[], None]):
        self._scene = scene
        self.overlay = overlay
        self._on_changed = on_changed
        self.grid = GridItem(overlay)
        scene.addItem(self.grid)
        self.handles = [CornerHandle(i, self._handle_moved) for i in range(4)]
        for handle in self.handles:
            scene.addItem(handle)
        self._sync_handles()

    def _handle_moved(self, index: int, pos: QPointF) -> None:
        self.overlay.corners[index] = [pos.x(), pos.y()]
        self.grid.refresh()
        self._on_changed()

    def _sync_handles(self) -> None:
        for handle, (x, y) in zip(self.handles, self.overlay.corners):
            handle.set_position_silently(x, y)

    def model_changed(self) -> None:
        """Call after the overlay model was edited externally (sidebar, detect)."""
        self._sync_handles()
        self.grid.refresh()

    def set_visible(self, visible: bool) -> None:
        self.grid.setVisible(visible)
        for handle in self.handles:
            handle.setVisible(visible)

    def remove(self) -> None:
        self._scene.removeItem(self.grid)
        for handle in self.handles:
            self._scene.removeItem(handle)
