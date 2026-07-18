"""Processing (readout) tab — the main working surface.

Phase 2: overlay grids with draggable corners, sidebar parameters, and
rect-select -> auto-detect. Layout: tool buttons | canvas | sidebar.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core import image_io
from app.core.detect import detect_chart_quad
from app.core.overlay import PRESETS, Overlay
from app.core.preview import to_display_u8
from app.ui.canvas import ImageCanvas
from app.ui.overlay_item import OverlayItem
from app.ui.sidebar import Sidebar


class ProcessingTab(QWidget):
    def __init__(self):
        super().__init__()
        self._current: image_io.LoadedImage | None = None
        self._overlay_items: list[OverlayItem] = []
        self._active_index: int = -1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_tool_column())

        self.canvas = ImageCanvas()
        self.canvas.rectSelected.connect(self._on_rect_selected)
        layout.addWidget(self.canvas, stretch=1)

        self.sidebar = Sidebar()
        self.sidebar.changed.connect(self._on_sidebar_edited)
        self.sidebar.overlaySelected.connect(self._on_overlay_selected)
        self.sidebar.overlayAdded.connect(self._add_overlay)
        self.sidebar.overlayRemoved.connect(self._remove_active_overlay)
        layout.addWidget(self.sidebar)

        self.top_bar = self._build_top_bar()

    # ------------------------------------------------------------- tools

    def _build_tool_column(self) -> QWidget:
        column = QWidget()
        column.setFixedWidth(44)
        vbox = QVBoxLayout(column)
        vbox.setContentsMargins(6, 12, 6, 12)
        vbox.setSpacing(6)

        group = QButtonGroup(column)
        group.setExclusive(True)
        self._tool_buttons: dict[str, QPushButton] = {}
        for tool, text, tip in (
            ("pan", "✥", "Pan / drag corners"),
            ("select", "⬚", "Select chart region (auto-detect)"),
        ):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setFixedSize(32, 32)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _=False, t=tool: self._set_tool(t))
            group.addButton(btn)
            vbox.addWidget(btn)
            self._tool_buttons[tool] = btn
        vbox.addStretch(1)

        self._tool_buttons["pan"].setChecked(True)
        return column

    def _set_tool(self, tool: str) -> None:
        self.canvas.set_tool(tool)
        self._tool_buttons[tool].setChecked(True)

    # --------------------------------------------------------- overlays

    def _active_item(self) -> OverlayItem | None:
        if 0 <= self._active_index < len(self._overlay_items):
            return self._overlay_items[self._active_index]
        return None

    def _default_corners(self) -> list[list[float]]:
        if self._current is None:
            return [[100, 100], [500, 100], [500, 350], [100, 350]]
        w, h = self._current.width, self._current.height
        return [
            [w * 0.25, h * 0.25],
            [w * 0.75, h * 0.25],
            [w * 0.75, h * 0.75],
            [w * 0.25, h * 0.75],
        ]

    def _add_overlay(self) -> None:
        name = f"Overlay {len(self._overlay_items) + 1}"
        overlay = Overlay.from_preset(
            PRESETS[0], name=name, corners=self._default_corners()
        )
        item = OverlayItem(self.canvas.scene(), overlay, self._on_corners_dragged)
        self._overlay_items.append(item)
        self._active_index = len(self._overlay_items) - 1
        self._refresh_sidebar()

    def _remove_active_overlay(self) -> None:
        item = self._active_item()
        if item is None:
            return
        item.remove()
        self._overlay_items.pop(self._active_index)
        self._active_index = min(self._active_index, len(self._overlay_items) - 1)
        self._refresh_sidebar()

    def _on_overlay_selected(self, index: int) -> None:
        self._active_index = index
        item = self._active_item()
        if item is not None:
            self.sidebar.show_overlay_values(item.overlay)

    def _on_sidebar_edited(self) -> None:
        item = self._active_item()
        if item is None:
            return
        self.sidebar.apply_to_overlay(item.overlay)
        item.model_changed()

    def _on_corners_dragged(self) -> None:
        pass  # corners aren't shown in the sidebar; nothing to sync yet

    def _refresh_sidebar(self) -> None:
        names = [item.overlay.name for item in self._overlay_items]
        self.sidebar.show_overlays(names, self._active_index)
        item = self._active_item()
        if item is not None:
            self.sidebar.show_overlay_values(item.overlay)

    # ------------------------------------------------------ auto-detect

    def _on_rect_selected(self, x0: float, y0: float, x1: float, y1: float) -> None:
        if self._current is None:
            return
        result = detect_chart_quad(self._current.pixels, (x0, y0, x1, y1))
        item = self._active_item()
        if item is None:
            self._add_overlay()
            item = self._active_item()
        item.overlay.corners = [list(pt) for pt in result.corners]
        item.model_changed()
        self._set_tool("pan")

    # --------------------------------------------------------- top bar

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._filename_label = QLabel("No image")
        layout.addWidget(self._filename_label)

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(28)
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        layout.addWidget(self._prev_btn)

        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedWidth(28)
        self._next_btn.clicked.connect(lambda: self._step(+1))
        layout.addWidget(self._next_btn)

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self._on_load_clicked)
        layout.addWidget(load_btn)

        self._update_nav_state()
        return bar

    # --------------------------------------------------------- loading

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "TIFF images (*.tif *.tiff)"
        )
        if path:
            self.open_image(path)

    def open_image(self, path: str | Path) -> None:
        try:
            loaded = image_io.load_image(path)
        except Exception as exc:  # decode errors surface to the user, not the console
            QMessageBox.critical(self, "Load failed", f"{Path(path).name}\n\n{exc}")
            return
        first_image = self._current is None
        self._current = loaded
        self.canvas.set_display_image(to_display_u8(loaded.pixels))
        self._filename_label.setText(loaded.path.name)
        self._update_nav_state()
        if first_image:
            self._add_overlay()

    def _step(self, step: int) -> None:
        if self._current is None:
            return
        target = image_io.neighbor_image(self._current.path, step)
        if target is not None:
            self.open_image(target)

    def _update_nav_state(self) -> None:
        if self._current is None:
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            return
        path = self._current.path
        self._prev_btn.setEnabled(image_io.neighbor_image(path, -1) is not None)
        self._next_btn.setEnabled(image_io.neighbor_image(path, +1) is not None)
