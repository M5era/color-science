"""Processing (readout) tab — the main working surface.

Phase 1: canvas with image loading, folder prev/next, zoom controls.
The tab also provides the top-bar widget (filename, arrows, Load Image),
which MainWindow shows while this tab is active.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core import image_io
from app.core.preview import to_display_u8
from app.ui.canvas import ImageCanvas


class ProcessingTab(QWidget):
    def __init__(self):
        super().__init__()
        self._current: image_io.LoadedImage | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas)

        self.top_bar = self._build_top_bar()

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
        self._current = loaded
        self.canvas.set_display_image(to_display_u8(loaded.pixels))
        self._filename_label.setText(loaded.path.name)
        self._update_nav_state()

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
