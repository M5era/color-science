"""Processing (readout) tab — the main working surface.

Phase 2: overlay grids with draggable corners, sidebar parameters, and
rect-select -> auto-detect. Layout: tool buttons | canvas | sidebar.
"""

from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core import image_io
from app.core.csv_export import combined_csv, exportable_count
from app.core.detect import detect_chart_quad
from app.core.overlay import PRESETS, Overlay
from app.core.project import ImageEntry, ProjectStore
from app.core.preview import to_display_u8
from app.core.refine import align_grid
from app.core.sampler import sample_overlay
from app.ui.canvas import ImageCanvas
from app.ui.overlay_item import OverlayItem
from app.ui.patch_table import PatchTable
from app.ui.session_list import SessionList
from app.ui.sidebar import Sidebar


def _sample_from_dict(data: dict):
    from app.core.sampler import PatchSample

    return PatchSample(
        row=data["row"],
        col=data["col"],
        rgb=tuple(data["rgb"]),
        pixel_count=data.get("pixel_count", 0),
    )


class ProcessingTab(QWidget):
    #: emitted whenever the project store's content changes (dirty tracking)
    storeChanged = Signal()

    def __init__(self):
        super().__init__()
        self._current: image_io.LoadedImage | None = None
        self._overlay_items: list[OverlayItem] = []
        self._active_index: int = -1
        self._last_samples = []
        self.store = ProjectStore()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_tool_column())

        center = QWidget()
        center_box = QVBoxLayout(center)
        center_box.setContentsMargins(0, 0, 0, 0)
        center_box.setSpacing(0)
        self.canvas = ImageCanvas()
        self.canvas.rectSelected.connect(self._on_rect_selected)
        center_box.addWidget(self.canvas, stretch=1)
        self.table = PatchTable()
        self.table.hide()  # appears after the first Process Grid
        center_box.addWidget(self.table)
        layout.addWidget(center, stretch=1)

        self.sidebar = Sidebar()
        self.sidebar.changed.connect(self._on_sidebar_edited)
        self.sidebar.overlaySelected.connect(self._on_overlay_selected)
        self.sidebar.overlayAdded.connect(self._add_overlay)
        self.sidebar.overlayRemoved.connect(self._remove_active_overlay)
        self.sidebar.processClicked.connect(self.process_grid)
        self.sidebar.exportClicked.connect(self.export_csv)
        self.sidebar.previewClicked.connect(self.preview_csv)
        self.sidebar.overlayUseToggled.connect(self._on_overlay_use_toggled)
        layout.addWidget(self.sidebar)

        self.session_list = SessionList()
        self.session_list.entryActivated.connect(self._on_entry_activated)
        self.session_list.entriesEdited.connect(self._on_entries_edited)
        self.session_list.moveRequested.connect(self._on_move_entry)
        self.session_list.sortByEvRequested.connect(self._on_sort_by_ev)
        self.session_list.removeRequested.connect(self._on_remove_entry)
        # Between the parameter grid and the Process button in the sidebar.
        self.sidebar.layout().insertWidget(1, self.session_list, stretch=1)

        self.top_bar = self._build_top_bar()

    # ------------------------------------------------------------- tools

    def _build_tool_column(self) -> QWidget:
        column = QWidget()
        column.setFixedWidth(72)
        vbox = QVBoxLayout(column)
        vbox.setContentsMargins(6, 12, 6, 12)
        vbox.setSpacing(6)

        group = QButtonGroup(column)
        group.setExclusive(True)
        self._tool_buttons: dict[str, QPushButton] = {}
        for tool, text, tip in (
            ("pan", "Pan", "Drag to pan the image; drag the white dots to adjust corners"),
            ("select", "Detect", "Drag a loose box around the chart to auto-detect it"),
        ):
            btn = QPushButton(text)
            btn.setCheckable(True)
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
            entry = self._current_entry()
            if entry is not None and entry.patch_results:
                self._show_results_for_active_overlay(entry.patch_results)
        self._sync_overlay_use_ui()

    # ---------------------------------------------- per-frame overlay use

    def _overlay_enabled_here(self, overlay: Overlay) -> bool:
        entry = self._current_entry()
        return entry is None or overlay.name not in entry.disabled_overlays

    def _on_overlay_use_toggled(self, checked: bool) -> None:
        item = self._active_item()
        entry = self._current_entry()
        if item is None or entry is None:
            return
        name = item.overlay.name
        if checked and name in entry.disabled_overlays:
            entry.disabled_overlays.remove(name)
        elif not checked and name not in entry.disabled_overlays:
            entry.disabled_overlays.append(name)
        item.set_visible(checked)
        self.storeChanged.emit()

    def _sync_overlay_use_ui(self) -> None:
        item = self._active_item()
        available = item is not None and self._current_entry() is not None
        enabled = item is not None and self._overlay_enabled_here(item.overlay)
        self.sidebar.set_overlay_use(enabled, available)

    def _apply_overlay_visibility_for_frame(self) -> None:
        for item in self._overlay_items:
            item.set_visible(self._overlay_enabled_here(item.overlay))

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

    # ------------------------------------------------------- sampling

    def process_grid(self) -> None:
        """Sample EVERY overlay on this frame from the raw buffer (chart
        grids and light-source squares alike), store all results tagged
        with their overlay, and show the active overlay in the table."""
        if self._current is None or not self._overlay_items:
            return
        results = []
        for item in self._overlay_items:
            overlay = item.overlay
            if not self._overlay_enabled_here(overlay):
                continue  # e.g. light-source square unticked for this frame
            for sample in sample_overlay(self._current.pixels, overlay):
                row = sample.to_dict()
                row["overlay"] = overlay.name
                row["kind"] = overlay.kind
                results.append(row)

        entry = self._current_entry()
        if entry is not None:
            entry.overlays = [o.overlay.to_dict() for o in self._overlay_items]
            entry.patch_results = results
            self._refresh_session_list()
            self.storeChanged.emit()

        # Show the chart in the table, not whichever overlay happened to be
        # selected (a 1x1 light square makes it look like nothing else ran).
        for i, item in enumerate(self._overlay_items):
            if item.overlay.kind == "reflective" and self._overlay_enabled_here(item.overlay):
                self._active_index = i
                break
        self._refresh_sidebar()
        self._sync_overlay_use_ui()
        self._show_results_for_active_overlay(results)

    def _show_results_for_active_overlay(self, results: list[dict]) -> None:
        item = self._active_item()
        if item is None:
            return
        name = item.overlay.name
        rows = [r for r in results if r.get("overlay", name) == name]
        if not rows:
            return
        self.table.show_samples(
            [_sample_from_dict(r) for r in rows],
            item.overlay.rows,
            item.overlay.cols,
        )
        self.table.show()

    # --------------------------------------------------------- export

    def _csv_or_complain(self) -> str | None:
        count, skipped = exportable_count(self.store.images)
        if count == 0:
            QMessageBox.information(
                self,
                "Nothing to export",
                "No checked entries with processed results.\n"
                "Run Process Grid on the exposures you want to export.",
            )
            return None
        text = combined_csv(self.store.images)
        if skipped:
            QMessageBox.warning(
                self,
                "Some entries skipped",
                f"{skipped} checked entr{'y' if skipped == 1 else 'ies'} "
                "without processed results were left out.",
            )
        return text

    def preview_csv(self) -> None:
        text = self._csv_or_complain()
        if text is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("CSV Preview")
        dialog.resize(760, 480)
        box = QVBoxLayout(dialog)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        font = QFont("Menlo")
        font.setStyleHint(QFont.StyleHint.Monospace)
        view.setFont(font)
        box.addWidget(view)
        dialog.exec()

    def export_csv(self) -> None:
        text = self._csv_or_complain()
        if text is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "patches.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            Path(path).write_text(text)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # -------------------------------------------------------- session

    def set_store(self, store: ProjectStore) -> None:
        """Replace the project (project open). Canvas keeps showing the
        current image; entries are opened by clicking them in the list."""
        self.store = store
        self._refresh_session_list()

    def _current_entry(self) -> ImageEntry | None:
        if self._current is None:
            return None
        path = str(self._current.path)
        for entry in self.store.images:
            if entry.source_path == path:
                return entry
        return None

    def _ensure_entry(self, path) -> ImageEntry:
        for entry in self.store.images:
            if entry.source_path == str(path):
                return entry
        parsed_ev = image_io.parse_ev_from_filename(path.name)
        entry = ImageEntry(
            source_path=str(path),
            label=path.name,
            # Unmarked filenames mean the reference setup: EV 0 at 5600K.
            ev=0.0 if parsed_ev is None else parsed_ev,
            group=image_io.parse_group_from_filename(path.name) or "5600K",
        )
        self.store.images.append(entry)
        self.storeChanged.emit()
        return entry

    def _entry_index(self) -> int:
        entry = self._current_entry()
        return self.store.images.index(entry) if entry in self.store.images else -1

    def _refresh_session_list(self) -> None:
        self.session_list.set_entries(self.store.images, self._entry_index())

    def _on_entry_activated(self, index: int) -> None:
        if 0 <= index < len(self.store.images):
            entry = self.store.images[index]
            if self._current is None or str(self._current.path) != entry.source_path:
                self.open_image(entry.source_path)

    def _on_entries_edited(self) -> None:
        self.session_list.apply_edits(self.store.images)
        self.storeChanged.emit()

    def _on_move_entry(self, from_index: int, to_index: int) -> None:
        images = self.store.images
        images.insert(to_index, images.pop(from_index))
        self._refresh_session_list()
        self.session_list.table.selectRow(to_index)
        self.storeChanged.emit()

    def _on_sort_by_ev(self) -> None:
        # Entries without an EV keep their relative order, after the sorted ones.
        self.store.images.sort(key=lambda e: (e.ev is None, e.ev if e.ev is not None else 0))
        self._refresh_session_list()
        self.storeChanged.emit()

    def _on_remove_entry(self, index: int) -> None:
        if 0 <= index < len(self.store.images):
            self.store.images.pop(index)
            self._refresh_session_list()
            self.storeChanged.emit()

    # ------------------------------------------------------ auto-detect

    def _on_rect_selected(self, x0: float, y0: float, x1: float, y1: float) -> None:
        if self._current is None:
            return
        result = detect_chart_quad(self._current.pixels, (x0, y0, x1, y1))
        item = self._active_item()
        if item is None:
            self._add_overlay()
            item = self._active_item()
        overlay = item.overlay

        # Snap the grid onto the patch centers: the detected quad may span
        # more patches than the working grid (SG is physically 10x14, the
        # preset grid 8x12), so solve patch pitch + phase per axis and move
        # the corners to the aligned window. Margins become 0 by definition.
        aligned = align_grid(
            self._current.pixels,
            [list(pt) for pt in result.corners],
            overlay.rows,
            overlay.cols,
        )
        if np.isfinite(aligned.score):
            overlay.corners = aligned.corners
            overlay.margin_x = 0.0
            overlay.margin_y = 0.0
        else:
            overlay.corners = [list(pt) for pt in result.corners]

        item.model_changed()
        self.sidebar.show_overlay_values(overlay)
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

        folder_btn = QPushButton("Load Folder")
        folder_btn.setToolTip("Import every TIFF in a folder as session entries")
        folder_btn.clicked.connect(self._on_load_folder_clicked)
        layout.addWidget(folder_btn)

        self._update_nav_state()
        return bar

    # --------------------------------------------------------- loading

    def _on_load_clicked(self) -> None:
        """Multi-select import: pick one file or a whole EV sweep at once
        (Cmd-A in the dialog). Every picked file becomes a session entry;
        the first one opens on the canvas."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Images", "", "TIFF images (*.tif *.tiff)"
        )
        if not paths:
            return
        for path in paths:
            self._ensure_entry(Path(path))
        self._refresh_session_list()
        self.open_image(paths[0])

    def _on_load_folder_clicked(self) -> None:
        """Import every TIFF in a chosen folder (sorted by name)."""
        folder = QFileDialog.getExistingDirectory(self, "Load Folder")
        if not folder:
            return
        paths = sorted(
            p for p in Path(folder).iterdir()
            if p.suffix.lower() in image_io.SUPPORTED_SUFFIXES
            and not p.name.startswith(".")
        )
        if not paths:
            QMessageBox.information(
                self, "No TIFFs found", "That folder contains no .tif/.tiff files."
            )
            return
        for path in paths:
            self._ensure_entry(path)
        self._refresh_session_list()
        self.open_image(paths[0])

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
        self._ensure_entry(loaded.path)
        self._refresh_session_list()

        # Per-frame overlay enablement follows the frame.
        self._apply_overlay_visibility_for_frame()
        self._sync_overlay_use_ui()

        # Show this image's stored results if it was processed before.
        entry = self._current_entry()
        if entry is not None and entry.patch_results:
            self._show_results_for_active_overlay(entry.patch_results)

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
