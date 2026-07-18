"""Right sidebar: overlay selection and grid parameters.

Mirrors the reference tool's fields: Overlay picker with +/−, Preset,
Rows/Columns, Margin X/Y, Patch Size (%), Patch Offset (%).
Emits `changed` when the user edits a field, `overlaySelected`,
`overlayAdded`, `overlayRemoved` for list management.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.overlay import PRESETS, Overlay


class Sidebar(QWidget):
    changed = Signal()
    overlaySelected = Signal(int)
    overlayAdded = Signal()
    overlayRemoved = Signal()
    processClicked = Signal()
    processAllClicked = Signal()
    exportClicked = Signal()
    previewClicked = Signal()
    overlayUseToggled = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(300)
        self._updating = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        grid = QGridLayout()
        grid.setVerticalSpacing(8)
        root.addLayout(grid)
        root.addStretch(1)
        row = 0

        grid.addWidget(QLabel("Overlay"), row, 0)
        overlay_row = QHBoxLayout()
        self.overlay_combo = QComboBox()
        self.overlay_combo.currentIndexChanged.connect(self._overlay_picked)
        overlay_row.addWidget(self.overlay_combo, stretch=1)
        add_btn = QPushButton("+")
        add_btn.setFixedWidth(28)
        add_btn.clicked.connect(self.overlayAdded.emit)
        overlay_row.addWidget(add_btn)
        remove_btn = QPushButton("−")
        remove_btn.setFixedWidth(28)
        remove_btn.clicked.connect(self.overlayRemoved.emit)
        overlay_row.addWidget(remove_btn)
        grid.addLayout(overlay_row, row, 1)
        row += 1

        self.use_check = QCheckBox("Use on this frame")
        self.use_check.setToolTip(
            "Untick to skip this overlay on the current frame\n"
            "(e.g. the light-source square on frames without the light)"
        )
        self.use_check.toggled.connect(self._use_toggled)
        grid.addWidget(self.use_check, row, 1)
        row += 1

        grid.addWidget(QLabel("Preset"), row, 0)
        self.preset_combo = QComboBox()
        for preset in PRESETS:
            self.preset_combo.addItem(preset.name)
        self.preset_combo.currentIndexChanged.connect(self._preset_picked)
        grid.addWidget(self.preset_combo, row, 1)
        row += 1

        grid.addWidget(QLabel("Rows"), row, 0)
        dims_row = QHBoxLayout()
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 32)
        dims_row.addWidget(self.rows_spin)
        dims_row.addWidget(QLabel("Columns:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 32)
        dims_row.addWidget(self.cols_spin)
        grid.addLayout(dims_row, row, 1)
        row += 1

        grid.addWidget(QLabel("Margin X"), row, 0)
        margin_row = QHBoxLayout()
        self.margin_x_spin = self._pct_spin(0.0, 40.0)
        margin_row.addWidget(self.margin_x_spin)
        margin_row.addWidget(QLabel("Margin Y:"))
        self.margin_y_spin = self._pct_spin(0.0, 40.0)
        margin_row.addWidget(self.margin_y_spin)
        grid.addLayout(margin_row, row, 1)
        row += 1

        grid.addWidget(QLabel("Patch Size (%)"), row, 0)
        self.patch_size_spin = self._pct_spin(1.0, 100.0)
        grid.addWidget(self.patch_size_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("Patch Offset (%)"), row, 0)
        self.patch_offset_spin = self._pct_spin(-50.0, 50.0)
        grid.addWidget(self.patch_offset_spin, row, 1)

        process_row = QHBoxLayout()
        self.process_btn = QPushButton("Process Grid")
        self.process_btn.clicked.connect(self.processClicked.emit)
        process_row.addWidget(self.process_btn)
        self.process_all_btn = QPushButton("Process All")
        self.process_all_btn.setToolTip(
            "Process every image in the session with the current overlays"
        )
        self.process_all_btn.clicked.connect(self.processAllClicked.emit)
        process_row.addWidget(self.process_all_btn)
        root.addLayout(process_row)

        export_row = QHBoxLayout()
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.exportClicked.emit)
        export_row.addWidget(self.export_btn)
        self.preview_btn = QPushButton("Preview CSV")
        self.preview_btn.clicked.connect(self.previewClicked.emit)
        export_row.addWidget(self.preview_btn)
        root.addLayout(export_row)

        for spin in (self.rows_spin, self.cols_spin):
            spin.valueChanged.connect(self._field_edited)
        for spin in (
            self.margin_x_spin,
            self.margin_y_spin,
            self.patch_size_spin,
            self.patch_offset_spin,
        ):
            spin.valueChanged.connect(self._field_edited)

    @staticmethod
    def _pct_spin(minimum: float, maximum: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        return spin

    # ----------------------------------------------------------- update

    def show_overlays(self, names: list[str], current: int) -> None:
        self._updating = True
        self.overlay_combo.clear()
        self.overlay_combo.addItems(names)
        if 0 <= current < len(names):
            self.overlay_combo.setCurrentIndex(current)
        self._updating = False

    def show_overlay_values(self, overlay: Overlay) -> None:
        self._updating = True
        idx = self.preset_combo.findText(overlay.preset_name)
        self.preset_combo.setCurrentIndex(idx if idx >= 0 else self.preset_combo.count() - 1)
        self.rows_spin.setValue(overlay.rows)
        self.cols_spin.setValue(overlay.cols)
        self.margin_x_spin.setValue(overlay.margin_x)
        self.margin_y_spin.setValue(overlay.margin_y)
        self.patch_size_spin.setValue(overlay.patch_size)
        self.patch_offset_spin.setValue(overlay.patch_offset)
        self._updating = False

    def apply_to_overlay(self, overlay: Overlay) -> None:
        overlay.rows = self.rows_spin.value()
        overlay.cols = self.cols_spin.value()
        overlay.margin_x = self.margin_x_spin.value()
        overlay.margin_y = self.margin_y_spin.value()
        overlay.patch_size = self.patch_size_spin.value()
        overlay.patch_offset = self.patch_offset_spin.value()
        overlay.preset_name = self.preset_combo.currentText()
        # The preset determines reflective vs emissive — picking Light
        # Source from the dropdown must tag the overlay emissive.
        for preset in PRESETS:
            if preset.name == overlay.preset_name:
                overlay.kind = preset.kind
                break

    def set_overlay_use(self, enabled: bool, available: bool) -> None:
        self._updating = True
        self.use_check.setChecked(enabled)
        self.use_check.setEnabled(available)
        self._updating = False

    # ---------------------------------------------------------- signals

    def _use_toggled(self, checked: bool) -> None:
        if not self._updating:
            self.overlayUseToggled.emit(checked)

    def _field_edited(self) -> None:
        if not self._updating:
            self.changed.emit()

    def _overlay_picked(self, index: int) -> None:
        if not self._updating and index >= 0:
            self.overlaySelected.emit(index)

    def _preset_picked(self, index: int) -> None:
        if self._updating or index < 0:
            return
        preset = PRESETS[index]
        self._updating = True
        self.rows_spin.setValue(preset.rows)
        self.cols_spin.setValue(preset.cols)
        self.margin_x_spin.setValue(preset.margin_x)
        self.margin_y_spin.setValue(preset.margin_y)
        self.patch_size_spin.setValue(preset.patch_size)
        self._updating = False
        self.changed.emit()
