"""Session list: one row per exposure in the project.

Columns: include-checkbox, Label, EV, Group — label/ev/group editable in
place. Row order IS the export order; reorder with the arrow buttons or
Sort by EV. Clicking a row opens that image. A dot marks entries that
already have processed patch results.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.project import ImageEntry


class SessionList(QWidget):
    entryActivated = Signal(int)  # row clicked -> open that image
    entriesEdited = Signal()  # include/label/ev/group changed in place
    moveRequested = Signal(int, int)  # from_index, to_index
    sortByEvRequested = Signal()
    removeRequested = Signal(int)

    def __init__(self):
        super().__init__()
        self._updating = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(4)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["✓", "Label", "EV", "Group"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        # Multi-select (Cmd/Shift-click): editing EV/Group or toggling the
        # checkbox on one selected row applies to every selected row.
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setColumnWidth(0, 24)
        self.table.setColumnWidth(2, 40)
        self.table.setColumnWidth(3, 60)
        # currentCellChanged covers clicks AND arrow-key navigation, so
        # stepping through rows with the keyboard switches frames too.
        self.table.currentCellChanged.connect(self._current_cell_changed)
        self.table.itemChanged.connect(self._item_changed)
        root.addWidget(self.table, stretch=1)

        buttons = QHBoxLayout()
        for text, slot, tip in (
            ("↑", lambda: self._move(-1), "Move up"),
            ("↓", lambda: self._move(+1), "Move down"),
            ("Sort EV", self.sortByEvRequested.emit, "Sort by EV ascending"),
            ("✕", self._remove, "Remove from session"),
        ):
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        root.addLayout(buttons)

    # ------------------------------------------------------------ update

    def set_entries(self, entries: list[ImageEntry], current: int) -> None:
        self._updating = True
        self.table.setRowCount(len(entries))
        for i, entry in enumerate(entries):
            check = QTableWidgetItem()
            check.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            check.setCheckState(
                Qt.CheckState.Checked if entry.include else Qt.CheckState.Unchecked
            )
            self.table.setItem(i, 0, check)

            label = QTableWidgetItem(entry.label)
            if entry.patch_results:
                label.setToolTip("Processed")
                label.setText("● " + entry.label)
            self.table.setItem(i, 1, label)

            ev = QTableWidgetItem("" if entry.ev is None else f"{entry.ev:g}")
            self.table.setItem(i, 2, ev)
            self.table.setItem(i, 3, QTableWidgetItem(entry.group))

        if 0 <= current < len(entries):
            self.table.selectRow(current)
        self._updating = False

    def apply_edits(self, entries: list[ImageEntry]) -> None:
        for i, entry in enumerate(entries):
            if i >= self.table.rowCount():
                break
            entry.include = (
                self.table.item(i, 0).checkState() == Qt.CheckState.Checked
            )
            label_text = self.table.item(i, 1).text().removeprefix("● ").strip()
            if label_text:
                entry.label = label_text
            ev_text = self.table.item(i, 2).text().strip().replace(",", ".")
            try:
                entry.ev = float(ev_text) if ev_text else None
            except ValueError:
                pass  # keep previous value on unparseable input
            entry.group = self.table.item(i, 3).text().strip()

    # ----------------------------------------------------------- signals

    def selected_rows(self) -> list[int]:
        return sorted(i.row() for i in self.table.selectionModel().selectedRows())

    def _current_cell_changed(self, row: int, _col: int, prev_row: int, _pc: int) -> None:
        if self._updating or row < 0 or row == prev_row:
            return
        # Only switch frames on single-row focus moves; building a
        # multi-selection (Shift/Cmd-click) shouldn't load images.
        if len(self.selected_rows()) <= 1:
            self.entryActivated.emit(row)

    def _item_changed(self, item) -> None:
        if self._updating:
            return
        # Batch edit: propagate the change to every other selected row
        # (EV, Group, and the include checkbox; labels stay per-file).
        selected = {index.row() for index in self.table.selectionModel().selectedRows()}
        if item.row() in selected and len(selected) > 1 and item.column() in (0, 2, 3):
            self._updating = True
            for row in selected:
                if row == item.row():
                    continue
                target = self.table.item(row, item.column())
                if target is None:
                    continue
                if item.column() == 0:
                    target.setCheckState(item.checkState())
                else:
                    target.setText(item.text())
            self._updating = False
        self.entriesEdited.emit()

    def _move(self, delta: int) -> None:
        row = self.table.currentRow()
        target = row + delta
        if row >= 0 and 0 <= target < self.table.rowCount():
            self.moveRequested.emit(row, target)

    def _remove(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.removeRequested.emit(row)
