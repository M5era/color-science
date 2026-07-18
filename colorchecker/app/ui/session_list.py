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
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setColumnWidth(0, 24)
        self.table.setColumnWidth(2, 40)
        self.table.setColumnWidth(3, 60)
        self.table.cellClicked.connect(self._cell_clicked)
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

    def _cell_clicked(self, row: int, col: int) -> None:
        if col != 0:  # checkbox clicks shouldn't switch images
            self.entryActivated.emit(row)

    def _item_changed(self, _item) -> None:
        if not self._updating:
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
