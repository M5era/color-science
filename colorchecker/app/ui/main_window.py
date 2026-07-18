"""Main window: top bar with tab switcher, content area driven by a TabRouter.

All three tabs exist in the router from day one; Matching and LUT Inspector
are placeholders until their phases land. Adding them later means replacing
one widget, not restructuring the window.
"""

from enum import Enum
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.project import ProjectStore

from app.ui.tabs.processing_tab import ProcessingTab
from app.ui.tabs.matching_tab import MatchingTab
from app.ui.tabs.lut_inspector_tab import LutInspectorTab


class Tab(Enum):
    PROCESSING = "Processing"
    MATCHING = "Matching"
    LUT_INSPECTOR = "LUT Inspector"


class TabRouter:
    """Maps Tab enum values to widgets inside a QStackedWidget."""

    def __init__(self, stack: QStackedWidget):
        self._stack = stack
        self._indices: dict[Tab, int] = {}

    def register(self, tab: Tab, widget: QWidget) -> None:
        self._indices[tab] = self._stack.addWidget(widget)

    def select(self, tab: Tab) -> None:
        self._stack.setCurrentIndex(self._indices[tab])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Color Checker")
        self.resize(1380, 860)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._stack = QStackedWidget()
        self.router = TabRouter(self._stack)
        self._tabs: dict[Tab, QWidget] = {
            Tab.PROCESSING: ProcessingTab(),
            Tab.MATCHING: MatchingTab(),
            Tab.LUT_INSPECTOR: LutInspectorTab(),
        }
        for tab, widget in self._tabs.items():
            self.router.register(tab, widget)

        root.addWidget(self._build_top_bar())
        root.addWidget(self._stack, stretch=1)
        self.setCentralWidget(central)

        # Tabs may expose a `top_bar` widget shown on the right of the top
        # bar while they are active (Processing: filename / arrows / Load).
        for tab, widget in self._tabs.items():
            tab_bar = getattr(widget, "top_bar", None)
            if tab_bar is not None:
                self._top_right_layout.addWidget(tab_bar)

        self._tab_buttons[Tab.PROCESSING].setChecked(True)
        self._select_tab(Tab.PROCESSING)

        self._project_path: Path | None = None
        self._dirty = False
        self._tabs[Tab.PROCESSING].storeChanged.connect(self._mark_dirty)
        self._build_file_menu()
        self._update_title()

    # ----------------------------------------------------- project file

    def _build_file_menu(self) -> None:
        menu = self.menuBar().addMenu("File")
        for text, shortcut, slot in (
            ("New Project", QKeySequence.StandardKey.New, self.new_project),
            ("Open Project…", QKeySequence.StandardKey.Open, self.open_project),
            ("Save Project", QKeySequence.StandardKey.Save, self.save_project),
            ("Save Project As…", QKeySequence.StandardKey.SaveAs, self.save_project_as),
        ):
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            menu.addAction(action)

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_title()

    def _update_title(self) -> None:
        name = self._project_path.stem if self._project_path else "Untitled"
        star = " *" if self._dirty else ""
        self._title_label.setText(f"{name}{star}")
        self.setWindowTitle(f"Color Checker — {name}{star}")

    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self._tabs[Tab.PROCESSING].set_store(ProjectStore())
        self._project_path = None
        self._dirty = False
        self._update_title()

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Color Checker projects (*.ccproj.json *.json)"
        )
        if not path:
            return
        try:
            store = ProjectStore.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", f"{Path(path).name}\n\n{exc}")
            return
        self._tabs[Tab.PROCESSING].set_store(store)
        self._project_path = Path(path)
        self._dirty = False
        self._update_title()

    def save_project(self) -> None:
        if self._project_path is None:
            self.save_project_as()
            return
        try:
            self._tabs[Tab.PROCESSING].store.save(self._project_path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._dirty = False
        self._update_title()

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "untitled.ccproj.json",
            "Color Checker projects (*.ccproj.json *.json)",
        )
        if not path:
            return
        self._project_path = Path(path)
        self.save_project()

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "The current project has unsaved changes. Discard them?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def _select_tab(self, tab: Tab) -> None:
        self.router.select(tab)
        for other, widget in self._tabs.items():
            tab_bar = getattr(widget, "top_bar", None)
            if tab_bar is not None:
                tab_bar.setVisible(other is tab)

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("topBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)

        self._title_label = QLabel("Untitled")
        self._title_label.setObjectName("projectTitle")
        layout.addWidget(self._title_label)
        layout.addStretch(1)

        self._tab_buttons: dict[Tab, QPushButton] = {}
        group = QButtonGroup(bar)
        group.setExclusive(True)
        segment = QWidget()
        seg_layout = QHBoxLayout(segment)
        seg_layout.setContentsMargins(0, 0, 0, 0)
        seg_layout.setSpacing(4)
        for tab in Tab:
            btn = QPushButton(tab.value)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, t=tab: self._select_tab(t))
            group.addButton(btn)
            seg_layout.addWidget(btn)
            self._tab_buttons[tab] = btn
        layout.addWidget(segment)
        layout.addStretch(1)

        top_right = QWidget()
        self._top_right_layout = QHBoxLayout(top_right)
        self._top_right_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(top_right)

        return bar
