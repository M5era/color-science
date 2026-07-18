"""Main window: top bar with tab switcher, content area driven by a TabRouter.

All three tabs exist in the router from day one; Matching and LUT Inspector
are placeholders until their phases land. Adding them later means replacing
one widget, not restructuring the window.
"""

from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

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

        title = QLabel("Untitled")
        title.setObjectName("projectTitle")
        layout.addWidget(title)
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
