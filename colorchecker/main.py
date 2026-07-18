"""Color Checker — chart readout tool.

Entry point. Boots the AppKit application and opens the main window.
Run from source with:  python3 main.py
"""

import objc
from Cocoa import (
    NSApplication,
    NSApplicationActivationPolicyRegular,
    NSMenu,
    NSMenuItem,
    NSObject,
)

from app.ui.main_window import MainWindowController


class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        self.windowController = MainWindowController.alloc().init()
        self.windowController.showWindow_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        return True


def build_menu(app):
    main_menu = NSMenu.alloc().init()

    app_menu_item = NSMenuItem.alloc().init()
    main_menu.addItem_(app_menu_item)
    app_menu = NSMenu.alloc().init()
    app_menu.addItem_(
        NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit Color Checker", objc.selector(None, selector=b"terminate:"), "q"
        )
    )
    app_menu_item.setSubmenu_(app_menu)

    # Per-tab menus are namespaced so Matching / LUT Inspector can add their
    # own items later without restructuring the menu-building code.
    file_menu_item = NSMenuItem.alloc().init()
    main_menu.addItem_(file_menu_item)
    file_menu = NSMenu.alloc().initWithTitle_("File")
    file_menu_item.setSubmenu_(file_menu)

    app.setMainMenu_(main_menu)


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    build_menu(app)
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
