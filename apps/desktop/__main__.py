"""Entry point: python -m apps.desktop"""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from apps.desktop.main_window import MainWindow
from apps.desktop.theme import apply_palette


def main() -> int:
    # High-DPI is default on Qt6, but keep explicit style hints.
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Nepse Quant Workstation")
    app.setOrganizationName("nepse-quant")

    apply_palette(app)

    win = MainWindow()
    win.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
