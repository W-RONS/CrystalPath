from __future__ import annotations

import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from crystalpath.ui.main_window import MainWindow


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CrystalPath")
    app.setOrganizationName("CrystalPath")
    window = MainWindow()
    window.show()
    if os.environ.get("CRYSTALPATH_SMOKE_TEST") == "1":
        QTimer.singleShot(2500, app.quit)
    return app.exec()
