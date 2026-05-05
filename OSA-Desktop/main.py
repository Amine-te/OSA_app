import sys
from pathlib import Path

# Application packages (`ui`, `core`, `workers`, …) live next to this file.
_APP_ROOT = Path(__file__).resolve().parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

# Pipeline code lives at repo root (`src/`). Add it so `from src...` works
# when launching via `python OSA-Desktop/main.py`.
_REPO_ROOT = _APP_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import yaml

from ui.main_window import MainWindow
from ui.styles import GLOBAL_QSS, FONT_FAMILY, set_theme


def main():
    # Enable High DPI rendering and scaling
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Apply global font — Inter with Segoe UI fallback
    font = QFont("Inter", 13)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    # Apply light theme stylesheet (default)
    set_theme("light", app)

    # Load configuration
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Store app reference on config so MainWindow can pass it to toggle_theme
    config["_app"] = app

    # Initialize main window
    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
