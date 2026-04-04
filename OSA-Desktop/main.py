import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import yaml
from pathlib import Path

# Add parent path to path to allow src absolute imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ui.main_window import MainWindow
from ui.styles import GLOBAL_QSS, FONT_FAMILY


def main():
    # Enable High DPI rendering and scaling
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Apply global font
    font = QFont("Helvetica Neue", 13)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    # Apply dark theme stylesheet
    app.setStyleSheet(GLOBAL_QSS)

    # Load configuration
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize main window
    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
