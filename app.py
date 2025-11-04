import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    """Entry point for Curved MPR application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Curved MPR")
    app.setOrganizationName("Medical Imaging Research")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()