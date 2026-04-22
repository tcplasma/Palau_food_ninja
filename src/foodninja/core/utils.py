import os
import sys

def get_resource_path(relative_path: str) -> str:
    """
    Get the absolute path to a resource, works for dev and for PyInstaller.
    PyInstaller creates a temporary folder and stores path in _MEIPASS.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # If not running as EXE, use current directory
        # We assume the script is run from the project root
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
