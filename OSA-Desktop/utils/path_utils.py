import os
from pathlib import Path

def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    # Assuming path_utils.py is inside OSA-Desktop/utils
    return Path(__file__).resolve().parent.parent.parent

def resolve_path(relative_path: str) -> Path:
    """Resolves a relative path robustly regardless of OS."""
    return (get_project_root() / relative_path).resolve()
