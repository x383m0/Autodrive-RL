"""Where things live.

From source, everything is in the project folder. In the packaged app
(PyInstaller, see packaging/), the program files are read-only, so the
player's data (progress, trained cars) goes to a folder in their home:
~/AutodriveGame (override with the AUTODRIVE_DATA environment variable).
"""
import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
APP_DIR = Path(__file__).resolve().parent.parent          # code + assets (read-only when packaged)
DATA_DIR = Path(os.environ.get("AUTODRIVE_DATA") or
                (Path.home() / "AutodriveGame" if FROZEN else APP_DIR))
EXPERIMENTS = DATA_DIR / "experiments"


def train_command():
    """How to start a training run in a separate process."""
    if FROZEN:
        return [sys.executable, "--train"]                # the app itself, see play.py
    return [sys.executable, str(APP_DIR / "train.py")]
