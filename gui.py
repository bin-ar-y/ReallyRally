"""Backwards-compatible GUI launcher. Prefer `really-rally-gui` after pip install."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from really_rally.gui import main

if __name__ == "__main__":
    main()
