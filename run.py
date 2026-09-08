"""Backwards-compatible CLI launcher. Prefer `really-rally` after pip install."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from really_rally.cli import main

if __name__ == "__main__":
    main()
