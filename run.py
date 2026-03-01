import os
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from really_rally.cli import main

if __name__ == "__main__":
    main()
