"""Streamlit dashboard entrypoint inside ui package."""

import sys
from pathlib import Path

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import main

if __name__ == "__main__":
    main()

