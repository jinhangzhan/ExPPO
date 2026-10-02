#!/usr/bin/env python3
"""Run the paper's shared-checkpoint, one-update controlled experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "continuation"))
from workflow import main

if __name__ == "__main__":
    main()
