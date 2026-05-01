from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

for src_dir in ROOT.glob("*/projects/*/src"):
    src_path = str(src_dir)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)