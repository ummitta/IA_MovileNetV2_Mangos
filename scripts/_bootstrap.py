"""Make ``src/`` importable when the scripts are executed without installing."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_src_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
