"""
result_cleanup.py — remove stale ablation artefacts before a fresh sweep.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def clear_results_dir(output_root: Path) -> int:
    """
    Delete prior run outputs under ``output_root``.

    Keeps ``.gitkeep``. Returns the number of files/directories removed.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    removed = 0
    for child in list(output_root.iterdir()):
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed += 1
    return removed
