#!/usr/bin/env python3
"""Sync Wind2D static assets into a mounted runtime directory."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/opt/corsewind-wind2d-static")
TARGET = ROOT / "visualizations/wind2d"
STATIC_FILES = ("index.html", "wind2d.css", "wind2d.js")


def main() -> None:
    if not SOURCE.exists():
        return
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in STATIC_FILES:
        source = SOURCE / name
        if source.exists():
            shutil.copy2(source, TARGET / name)


if __name__ == "__main__":
    main()
