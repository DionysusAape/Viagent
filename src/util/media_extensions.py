"""Video/media extensions for scanning data dirs (batch list, evidence resolve)."""
from __future__ import annotations

from pathlib import Path
from typing import List

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".gif"})


def is_video_suffix(suffix: str) -> bool:
    return suffix.lower() in VIDEO_EXTENSIONS


def iter_video_files_under(root: Path) -> List[Path]:
    """All files under root whose suffix is in VIDEO_EXTENSIONS, sorted by path (case-insensitive)."""
    if not root.exists():
        return []
    found: set[Path] = set()
    for ext in VIDEO_EXTENSIONS:
        for p in root.rglob(f"*{ext}"):
            if p.is_file():
                found.add(p.resolve())
    return sorted(found, key=lambda p: p.as_posix().lower())
