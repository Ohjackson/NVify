"""Facial expression recognition placeholder."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def infer_emotion(image_files: Iterable[Path]) -> tuple[float, float]:
    """Return aggregated (valence, arousal) from facial frames."""
    raise NotImplementedError("Derive VA coordinates from facial expressions")
