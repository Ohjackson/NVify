"""Speech emotion recognition placeholder."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def infer_emotion(audio_files: Iterable[Path]) -> tuple[float, float]:
    """Return aggregated (valence, arousal) from speech inputs."""
    raise NotImplementedError("Derive VA coordinates from speech signals")
