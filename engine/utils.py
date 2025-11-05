"""Shared helpers for the recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class Track:
    """Lightweight representation of a track and its emotional embedding."""

    track_id: str
    title: str
    artist: str
    valence: float
    arousal: float
    tempo: float | None = None
    energy: float | None = None
    danceability: float | None = None


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """Compute cosine similarity between two vectors."""
    raise NotImplementedError("Replace with actual cosine similarity logic")
