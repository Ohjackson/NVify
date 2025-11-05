"""Recommendation routines for TwoOfUsTunes."""

from __future__ import annotations

from typing import Iterable, Sequence

from . import blend
from .utils import Track


def recommend(state: dict[str, float], topk: int = 20) -> Sequence[Track]:
    """Return Top-K tracks for a single user."""
    raise NotImplementedError("Implement single-user recommendation logic")


def rerank_two_users(
    e_a: tuple[float, float],
    e_b: tuple[float, float],
    strategy: str = "avg",
    w: float = 0.5,
    topk: int = 20,
) -> Sequence[Track]:
    """Re-rank two recommendation lists based on the chosen blend strategy."""
    raise NotImplementedError("Implement two-user reranking logic using blend strategies")


def ensure_candidate_pool(state: dict[str, float]) -> Iterable[Track]:
    """Provide a candidate pool constrained by valence/arousal proximity."""
    raise NotImplementedError("Hydrate candidate tracks from feature store or index")
