"""Blend strategies for reconciling two emotional states."""

from __future__ import annotations

from typing import Iterable, Sequence

from .utils import Track


Strategy = str


def blend_average(
    user_a: Iterable[Track], user_b: Iterable[Track], weight: float = 0.5
) -> Sequence[Track]:
    """Average rankings or scores from two users."""
    raise NotImplementedError("Implement weighted average blending")


def blend_intersection(user_a: Iterable[Track], user_b: Iterable[Track]) -> Sequence[Track]:
    """Return the intersection of two playlists prioritising shared items."""
    raise NotImplementedError("Implement intersection-based blending")


def blend_union(user_a: Iterable[Track], user_b: Iterable[Track]) -> Sequence[Track]:
    """Return a union of two playlists while preserving diversity constraints."""
    raise NotImplementedError("Implement union-based blending")


STRATEGIES: dict[Strategy, callable] = {
    "avg": blend_average,
    "intersect": blend_intersection,
    "union": blend_union,
}


def resolve(strategy: Strategy) -> callable:
    """Return the strategy implementation callable."""
    try:
        return STRATEGIES[strategy]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported strategy: {strategy}") from exc
