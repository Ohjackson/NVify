"""Emotion state estimation hooks."""

from __future__ import annotations

from typing import Any, Mapping


Observation = Mapping[str, Any]


def estimate(observation: Observation) -> dict[str, float]:
    """Estimate valence/arousal from multi-modal observations."""
    raise NotImplementedError("Fuse inputs and return {'v': float, 'a': float}")
