"""Feedback ingestion scaffolding."""

from __future__ import annotations

from typing import Iterable, Mapping


FeedbackEvent = Mapping[str, object]


def record_events(events: Iterable[FeedbackEvent]) -> None:
    """Persist user feedback events."""
    raise NotImplementedError("Implement feedback storage or streaming hook")


def compute_rewards(events: Iterable[FeedbackEvent]) -> float:
    """Convert feedback events to a scalar reward signal."""
    raise NotImplementedError("Derive reward signal for online updates")
