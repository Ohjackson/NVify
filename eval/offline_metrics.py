"""Offline evaluation scaffolding."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def ndcg_at_k(recommended: Sequence[str], ground_truth: Sequence[str], k: int = 10) -> float:
    """Compute NDCG@K for a single example."""
    raise NotImplementedError("Implement NDCG@K")


def hit_at_k(recommended: Sequence[str], ground_truth: Sequence[str], k: int = 10) -> float:
    """Compute Hit@K for a single example."""
    raise NotImplementedError("Implement Hit@K")


def evaluate_batch(recs_path: Path, gt_path: Path) -> dict[str, float]:
    """Evaluate an offline recommendation batch."""
    raise NotImplementedError("Implement offline evaluation loader and aggregation")
