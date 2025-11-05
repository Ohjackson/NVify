"""Approximate nearest neighbour index scaffolding."""

from pathlib import Path


def build_index(feature_table: Path, output_dir: Path) -> None:
    """Create an ANN index (e.g., FAISS) from the provided feature table."""
    raise NotImplementedError("Implement ANN index builder")
