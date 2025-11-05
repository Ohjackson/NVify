"""Lyric embedding extraction scaffolding."""

from pathlib import Path
from typing import Iterable


def extract_lyric_embeddings(input_dir: Path, output_path: Path) -> None:
    """Compute textual embeddings for lyric files and persist them."""
    raise NotImplementedError("Implement lyric embedding extraction")


def main(args: Iterable[str] | None = None) -> None:
    """Optional CLI entry point for lyric embedding extraction."""
    raise NotImplementedError("Parse arguments and call 'extract_lyric_embeddings'")


if __name__ == "__main__":  # pragma: no cover
    main()
