"""Audio feature extraction entry point.

This module provides a CLI-style function for turning raw audio assets into
processed tabular features stored on disk. Implementations are left as TODOs so
contributors can plug in their preferred tooling (e.g., librosa).
"""

from pathlib import Path
from typing import Iterable


def extract_audio_features(input_dir: Path, output_path: Path) -> None:
    """Extract features from audio files in ``input_dir`` and save to ``output_path``.

    Args:
        input_dir: Directory containing raw audio assets (e.g., WAV/MP3).
        output_path: Destination file (e.g., Parquet/CSV) for extracted features.
    """
    raise NotImplementedError("Implement audio feature extraction pipeline")


def main(args: Iterable[str] | None = None) -> None:
    """Optional CLI wrapper around :func:`extract_audio_features`."""
    raise NotImplementedError("Parse arguments and call 'extract_audio_features'")


if __name__ == "__main__":  # pragma: no cover
    main()
