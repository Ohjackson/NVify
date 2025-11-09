"""Loader/adapter for SVD CF artifacts produced outside of this refactor."""

from __future__ import annotations

import logging
import pickle
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

LOGGER = logging.getLogger("nvify.svd_entry")

MODEL_CANDIDATES = (
    "cf_model_final.pkl",
    "cf_model_svd.pkl",
    "svd_model.pkl",
)

TRAINING_SCRIPT_NAME = "final_train_cf_mart.py"


def _find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            LOGGER.info("CF 모델 아티팩트 발견: %s", path)
            return path
    return None


@contextmanager
def _temporary_argv(argv):
    original = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original


def _select_input_csv(cfg: dict) -> Optional[Path]:
    processed_dir = Path(cfg["paths"]["processed_dir"])
    candidates = [processed_dir / "hybrid_drop1.csv", processed_dir / "interactions.csv"]
    for path in candidates:
        if path.exists():
            return path
    return None


def _run_training_script(repo_root: Path, cfg: dict) -> None:
    candidate = repo_root / TRAINING_SCRIPT_NAME
    if not candidate.exists():
        LOGGER.warning("학습 스크립트를 찾을 수 없습니다. 재학습 불가: %s", candidate)
        return

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    tracks_csv = Path(cfg["paths"]["processed_dir"]) / "tracks.csv"
    argv = [candidate.name, "--artifacts_dir", str(artifacts_dir)]

    input_csv = _select_input_csv(cfg)
    if input_csv:
        argv.extend(["--input_csv", str(input_csv)])
    if tracks_csv.exists():
        argv.extend(["--tracks_csv", str(tracks_csv)])

    LOGGER.info("CF 학습 스크립트 실행: %s", " ".join(argv))
    with _temporary_argv(argv):
        runpy.run_path(str(candidate), run_name="__main__")


def _load_pickle(path: Path):
    LOGGER.info("CF 모델 로드: %s", path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_model(cfg: dict):
    """CF 모델 아티팩트 로드. 없으면 스크립트로 학습 시도."""

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    candidates = [artifacts_dir / name for name in MODEL_CANDIDATES]
    model_path = _find_first_existing(candidates)

    if model_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        LOGGER.info("CF 모델 아티팩트 없음. 학습 스크립트 실행 시도.")
        _run_training_script(repo_root, cfg)
        model_path = _find_first_existing(candidates)

    if model_path:
        try:
            return _load_pickle(model_path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("모델 로드 실패 %s: %s", model_path, exc)
            return None

    LOGGER.warning("학습 시도 후에도 CF 모델 아티팩트가 없습니다.")
    return None


__all__ = ["load_model"]
