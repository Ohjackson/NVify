import argparse
import logging
import os
import runpy
import shutil
import sys
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.cf.svd_entry_shim import load_model

LOGGER = logging.getLogger("nvify.main")


def _parse_args():
    parser = argparse.ArgumentParser(description="NVify 파이프라인: 전처리 → 매핑 → CF → 추천 → 평가")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    parser.add_argument("--user_id", default="u0", help="추천 대상 사용자 ID")
    parser.add_argument("--valence", type=float, help="사용자 선호 Valence (미지정 시 기본값)")
    parser.add_argument("--energy", type=float, help="사용자 선호 Energy (미지정 시 기본값)")
    parser.add_argument("--top_k", type=int, help="추천 개수 (미지정 시 기본값)")
    parser.add_argument("--eval", action="store_true", help="평가 단계 실행 여부")
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _ensure_dirs(cfg: dict) -> None:
    for key in ("raw_dir", "processed_dir", "artifacts_dir"):
        target = Path(cfg["paths"][key])
        target.mkdir(parents=True, exist_ok=True)
        LOGGER.debug("디렉터리 보장: %s", target)


def _prepare_raw_inputs(cfg: dict) -> None:
    """Copy user-provided raw CSVs into locations expected by legacy scripts."""

    raw_dir = Path(cfg["paths"]["raw_dir"])
    data_root = REPO / "data"
    data_root.mkdir(exist_ok=True)

    mapping = {
        "Music Info.csv": "music_info.csv",
        "music_info.csv": "music_info.csv",
        "User Listening History.csv": "user_listening_history.csv",
        "user_listening_history.csv": "user_listening_history.csv",
    }

    for src_name, dest_name in mapping.items():
        src = raw_dir / src_name
        if not src.exists():
            continue
        dest = data_root / dest_name
        shutil.copy2(src, dest)
        LOGGER.info("RAW → DATA 복사: %s → %s", src, dest)


def _sync_processed_outputs(cfg: dict) -> None:
    data_root = REPO / "data"
    processed_dir = Path(cfg["paths"]["processed_dir"])

    mapping = {
        data_root / "music_emotion_clean.csv": processed_dir / "tracks.csv",
        data_root / "user_ratings_normalized.csv": processed_dir / "interactions.csv",
        data_root / "hybrid_preprocessed.csv": processed_dir / "hybrid_drop1.csv",
    }

    for src, dest in mapping.items():
        if not src.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        LOGGER.info("전처리 산출물 복사: %s → %s", src, dest)


def _safe_import(module_name: str):
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        LOGGER.debug("모듈 없음: %s", module_name)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("모듈 임포트 실패 %s: %s", module_name, exc)
    return None


def _run_script(script_path: Path, ignore_errors: bool = False) -> None:
    LOGGER.info("스크립트 폴백 실행: %s", script_path)
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except ModuleNotFoundError as exc:
        if ignore_errors:
            LOGGER.warning("의존성 부재로 스크립트를 건너뜁니다 (%s): %s", script_path, exc)
            return
        raise


def _exec_module_or_script(
    module_name: str,
    script_path: Path,
    func_candidates=("main", "run"),
    kwargs=None,
    ignore_errors: bool = False,
) -> None:
    module = _safe_import(module_name)
    if module:
        for candidate in func_candidates:
            entry = getattr(module, candidate, None)
            if callable(entry):
                LOGGER.info("모듈 함수 실행: %s.%s", module_name, candidate)
                if kwargs:
                    entry(**kwargs)
                else:
                    entry()
                return
    _run_script(script_path, ignore_errors=ignore_errors)


def _run_preprocess(cfg: dict) -> None:
    processed_dir = Path(cfg["paths"]["processed_dir"])
    inter_path = processed_dir / "interactions.csv"
    track_path = processed_dir / "tracks.csv"
    alt_inter = REPO / "data" / "user_ratings_normalized.csv"
    alt_track = REPO / "data" / "music_emotion_clean.csv"

    def _outputs_ready() -> bool:
        return (inter_path.exists() and track_path.exists()) or (alt_inter.exists() and alt_track.exists())

    if _outputs_ready():
        LOGGER.info("전처리 산출물이 이미 있습니다. 전처리 건너뜁니다.")
        return

    pipeline_path = REPO / "src" / "preprocess" / "data_preprocessing_pipeline.py"
    module = _safe_import("src.preprocess.data_preprocessing_pipeline")
    if module:
        invoked = False
        for candidate in ("main", "run"):
            entry = getattr(module, candidate, None)
            if callable(entry):
                LOGGER.info("전처리 파이프라인 실행: %s.%s", module.__name__, candidate)
                entry()
                invoked = True
                break
        if not invoked:
            _run_script(pipeline_path)
    else:
        _run_script(pipeline_path)

    if _outputs_ready():
        return

    steps = [
        ("src.preprocess.data_pull", REPO / "src" / "preprocess" / "data_pull.py", True),
        ("src.preprocess.lastfm_preprocessing", REPO / "src" / "preprocess" / "lastfm_preprocessing.py", True),
        ("src.preprocess.merge_preprocessed", REPO / "src" / "preprocess" / "merge_preprocessed.py", False),
        ("src.preprocess.data_preprocessing_drop1", REPO / "src" / "preprocess" / "data_preprocessing_drop1.py", False),
    ]

    for module_name, script_path, ignore_errors in steps:
        if _outputs_ready():
            break
        _exec_module_or_script(module_name, script_path, ignore_errors=ignore_errors)

    if not _outputs_ready():
        LOGGER.warning("전처리 산출물이 생성되지 않았습니다: %s, %s", inter_path, track_path)


def _run_mapping(cfg: dict) -> None:
    processed_dir = Path(cfg["paths"]["processed_dir"])
    tracks_csv = processed_dir / "tracks.csv"
    try:
        if tracks_csv.exists():
            import pandas as pd  # local import to avoid heavy dependency at module load

            header = pd.read_csv(tracks_csv, nrows=0)
            if "spotify_id" in header.columns:
                LOGGER.info("tracks.csv에 spotify_id 컬럼이 이미 있습니다. 매핑 단계를 건너뜁니다.")
                return
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Spotify ID 확인 중 오류: %s", exc)

    merge_path = REPO / "src" / "mapping" / "merge_spotify_id.py"
    map_path = REPO / "src" / "mapping" / "map_spotify_ids.py"

    _exec_module_or_script("src.mapping.merge_spotify_id", merge_path)
    _exec_module_or_script("src.mapping.map_spotify_ids", map_path)


@contextmanager
def _temp_env(env_vars: dict):
    previous = {}
    try:
        for key, value in env_vars.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_recommend(cfg: dict, user_id: str, valence: float, energy: float, top_k: int) -> None:
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    env_payload = {
        "NVIFY_USER_ID": user_id,
        "NVIFY_VALENCE": valence,
        "NVIFY_ENERGY": energy,
        "NVIFY_TOP_K": top_k,
        "NVIFY_ARTIFACTS_DIR": str(artifacts_dir),
    }

    def _call_rec(module_name: str, script_path: Path):
        module = _safe_import(module_name)
        if module:
            for candidate in ("main", "run", "generate_recommendations", "recommend"):
                entry = getattr(module, candidate, None)
                if callable(entry):
                    LOGGER.info("추천 모듈 실행: %s.%s", module_name, candidate)
                    entry(user_id=user_id, valence=valence, energy=energy, top_k=top_k, cfg=cfg)
                    return True
        with _temp_env(env_payload):
            _run_script(script_path)
        return True

    serve_dir = REPO / "src" / "serve"
    if _call_rec("src.serve.recommender", serve_dir / "recommender.py"):
        return
    _call_rec("src.serve.recommend", serve_dir / "recommend.py")


def _run_eval() -> None:
    eval_path = REPO / "src" / "eval" / "recommend_evaluate.py"
    module = _safe_import("src.eval.recommend_evaluate")
    if module:
        for candidate in ("main", "run", "evaluate"):
            entry = getattr(module, candidate, None)
            if callable(entry):
                LOGGER.info("평가 모듈 실행: %s.%s", module.__name__, candidate)
                entry()
                return
    _run_script(eval_path)


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)

    _ensure_dirs(cfg)
    _prepare_raw_inputs(cfg)
    _run_preprocess(cfg)
    _sync_processed_outputs(cfg)
    _run_mapping(cfg)

    model = load_model(cfg)
    if model is None:
        LOGGER.warning("CF 모델 로드 실패. 후속 스크립트에서 학습을 시도할 수 있습니다.")

    valence = args.valence if args.valence is not None else cfg["defaults"]["valence"]
    energy = args.energy if args.energy is not None else cfg["defaults"]["energy"]
    top_k = args.top_k if args.top_k is not None else cfg["serve"]["top_k"]

    _run_recommend(cfg, args.user_id, valence, energy, top_k)

    if args.eval:
        _run_eval()

    rec_path = Path(cfg["paths"]["artifacts_dir"]) / "recommendations.csv"
    if rec_path.exists():
        LOGGER.info("추천 결과 파일 생성됨: %s", rec_path)
    else:
        LOGGER.warning("추천 결과 파일이 보이지 않습니다: %s", rec_path)


if __name__ == "__main__":
    main()
