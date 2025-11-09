import argparse
import logging
import os
import runpy
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


def _safe_import(module_name: str):
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        LOGGER.debug("모듈 없음: %s", module_name)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("모듈 임포트 실패 %s: %s", module_name, exc)
    return None


def _run_script(script_path: Path) -> None:
    LOGGER.info("스크립트 폴백 실행: %s", script_path)
    runpy.run_path(str(script_path), run_name="__main__")


def _exec_module_or_script(module_name: str, script_path: Path, func_candidates=("main", "run"), kwargs=None) -> None:
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
    _run_script(script_path)


def _run_preprocess(cfg: dict) -> None:
    processed_dir = Path(cfg["paths"]["processed_dir"])
    inter_path = processed_dir / "interactions.csv"
    track_path = processed_dir / "tracks.csv"

    if inter_path.exists() and track_path.exists():
        LOGGER.info("전처리 산출물이 이미 있습니다. 전처리 건너뜁니다.")
        return

    pipeline_path = REPO / "src" / "preprocess" / "data_preprocessing_pipeline.py"
    module = _safe_import("src.preprocess.data_preprocessing_pipeline")
    if module:
        for candidate in ("main", "run"):
            entry = getattr(module, candidate, None)
            if callable(entry):
                LOGGER.info("전처리 파이프라인 실행: %s.%s", module.__name__, candidate)
                entry()
                break
        if inter_path.exists() and track_path.exists():
            return
    else:
        _run_script(pipeline_path)
        if inter_path.exists() and track_path.exists():
            return

    steps = [
        ("src.preprocess.data_pull", REPO / "src" / "preprocess" / "data_pull.py"),
        ("src.preprocess.lastfm_preprocessing", REPO / "src" / "preprocess" / "lastfm_preprocessing.py"),
        ("src.preprocess.merge_preprocessed", REPO / "src" / "preprocess" / "merge_preprocessed.py"),
        ("src.preprocess.data_preprocessing_drop1", REPO / "src" / "preprocess" / "data_preprocessing_drop1.py"),
    ]

    for module_name, script_path in steps:
        if inter_path.exists() and track_path.exists():
            break
        _exec_module_or_script(module_name, script_path)

    if not (inter_path.exists() and track_path.exists()):
        LOGGER.warning("전처리 산출물이 생성되지 않았습니다: %s, %s", inter_path, track_path)


def _run_mapping(cfg: dict) -> None:
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
    _run_preprocess(cfg)
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
