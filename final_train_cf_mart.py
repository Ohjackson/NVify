"""Utility script mirroring the original final_train_cf_mart.ipynb pipeline."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD

LOGGER = logging.getLogger("nvify.final_train_cf")
FEATURE_COLUMNS = ["taste_score", "emotion_score", "novelty_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CF + LambdaMART 학습 스크립트")
    parser.add_argument("--input_csv", default="data/processed/hybrid_drop1.csv", help="학습 입력 CSV 경로")
    parser.add_argument("--tracks_csv", default="data/processed/tracks.csv", help="곡 특성(tracks) CSV 경로")
    parser.add_argument("--artifacts_dir", default="data/artifacts", help="아티팩트 저장 디렉터리")
    parser.add_argument("--cf_model_name", default="cf_model_final.pkl", help="CF 모델 파일명")
    parser.add_argument("--track_meta_name", default="track_meta_db.pkl", help="트랙 메타 DB 파일명")
    parser.add_argument("--ranking_model_name", default="ranking_model_final.pkl", help="랭킹 모델 파일명")
    parser.add_argument("--rating_scale_min", type=float, default=0.0, help="레이팅 최소값")
    parser.add_argument("--rating_scale_max", type=float, default=1.0, help="레이팅 최대값")
    parser.add_argument("--rating_threshold", type=float, default=0.5, help="라벨 기준 임계값")
    parser.add_argument("--n_factors", type=int, default=100, help="SVD 잠재요인 수")
    parser.add_argument("--n_epochs", type=int, default=20, help="SVD 학습 epoch")
    parser.add_argument("--rank_estimators", type=int, default=500, help="LambdaMART 트리 개수")
    parser.add_argument("--rank_lr", type=float, default=0.05, help="LambdaMART 학습률")
    parser.add_argument("--random_state", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--user_emotion_valence", type=float, default=0.5, help="사용자 Valence(가상)")
    parser.add_argument("--user_emotion_energy", type=float, default=0.5, help="사용자 Energy(가상)")
    parser.add_argument("--skip_ranking", action="store_true", help="LambdaMART 학습 건너뛰기")
    parser.add_argument("--max_rows", type=int, default=0, help="학습에 사용할 최대 행 수 (0이면 전체)")
    return parser.parse_args()


def load_dataset(path: Path, tracks_path: Path | None = None, max_rows: int = 0, random_state: int = 42) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"입력 CSV를 찾을 수 없습니다: {path}")
    df = pd.read_csv(path)
    # Accept datasets with either spotify_id or track_id; normalize to spotify_id when missing
    base_required = {"user_id", "rating"}
    if not base_required.issubset(df.columns):
        missing = sorted(base_required - set(df.columns))
        raise ValueError(f"필수 컬럼 누락 {path}: {missing}")

    has_spotify = "spotify_id" in df.columns
    has_track = "track_id" in df.columns
    if not has_spotify and not has_track:
        raise ValueError(f"spotify_id/track_id 둘 다 없음: {path}")

    if not has_spotify and has_track:
        # Create a string spotify_id surrogate from track_id so downstream stays consistent
        df["spotify_id"] = df["track_id"].astype(str)

    needs_emotions = any(col not in df.columns for col in ("valence", "energy"))
    if needs_emotions:
        if tracks_path is None or not Path(tracks_path).exists():
            raise ValueError("Valence/Energy가 없고 tracks CSV도 없어 병합할 수 없습니다.")
        tracks_df = pd.read_csv(tracks_path)
        # Allow tracks.csv without spotify_id: align on track_id and then map to surrogate spotify_id
        if {"spotify_id", "valence", "energy"}.issubset(tracks_df.columns):
            right = tracks_df[["spotify_id", "valence", "energy"]]
            on_col = "spotify_id"
        elif {"track_id", "valence", "energy"}.issubset(tracks_df.columns):
            right = tracks_df[["track_id", "valence", "energy"]].rename(columns={"track_id": "spotify_id"})
            on_col = "spotify_id"
        else:
            raise ValueError("tracks CSV에는 (spotify_id 또는 track_id), valence, energy 컬럼이 필요합니다")

        df = df.merge(right, on=on_col, how="left")

    required_emotion = {"valence", "energy"}
    missing_emotion = [col for col in required_emotion if col not in df.columns]
    if missing_emotion:
        raise ValueError(f"Could not resolve emotion columns: {missing_emotion}")

    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)
        LOGGER.info("데이터 샘플링 적용 (행=%d)", len(df))
    LOGGER.info("데이터 로드 완료 %s (행=%d)", path, len(df))
    return df


def train_cf_model(df: pd.DataFrame, rating_scale: tuple[float, float], n_factors: int, n_epochs: int, random_state: int) -> SVD:
    reader = Reader(rating_scale=rating_scale)
    dataset = Dataset.load_from_df(df[["user_id", "spotify_id", "rating"]], reader)
    trainset = dataset.build_full_trainset()
    LOGGER.info("Surprise SVD 학습 중 (n_factors=%d, n_epochs=%d)", n_factors, n_epochs)
    model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state, verbose=False)
    model.fit(trainset)
    return model


def build_track_meta(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    rating_counts = df["spotify_id"].value_counts().reset_index()
    rating_counts.columns = ["spotify_id", "total_rating_count"]
    track_info = df[["spotify_id", "valence", "energy"]].drop_duplicates(subset=["spotify_id"])
    meta_df = pd.merge(track_info, rating_counts, on="spotify_id", how="left")
    meta_df["total_rating_count"] = meta_df["total_rating_count"].fillna(0).astype(int)
    LOGGER.info("트랙 메타 DB 구축 완료 (개수=%d)", len(meta_df))
    return meta_df.set_index("spotify_id").to_dict("index")


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)
    LOGGER.info("아티팩트 저장: %s", path)


def build_ltr_features(
    dataframe: pd.DataFrame,
    cf_model: SVD,
    meta_db: Dict[str, Dict[str, float]],
    user_emotion_v: float,
    user_emotion_a: float,
    rating_threshold: float,
) -> pd.DataFrame:
    df = dataframe.copy()
    LOGGER.info("LTR 피처 생성 중 (행=%d)", len(df))
    df["taste_score"] = df.apply(lambda row: cf_model.predict(row["user_id"], row["spotify_id"]).est, axis=1)

    def calculate_cb_scores(row):
        meta = meta_db.get(row["spotify_id"], {})
        track_valence = row.get("valence", np.nan)
        track_energy = row.get("energy", np.nan)
        distance = np.sqrt((user_emotion_v - track_valence) ** 2 + (user_emotion_a - track_energy) ** 2)
        emotion_score = 1 / (1 + distance)
        novelty_score = 1 / (meta.get("total_rating_count", 0) + 1)
        label = int(row.get("rating", 0) >= rating_threshold)
        return pd.Series([emotion_score, novelty_score, label])

    df[["emotion_score", "novelty_score", "label"]] = df.apply(calculate_cb_scores, axis=1)
    df["query_id"] = df["user_id"].astype("category").cat.codes
    ltr_df = df.dropna(subset=FEATURE_COLUMNS + ["label"])
    LOGGER.info("LTR 피처 데이터셋 준비 완료 (행=%d)", len(ltr_df))
    return ltr_df


def train_lambda_mart(ltr_df: pd.DataFrame, n_estimators: int, learning_rate: float, random_state: int) -> lgb.LGBMRanker:
    X = ltr_df[FEATURE_COLUMNS]
    y = ltr_df["label"]
    group = ltr_df.groupby("query_id").size().tolist()
    LOGGER.info("LambdaMART 학습 (행=%d, 그룹=%d)", len(ltr_df), len(group))
    model = lgb.LGBMRanker(
        objective="lambdarank",
        device="cpu",
        random_state=random_state,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        verbose=-1,
    )
    model.fit(X=X, y=y, group=group)
    return model


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    data_path = Path(args.input_csv)
    artifacts_dir = Path(args.artifacts_dir)
    tracks_path = Path(args.tracks_csv) if args.tracks_csv else None
    df = load_dataset(data_path, tracks_path, max_rows=args.max_rows, random_state=args.random_state)

    rating_scale = (args.rating_scale_min, args.rating_scale_max)
    cf_model = train_cf_model(df, rating_scale, args.n_factors, args.n_epochs, args.random_state)
    save_pickle(cf_model, artifacts_dir / args.cf_model_name)

    track_meta_db = build_track_meta(df)
    save_pickle(track_meta_db, artifacts_dir / args.track_meta_name)

    if args.skip_ranking:
        LOGGER.info("Skipping LambdaMART training per --skip_ranking")
        return

    ltr_df = build_ltr_features(
        df,
        cf_model,
        track_meta_db,
        args.user_emotion_valence,
        args.user_emotion_energy,
        args.rating_threshold,
    )

    ranking_model = train_lambda_mart(ltr_df, args.rank_estimators, args.rank_lr, args.random_state)
    save_pickle(ranking_model, artifacts_dir / args.ranking_model_name)


if __name__ == "__main__":
    main()
