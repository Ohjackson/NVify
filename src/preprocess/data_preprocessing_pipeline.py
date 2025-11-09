"""
============================================================
DATA PREPROCESSING PIPELINE for Emotion-based Music Recommender
============================================================

1. music_info.csv  →  music_emotion_clean.csv
   - 감정 피처(valence, energy) 중심 정제

2. user_listening_history.csv  →  user_ratings_normalized.csv
   - playcount → log scaling + percentile normalization (per user)

3. 병합 (track_id 기준) → hybrid_preprocessed.csv
   - 감정 정보 + 사용자 rating 결합

------------------------------------------------------------
Project: NVify Term Project (Emotion-based Music Recommender)
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


# ------------------------------------------------------------
# 1. MUSIC INFO PREPROCESSING (Content-based)
# ------------------------------------------------------------
def preprocess_music_info(path_in, path_out):
    print("[1] Music Info Preprocessing 시작")

    df = pd.read_csv(path_in)
    print("원본 shape:", df.shape)

    # 주요 피처만 선택
    keep_cols = ['track_id', 'spotify_id', 'name', 'artist', 'valence', 'energy']
    available_cols = [col for col in keep_cols if col in df.columns]
    df = df[available_cols]

    # 결측치 제거
    df = df.dropna(subset=['valence', 'energy'])

    # 0~1 정규화 (이미 0~1이면 영향 없음)
    scaler = MinMaxScaler()
    df[['valence', 'energy']] = scaler.fit_transform(df[['valence', 'energy']])

    # 감정 벡터 추가
    df['emotion_vector'] = df[['valence', 'energy']].values.tolist()

    # 저장
    df.to_csv(path_out, index=False, encoding='utf-8-sig')
    print(f"Music info 전처리 완료 → {path_out}")
    return df


# ------------------------------------------------------------
# 2. USER LISTENING HISTORY PREPROCESSING (Collaborative-based)
# ------------------------------------------------------------
def preprocess_user_history(path_in, path_out):
    print("\n[2] User Listening History Preprocessing 시작")

    df = pd.read_csv(path_in)
    print("원본 shape:", df.shape)

    # 로그 스케일로 long-tail 완화
    df['playcount_log'] = np.log1p(df['playcount'])

    # 사용자별 백분위(percentile) 기반 정규화
    def percentile_normalize(x):
        return x.rank(pct=True)  # 0~1 범위 자동 매핑

    df['rating'] = df.groupby('user_id')['playcount_log'].transform(percentile_normalize)

    # 클리핑 (안정성 확보)
    df['rating'] = df['rating'].clip(0, 1)

    # 필요한 컬럼만 남김
    df = df[['user_id', 'track_id', 'rating']]

    # 저장
    df.to_csv(path_out, index=False, encoding='utf-8-sig')
    print(f"User history 전처리 완료 → {path_out}")
    return df


# ------------------------------------------------------------
# 3. HYBRID DATA MERGE (track_id 기준)
# ------------------------------------------------------------
def merge_datasets(user_df, music_df, path_out):
    print("\n[3] Hybrid Data 병합 시작")

    merged = user_df.merge(music_df, on='track_id', how='inner')
    print("병합 결과 shape:", merged.shape)

    merged.to_csv(path_out, index=False, encoding='utf-8-sig')
    print(f"Hybrid 데이터 저장 완료 → {path_out}")
    return merged


# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    # 경로 설정
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    music_in = os.path.join(data_dir, "music_info.csv")
    user_in = os.path.join(data_dir, "user_listening_history.csv")

    music_out = os.path.join(data_dir, "music_emotion_clean.csv")
    user_out = os.path.join(data_dir, "user_ratings_normalized.csv")
    hybrid_out = os.path.join(data_dir, "hybrid_preprocessed.csv")

    # 단계별 실행
    music_df = preprocess_music_info(music_in, music_out)
    user_df = preprocess_user_history(user_in, user_out)
    merge_datasets(user_df, music_df, hybrid_out)

    print("\n모든 전처리 과정이 완료되었습니다.")
