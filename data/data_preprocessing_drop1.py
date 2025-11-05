"""
============================================================
DATA PREPROCESSING PIPELINE (DROP playcount == 1)
============================================================

기능:
 - playcount == 1 데이터 제거
 - log scaling + per-user percentile normalization
 - 감정 피처(valence, energy) 중심의 음악 데이터 정제
 - track_id 기준으로 hybrid 데이터 병합

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
    df = df[['track_id', 'name', 'artist', 'valence', 'energy']]

    # 결측치 제거
    df = df.dropna(subset=['valence', 'energy'])

    # 0~1 정규화 (이미 0~1 범위면 영향 없음)
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

    # 1️⃣ playcount == 1 인 데이터 제거
    before = len(df)
    df = df[df['playcount'] > 1]
    after = len(df)
    print(f"playcount > 1 필터링 완료: {before:,} → {after:,} 행 남음")

    # 2️⃣ 로그 스케일 변환 (long-tail 완화)
    df['playcount_log'] = np.log1p(df['playcount'])

    # 3️⃣ 사용자별 백분위 정규화 (0~1)
    def percentile_normalize(x):
        return x.rank(pct=True)

    df['rating'] = df.groupby('user_id')['playcount_log'].transform(percentile_normalize)
    df['rating'] = df['rating'].clip(0, 1)

    # 4️⃣ 필요한 컬럼만 남김
    df = df[['user_id', 'track_id', 'rating']]

    # 5️⃣ 중간 통계 출력
    n_users = df['user_id'].nunique()
    n_tracks = df['track_id'].nunique()
    print(f"남은 사용자 수: {n_users:,}")
    print(f"남은 트랙 수: {n_tracks:,}")
    print(f"남은 총 데이터 수: {len(df):,}")

    # 6️⃣ 저장
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
    user_out = os.path.join(data_dir, "user_ratings_drop1.csv")
    hybrid_out = os.path.join(data_dir, "hybrid_drop1.csv")

    # 단계별 실행
    music_df = preprocess_music_info(music_in, music_out)
    user_df = preprocess_user_history(user_in, user_out)
    merge_datasets(user_df, music_df, hybrid_out)

    print("\n모든 전처리 과정이 완료되었습니다. (DROP version)")
