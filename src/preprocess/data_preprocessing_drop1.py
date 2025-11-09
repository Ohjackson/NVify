"""
============================================================
DATA PREPROCESSING PIPELINE (DROP playcount == 1)
============================================================

기능:
 - playcount == 1 데이터 제거
 - log scaling + per-user percentile normalization
 - 감정 피처(valence, energy) 중심의 음악 데이터 정제
 - track_id 기준으로 hybrid 데이터 병합
 - track_id drop 하고 spotify_id 피쳐 사용

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

    # 'spotify_id'를 포함하여 선택
    df = df[['track_id', 'spotify_id', 'name', 'artist', 'valence', 'energy']]

    # spotify_id가 없는 트랙은 이후 병합에 사용할 수 없으므로 제거
    df = df.dropna(subset=['valence', 'energy', 'spotify_id'])

    # 0~1 정규화
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

    # playcount == 1 인 데이터 제거
    before = len(df)
    df = df[df['playcount'] > 1]
    after = len(df)
    print(f"playcount > 1 필터링 완료: {before:,} → {after:,} 행 남음")

    # 로그 스케일 변환 (long-tail 완화)
    df['playcount_log'] = np.log1p(df['playcount'])

    # 사용자별 백분위 정규화 (0~1)
    def percentile_normalize(x):
        return x.rank(pct=True)

    df['rating'] = df.groupby('user_id')['playcount_log'].transform(percentile_normalize)
    df['rating'] = df['rating'].clip(0, 1)

    # 필요한 컬럼만 남김
    df = df[['user_id', 'track_id', 'rating']]

    # 중간 통계 출력
    n_users = df['user_id'].nunique()
    n_tracks = df['track_id'].nunique()
    print(f"남은 사용자 수: {n_users:,}")
    print(f"남은 트랙 수: {n_tracks:,}")
    print(f"남은 총 데이터 수: {len(df):,}")

    # 저장
    df.to_csv(path_out, index=False, encoding='utf-8-sig')
    print(f"User history 전처리 완료 → {path_out}")

    return df

# ------------------------------------------------------------
# 3. HYBRID DATA MERGE (track_id 기준)
# ------------------------------------------------------------
def merge_datasets(user_df, music_df, path_out):
    print("\n[3] Hybrid Data 병합 시작")

    # 'track_id'를 기준으로 우선 병합
    merged = user_df.merge(music_df, on='track_id', how='inner')

    merged = merged.drop(columns=['track_id'])
    
    # 컬럼 순서 변경
    cols = ['spotify_id', 'user_id', 'rating', 'name', 'artist', 'valence', 'energy', 'emotion_vector']
    cols_to_use = [c for c in cols if c in merged.columns]
    merged = merged[cols_to_use]

    print("병합 및 ID 교체 완료. shape:", merged.shape)
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

    # 출력 파일 경로
    music_out = os.path.join(data_dir, "music_emotion_clean.csv")
    user_out = os.path.join(data_dir, "user_ratings_drop1.csv")
    hybrid_out = os.path.join(data_dir, "hybrid_drop1.csv")

    # 단계별 실행
    print("--- 전처리 파이프라인 시작 (data 폴더 기준) ---")
    music_df = preprocess_music_info(music_in, music_out)
    user_df = preprocess_user_history(user_in, user_out)
    merge_datasets(user_df, music_df, hybrid_out)

    print(f"최종 하이브리드 데이터가 '{hybrid_out}'에 저장되었습니다.")
