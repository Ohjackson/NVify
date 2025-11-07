# ==========================================
# NVify Recommendation Pipeline
# Author: Team NVify (Ohjackson, Kim Hobeom)
# Description:
#   Loads trained CF, MetaDB, and LambdaMART models (.pkl)
#   Generates ranked music recommendations for a given user
# ==========================================

import pickle
import numpy as np
import pandas as pd
import argparse
from typing import List


# -----------------------------
# (1) Load Trained Assets
# -----------------------------
# recommender.py
def load_assets(cf_path='cf_model_final.pkl',
                meta_path='track_meta_db.pkl',
                rank_path='ranking_model_final.pkl',
                # ⭐ 매핑 파일 경로 인자 추가
                mapping_path='lastfm_to_spotify_mapping.pkl'): 
    print("[INFO] Loading trained models and metadata...")
    try:
        # 기존 모델 로드
        cf_model = pickle.load(open(cf_path, 'rb'))
        meta_db = pickle.load(open(meta_path, 'rb'))
        ranker = pickle.load(open(rank_path, 'rb'))
        # ⭐ 매핑 DB 로드
        mapping_db = pickle.load(open(mapping_path, 'rb')) 
    except FileNotFoundError as e:
        print(f"[ERROR] Missing required file: {e}")
        raise SystemExit

    print("   - ... all assets loaded successfully.")
    # ⭐ mapping_db를 함께 반환
    return cf_model, meta_db, ranker, mapping_db

# -----------------------------
# (2) Compute Features per Track
# -----------------------------
def compute_features(user_id: str,
                     candidate_tracks: List[str], # Last.fm ID 목록
                     cf_model,
                     meta_db,
                     mapping_db, # ⭐ 매핑 DB 인자 추가
                     user_valence=0.5,
                     user_energy=0.5):
    """
    Compute features after mapping Last.fm ID to Spotify ID.
    """
    rows = []
    for lastfm_tid in candidate_tracks: # Last.fm ID 사용

        # ⭐⭐ 핵심 로직: Spotify ID로 변환
        spotify_tid = mapping_db.get(lastfm_tid, None)
        
        if spotify_tid is None:
            # 매핑에 실패한 4%의 트랙은 건너뜁니다.
            continue 

        # 이제 모든 특징 계산은 Spotify ID를 사용합니다.
        
        # taste_score (from CF model)
        try:
            taste = cf_model.predict(user_id, spotify_tid).est # Spotify ID 사용
        except Exception:
            taste = 0.5 

        # track metadata (valence, energy, rating count)
        meta = meta_db.get(spotify_tid, {'valence': 0.5, 'energy': 0.5, 'total_rating_count': 1}) # Spotify ID 사용
        valence, energy = meta.get('valence', 0.5), meta.get('energy', 0.5)
        count = meta.get('total_rating_count', 1)

        # emotion_score, novelty_score 계산 로직은 동일
        distance = np.sqrt((user_valence - valence) ** 2 + (user_energy - energy) ** 2)
        emotion = 1 / (1 + distance)
        novelty = 1 / (count + 1)

        # 최종 DF에는 원래의 Last.fm ID를 기록하여 테스트 데이터와 매칭합니다.
        rows.append([lastfm_tid, taste, emotion, novelty]) 

    df = pd.DataFrame(rows, columns=['track_id', 'taste_score', 'emotion_score', 'novelty_score'])
    return df


# -----------------------------
# (3) Rank Tracks with LambdaMART
# -----------------------------
def rank_tracks(df_features: pd.DataFrame, ranker, topK=20):
    feature_cols = ['taste_score', 'emotion_score', 'novelty_score']
    df_features['final_score'] = ranker.predict(df_features[feature_cols])
    ranked = df_features.sort_values('final_score', ascending=False).head(topK)
    return ranked


# -----------------------------
# (4) Full Recommendation Pipeline
# -----------------------------
def recommend(user_id: str,
              candidate_tracks: List[str],
              cf_model, meta_db, ranker, mapping_db,
              user_valence=0.5,
              user_energy=0.5,
              topK=20):
    
    
    print(f"[INFO] Generating recommendations for user '{user_id}' ...")

    # mapping_db를 compute_features에 전달
    df_features = compute_features(user_id, candidate_tracks, cf_model, meta_db, mapping_db,
                                   user_valence=user_valence, user_energy=user_energy)
    ranked = rank_tracks(df_features, ranker, topK=topK)

    print("\n===== Top Recommendations =====")
    print(ranked[['track_id', 'final_score', 'taste_score', 'emotion_score', 'novelty_score']])
    return ranked


# -----------------------------
# (5) CLI Execution
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NVify Recommendation Pipeline")
    parser.add_argument("--user", type=str, required=True, help="User ID for recommendation")
    parser.add_argument("--valence", type=float, default=0.5, help="User emotional valence (0~1)")
    parser.add_argument("--energy", type=float, default=0.5, help="User emotional energy (0~1)")
    parser.add_argument("--topk", type=int, default=20, help="Number of recommendations to output")
    parser.add_argument("--limit", type=int, default=1000, help="Number of candidate tracks to evaluate")
    args = parser.parse_args()

    # load metadata to get candidate tracks
    _, meta_db, _ = load_assets()
    candidate_tracks = list(meta_db.keys())[:args.limit]

    recommend(args.user, candidate_tracks,
              user_valence=args.valence,
              user_energy=args.energy,
              topK=args.topk)
