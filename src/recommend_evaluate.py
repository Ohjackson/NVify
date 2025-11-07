# ==========================================
# NVify Recommendation Evaluation
# Author: Kim Hobeom (Team NVify)
# Description:
#   Evaluate recommendation performance
#   Metrics: Precision@K, Recall@K, nDCG@K
# ==========================================

import numpy as np
import pandas as pd
import argparse
import pickle
import random
from typing import List

# -----------------------------
# (1) Metric Functions
# -----------------------------
def precision_at_k(y_true, y_pred, k=10):
    y_pred_k = y_pred[:k]
    return np.mean([1 if i in y_true else 0 for i in y_pred_k])

def recall_at_k(y_true, y_pred, k=10):
    y_pred_k = y_pred[:k]
    return len(set(y_true) & set(y_pred_k)) / len(y_true) if len(y_true) > 0 else 0.0

def ndcg_at_k(y_true, y_pred, k=10):
    y_pred_k = y_pred[:k]
    dcg = 0.0
    for i, pid in enumerate(y_pred_k):
        if pid in y_true:
            dcg += 1 / np.log2(i + 2)
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(y_true), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# -----------------------------
# (2) Evaluation Function (수정됨: 감정 값 인자 추가)
# -----------------------------
def evaluate_recommendations(test_df: pd.DataFrame, 
                             recommend_func, 
                             cf_model, meta_db, ranker, mapping_db,
                             users: List[str], 
                             candidate_tracks: List[str], 
                             user_valence: float,       # ⭐ 사용자가 입력한 감정 값
                             user_energy: float,        # ⭐ 사용자가 입력한 감정 값
                             topk=10):
    precisions, recalls, ndcgs = [], [], []
        
    for uid in users:
        gt_tracks = test_df[test_df["user_id"] == uid]
        # '관심 있음' (rating >= 0.5) 트랙을 Positive Label로 사용
        y_true = gt_tracks.loc[gt_tracks["rating"] >= 0.5, "track_id"].tolist()
        
        if not y_true:
            continue

        # ⭐ recommend_func 호출 시 입력받은 감정 값 전달
        recommended = recommend_func(
            user_id=uid, 
            candidate_tracks=candidate_tracks,
            cf_model=cf_model, meta_db=meta_db, ranker=ranker, mapping_db=mapping_db, 
            user_valence=user_valence, 
            user_energy=user_energy,
            topK=topk
        )["track_id"].tolist()

        precisions.append(precision_at_k(y_true, recommended, k=topk))
        recalls.append(recall_at_k(y_true, recommended, k=topk))
        ndcgs.append(ndcg_at_k(y_true, recommended, k=topk))

    results = {
        "Precision@K": np.mean(precisions),
        "Recall@K": np.mean(recalls),
        "nDCG@K": np.mean(ndcgs)
    }
    return results


# -----------------------------
# (3) Example Main (수정됨: CLI 감정 인자 추가 및 사용)
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NVify Recommendation Performance")
    parser.add_argument("--test", type=str, default="test_dataset.csv",
                        help="CSV file containing user_id, track_id, rating columns")
    parser.add_argument("--k", type=int, default=10, help="Cutoff K for metrics")
    parser.add_argument("--limit", type=int, default=1000, help="Number of candidate tracks to evaluate")
    parser.add_argument("--valence", type=float, default=0.5, 
                        help="User emotional valence (0~1) for all evaluated users") # ⭐ 추가
    parser.add_argument("--energy", type=float, default=0.5, 
                        help="User emotional energy (0~1) for all evaluated users") # ⭐ 추가
    args = parser.parse_args()

    # Load required modules and assets
    from recommender import recommend, load_assets
    import random
    
    cf_model, meta_db, ranker, mapping_db = load_assets()

    candidate_tracks = list(mapping_db.keys())

    # CLI 인자 limit에 따라 후보 트랙 수를 제한합니다.
    limit = args.limit if args.limit > 0 else len(candidate_tracks)
    if len(candidate_tracks) > limit:
        candidate_tracks = random.sample(candidate_tracks, limit)

    # Load your test dataset
    df_test = pd.read_csv(args.test)
    
    # 유니크 유저 샘플링 (평가 속도 위해)
    users = random.sample(sorted(df_test["user_id"].unique().tolist()), 30)

    print(f"[INFO] Evaluating on {len(users)} sampled users, K={args.k}, Candidates={len(candidate_tracks)}")
    print(f"[INFO] Using fixed emotional state: Valence={args.valence:.2f}, Energy={args.energy:.2f}")

    # ⭐ 평가 함수 호출 시 CLI에서 입력받은 감정 값 전달
    results = evaluate_recommendations(
        df_test, 
        recommend, 
        cf_model, meta_db, ranker, mapping_db,
        users, 
        candidate_tracks, 
        user_valence=args.valence, # ⭐ 전달
        user_energy=args.energy,   # ⭐ 전달
        topk=args.k
    )

    print("\n===== Evaluation Results =====")
    for metric, val in results.items():
        print(f"{metric}: {val:.4f}")