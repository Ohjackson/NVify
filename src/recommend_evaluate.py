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
# (2) Evaluation Function
# -----------------------------
def evaluate_recommendations(test_df, recommend_func, users, topk=10):
    precisions, recalls, ndcgs = [], [], []
    for uid in users:
        gt_tracks = test_df[test_df["user_id"] == uid]
        y_true = gt_tracks.loc[gt_tracks["rating"] >= 0.5, "track_id"].tolist()
        if not y_true:
            continue

        recommended = recommend_func(uid, topK=topk)["track_id"].tolist()

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
# (3) Example Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NVify Recommendation Performance")
    parser.add_argument("--test", type=str, default="test_dataset.csv",
                        help="CSV file containing user_id, track_id, rating columns")
    parser.add_argument("--k", type=int, default=10, help="Cutoff K for metrics")
    args = parser.parse_args()

    # Load your test dataset
    df_test = pd.read_csv(args.test)

    # Import your recommender
    from recommender import recommend
    import random

    users = random.sample(sorted(df_test["user_id"].unique().tolist()), 30)

    print(f"[INFO] Evaluating on {len(users)} sampled users, K={args.k}")
    results = evaluate_recommendations(df_test, recommend, users, topk=args.k)

    print("\n===== Evaluation Results =====")
    for metric, val in results.items():
        print(f"{metric}: {val:.4f}")
