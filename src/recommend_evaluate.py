# ==========================================================
# 🎵 NVify - 하이브리드 추천 시스템 최종 성능 평가
# ==========================================================
#
# 📝 설명:
#   최종 하이브리드 추천 파이프라인(CF + LTR)의 Top-K 성능을 검증합니다.
#   사용자 감성(Valence, Energy)을 인자로 받아 추천 함수에 전달하며, 
#   Precision@K, Recall@K, nDCG@K 세 가지 핵심 지표를 계산합니다.
#
# ----------------------------------------------------------
# 📁 파일 정보
# ----------------------------------------------------------
#
# ➡️ 입력 파일 (Input):
#   - test_dataset.csv (CLI 인자): user_id, track_id, rating이 포함된 테스트 데이터
#   - cf_model_final.pkl: 협업 필터링 모델
#   - ranking_model_final.pkl: LambdaMART 순위 학습 모델
#   - [recommender.py]: 'recommend' 및 'load_assets' 함수가 정의된 파일
#
# ⬅️ 출력 파일 (Output):
#   - (None): 콘솔에 최종 성능 지표 출력
#
# 🛠️ 주요 라이브러리:
#   - numpy, pandas, argparse, pickle
#
# ==========================================
# 1. 설정 및 초기화
# ==========================================

import numpy as np
import pandas as pd
import argparse
import pickle
import random
from typing import List

# ==========================================
# 2. 핵심 함수 정의 (Core Functions)
# ==========================================

# -----------------------------
# (1) Metric Functions
# -----------------------------
def precision_at_k(y_true, y_pred, k=10):
    """추천 목록 내 정답 항목의 비율"""
    y_pred_k = y_pred[:k]
    return np.mean([1 if i in y_true else 0 for i in y_pred_k])

def recall_at_k(y_true, y_pred, k=10):
    """전체 정답 항목 중 추천 목록에 포함된 비율"""
    y_pred_k = y_pred[:k]
    return len(set(y_true) & set(y_pred_k)) / len(y_true) if len(y_true) > 0 else 0.0

def ndcg_at_k(y_true, y_pred, k=10):
    """순위의 중요도를 반영한 지표"""
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
def evaluate_recommendations(test_df: pd.DataFrame, 
                             recommend_func, 
                             cf_model, meta_db, ranker, mapping_db,
                             users: List[str], 
                             candidate_tracks: List[str], 
                             user_valence: float,       # 사용자가 입력한 감정 값
                             user_energy: float,        # 사용자가 입력한 감정 값
                             topk=10):
    precisions, recalls, ndcgs = [], [], []
        
    for uid in users:
        gt_tracks = test_df[test_df["user_id"] == uid]
        # '관심 있음' (rating >= 0.5) 트랙을 Positive Label로 사용
        y_true = gt_tracks.loc[gt_tracks["rating"] >= 0.5, "track_id"].tolist()
        
        if not y_true:
            continue

        # recommend_func 호출 (감정 값 전달)
        recommended = recommend_func(
            user_id=uid, 
            candidate_tracks=candidate_tracks,
            cf_model=cf_model, meta_db=meta_db, ranker=ranker, mapping_db=mapping_db, 
            user_valence=user_valence, 
            user_energy=user_energy,
            topK=topk
        )["track_id"].tolist()

        # 지표 계산 및 누적
        precisions.append(precision_at_k(y_true, recommended, k=topk))
        recalls.append(recall_at_k(y_true, recommended, k=topk))
        ndcgs.append(ndcg_at_k(y_true, recommended, k=topk))

    results = {
        "Precision@K": np.mean(precisions),
        "Recall@K": np.mean(recalls),
        "nDCG@K": np.mean(ndcgs)
    }
    return results


# ==========================================
# 3. 메인 실행 로직 (__main__)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NVify Recommendation Performance")
    parser.add_argument("--test", type=str, default="test_dataset.csv",
                        help="CSV file containing user_id, track_id, rating columns")
    parser.add_argument("--k", type=int, default=10, help="Cutoff K for metrics")
    parser.add_argument("--limit", type=int, default=1000, help="Number of candidate tracks to evaluate")
    parser.add_argument("--valence", type=float, default=0.5, 
                        help="User emotional valence (0~1) for all evaluated users") 
    parser.add_argument("--energy", type=float, default=0.5, 
                        help="User emotional energy (0~1) for all evaluated users") 
    args = parser.parse_args()

    # Load required modules and assets
    # 'recommender' 모듈은 반드시 현재 경로에 존재해야 합니다.
    from recommender import recommend, load_assets 
    import random
    
    # 모델 및 DB 로드
    cf_model, meta_db, ranker, mapping_db = load_assets()

    candidate_tracks = list(mapping_db.keys())

    # 후보 트랙 수 제한
    limit = args.limit if args.limit > 0 else len(candidate_tracks)
    if len(candidate_tracks) > limit:
        candidate_tracks = random.sample(candidate_tracks, limit)

    # Load your test dataset
    df_test = pd.read_csv(args.test)
    
    # 유니크 유저 샘플링 (평가 속도 위해)
    users = random.sample(sorted(df_test["user_id"].unique().tolist()), 30)

    print(f"[INFO] Evaluating on {len(users)} sampled users, K={args.k}, Candidates={len(candidate_tracks)}")
    print(f"[INFO] Using fixed emotional state: Valence={args.valence:.2f}, Energy={args.energy:.2f}")

    # 평가 실행 (감정 값 전달)
    results = evaluate_recommendations(
        df_test, 
        recommend, 
        cf_model, meta_db, ranker, mapping_db,
        users, 
        candidate_tracks, 
        user_valence=args.valence, 
        user_energy=args.energy,   
        topk=args.k
    )

    print("\n===== Evaluation Results =====")
    for metric, val in results.items():
        print(f"{metric}: {val:.4f}")