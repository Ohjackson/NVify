# ==========================================================
# 🎵 NVify - LambdaMART 모델 성능 평가 (LTR Test)
# ==========================================================
#
# 📝 설명:
#   훈련된 LambdaMART (순위 학습, LTR) 모델의 성능을 테스트 데이터 상에서
#   검증합니다. CF 예측(taste), 감성 일치도(emotion), 인기도 역수(novelty)
#   세 가지 피처를 구축하여 순위 예측 후, NDCG@K, Precision@K, Recall@K를 계산합니다.
#
# ----------------------------------------------------------
# 📁 파일 정보
# ----------------------------------------------------------
#
# ➡️ 입력 파일 (Input):
#   - test_data_with_spotify_id.csv: Spotify ID가 추가된 최종 하이브리드 데이터
#   - cf_model_final.pkl: 협업 필터링(CF) 모델
#   - ranking_model_final.pkl: LambdaMART 순위 학습 모델
#   - track_meta_db.pkl: 트랙 메타데이터 (Novelty 계산에 사용될 수 있음)
#
# ⬅️ 출력 파일 (Output):
#   - (None): 콘솔에 최종 성능 지표 출력
#
# 🛠️ 주요 라이브러리:
#   - pandas, numpy, pickle, lightgbm, sklearn.metrics (ndcg_score)
#
# ==========================================
# 1. 설정 및 자산 로드
# ==========================================

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.metrics import ndcg_score 

# --- 파일 경로 설정 ---
TEST_MAPPED_FILE_PATH = 'test_data_with_spotify_id.csv' 
USER_EMOTION_V_TEST = 0.5 
USER_EMOTION_A_TEST = 0.5 
K = 10 

try:
    final_cf_model = pickle.load(open('cf_model_final.pkl', 'rb'))
    track_meta_db = pickle.load(open('track_meta_db.pkl', 'rb')) 
    final_ranking_model = pickle.load(open('ranking_model_final.pkl', 'rb'))
    df_test = pd.read_csv(TEST_MAPPED_FILE_PATH)
    print("✅ 모델 및 DB 로드 완료.")
    print(f"✅ 테스트 데이터({TEST_MAPPED_FILE_PATH}) 로드 완료: {len(df_test)}개")
except FileNotFoundError as e:
    print(f"❌ 오류: 필수 파일이 누락되었습니다. ({e})")
    exit()

# ==========================================
# 2. 핵심 함수 정의 (Core Functions)
# ==========================================

# --- LTR 피쳐 엔지니어링 함수 정의 (Novelty Score 직접 계산) ---

def build_ltr_features(dataframe, df_name, cf_model, meta_db, user_emotion_v, user_emotion_a):
    print(f"\n--- {df_name} 피쳐 구축 시작 (Novelty Score 직접 계산) ---")
    
    # 0. 트랙 인기도 사전 계산
    print(f"   - 피쳐 3 계산을 위한 트랙 출현 빈도 계산 중...")
    track_popularity = dataframe['spotify_id'].value_counts().to_dict()

    # 1. taste_score (CF 예측)
    print(f"   - 피쳐 1: 'taste_score' 계산 중...")
    dataframe['taste_score'] = dataframe.apply(
        lambda row: cf_model.predict(row['user_id'], row['spotify_id']).est,
        axis=1
    )

    # 2. emotion_score (CSV V/E 사용)
    print(f"   - 피쳐 2: 'emotion_score' 계산 중...")
    distance = np.sqrt(
        (user_emotion_v - dataframe['valence'])**2 + 
        (user_emotion_a - dataframe['energy'])**2
    )
    # 거리의 역수 형태로 변환하여 점수화
    dataframe['emotion_score'] = 1 / (1 + distance) 

    # 3. novelty_score (직접 계산하여 사용)
    print(f"   - 피쳐 3: 'novelty_score' 계산 중...")
    
    def get_novelty_score_calculated(row, current_popularity):
        track_id = row['spotify_id']
        rating_count = current_popularity.get(track_id, 0) 
        # 참신성 = 1 / (인기도 + 1)
        return 1 / (rating_count + 1)
    
    dataframe['novelty_score'] = dataframe.apply(
        lambda row: get_novelty_score_calculated(row, track_popularity), 
        axis=1
    )
    
    # 4. Label 및 Query 정리
    print("   - 피쳐 4, 5: 'Label', 'Query' 정리 중...")
    # rating >= 0.5를 관련 항목(Label=1)으로 간주
    dataframe['label'] = (dataframe['rating'] >= 0.5).astype(int)
    # user_id를 쿼리 그룹 ID로 변환
    dataframe['query_id'] = dataframe['user_id'].astype('category').cat.codes

    # 최종 LTR 데이터셋 완성 (필수 컬럼이 NaN인 행 제거)
    ltr_dataframe = dataframe.dropna(subset=['taste_score', 'rating', 'valence', 'energy']).copy()

    # LTR 피쳐 구축 완료
    print(f"   - {df_name} LTR 피쳐 구축 완료: {len(ltr_dataframe)}개")
        
    return ltr_dataframe[['user_id', 'spotify_id', 'rating', 'query_id', 'taste_score', 'emotion_score', 'novelty_score', 'label']]

# --- 다중 지표 계산 함수 ---

def calculate_all_metrics_by_query(df, k):
    
    # 쿼리 ID 및 예측 점수 순으로 정렬
    df_sorted = df.sort_values(
        ['query_id', 'predicted_score'], 
        ascending=[True, False]
    )

    ndcgs, precisions, recalls = [], [], []

    for _, group in df_sorted.groupby('query_id'):
        y_true = group['label'].values 
        y_score = group['predicted_score'].values 
        
        if len(y_true) == 0:
            continue

        actual_k = min(k, len(y_true))
        
        # 1. NDCG@K
        try:
            ndcg = ndcg_score([y_true], [y_score], k=actual_k)
            ndcgs.append(ndcg)
        except ValueError:
            pass

        # 2. Precision@K
        top_k_items = y_true[:actual_k]
        hits_at_k = np.sum(top_k_items) 
        if actual_k > 0:
            precisions.append(hits_at_k / actual_k)
        
        # 3. Recall@K
        total_relevant = np.sum(y_true) 
        if total_relevant > 0:
            recalls.append(hits_at_k / total_relevant)
        
    mean_ndcg = np.mean(ndcgs) if ndcgs else 0
    mean_precision = np.mean(precisions) if precisions else 0
    mean_recall = np.mean(recalls) if recalls else 0
    
    return mean_ndcg, mean_precision, mean_recall


# ==========================================
# 3. 메인 실행 로직 (__main__)
# ==========================================

if __name__ == "__main__":
    
    # 1. 테스트 LTR 피쳐 구축
    ltr_test_df = build_ltr_features(
        df_test.copy(), 
        "Test_DF", 
        final_cf_model, 
        track_meta_db, 
        USER_EMOTION_V_TEST, 
        USER_EMOTION_A_TEST
    )

    # 2. 순위 예측
    feature_columns = ['taste_score', 'emotion_score', 'novelty_score']
    X_test = ltr_test_df[feature_columns]

    print("\n--- 4. 순위 예측 및 다중 지표 계산 ---")
    ltr_test_df['predicted_score'] = final_ranking_model.predict(X_test)

    # 3. 다중 지표 계산 실행
    mean_ndcg, mean_precision, mean_recall = calculate_all_metrics_by_query(ltr_test_df, k=K)

    # 4. 결과 출력
    print(f"\n==========================================")
    print(f"🏆 최종 LambdaMART 모델 성능 지표 (Test, K={K})")
    print(f"   - Novelty Score는 테스트 데이터 내 출현 빈도를 사용하여 계산됨.")
    print(f"   - 테스트 쿼리 수 (사용자 수): {ltr_test_df['query_id'].nunique()} 명")
    print(f"   - **Mean NDCG@{K}: {mean_ndcg:.4f}**")
    print(f"   - **Mean Precision@{K}: {mean_precision:.4f}**")
    print(f"   - **Mean Recall@{K}: {mean_recall:.4f}**")
    print(f"==========================================")