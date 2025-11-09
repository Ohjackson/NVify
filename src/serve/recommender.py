# ==========================================================
# 🎵 NVify - 하이브리드 음악 추천 시스템 실행 (Demo)
# ==========================================================
#
# 📝 설명:
#   사용자 ID와 입력 감성(Valence/Energy)을 기반으로, 훈련된 CF 모델과 LambdaMART 모델을
#   활용하여 3가지 피처(취향, 감성, 참신성)를 통합한 Top-N 맞춤형 음악을 추천하고,
#   추천 결과를 콘솔에 출력하는 최종 실행 파일입니다.
#
# ----------------------------------------------------------
# 📁 파일 정보
# ----------------------------------------------------------
#
# ➡️ 입력 파일 (Input):
#   - test_data_with_spotify_id.csv: 트랙 메타데이터 및 V/E 정보
#   - cf_model_final.pkl: 협업 필터링(CF) 모델
#   - final_ranking_model.pkl: LambdaMART 순위 학습 모델
#   - track_meta_db.pkl: 트랙 메타데이터 (참신성 점수 계산에 사용될 수 있음)
#
# ⬅️ 출력 파일 (Output):
#   - (None): 콘솔에 Top-N 추천 목록 출력
#
# 🛠️ 주요 라이브러리:
#   - pandas, numpy, pickle, surprise, lightgbm
#
# ==========================================
# 1. 사용자 입력 및 설정
# ==========================================

import pandas as pd
import numpy as np
import pickle
from surprise import SVD 
import lightgbm as lgb

# A. 추천을 받고 싶은 유저 ID를 설정하세요. (예: '-MJ-')
TARGET_USER_ID = input("추천을 받을 유저 ID를 입력하세요 (예: -MJ-): ") or '-MJ-'

print("\n--- 👤 사용자 감성 입력 ---")
try:
    # Valence (행복/슬픔: 0.0 ~ 1.0) 입력 받기
    user_emotion_v = float(input("Valence (행복/슬픔, 0.0 ~ 1.0): "))
    if not (0.0 <= user_emotion_v <= 1.0):
        raise ValueError
    
    # Energy (활력/차분: 0.0 ~ 1.0) 입력 받기
    user_emotion_a = float(input("Energy (활력/차분, 0.0 ~ 1.0): "))
    if not (0.0 <= user_emotion_a <= 1.0):
        raise ValueError
        
    print(f"\n✅ 사용자 설정 감성: Valence={user_emotion_v:.2f}, Energy={user_emotion_a:.2f}")
except ValueError:
    print("❌ 오류: 감성 값은 0.0~1.0 사이의 실수로 입력해야 합니다. 기본값(0.5, 0.5)을 사용합니다.")
    user_emotion_v = 0.5
    user_emotion_a = 0.5

N_RECOMMENDATIONS = 10 # 받고 싶은 추천 트랙 개수

# ==========================================
# 2. 모델 및 데이터 로드
# ==========================================
TEST_MAPPED_FILE_PATH = 'test_data_with_spotify_id.csv' 
try:
    final_cf_model = pickle.load(open('cf_model_final.pkl', 'rb'))
    track_meta_db = pickle.load(open('track_meta_db.pkl', 'rb')) 
    final_ranking_model = pickle.load(open('ranking_model_final.pkl', 'rb'))
    # 모든 트랙 정보(V/E, ID 등)를 포함하는 마스터 트랙 리스트로 활용
    df_all_tracks = pd.read_csv(TEST_MAPPED_FILE_PATH) 
    print("\n✅ 모델 및 데이터 로드 완료.")
except FileNotFoundError as e:
    print(f"\n❌ 오류: 필수 파일이 누락되었습니다. 학습 단계가 완료되었는지 확인하세요. ({e})")
    exit()

# ==========================================
# 3. 핵심 함수 정의 (Core Functions)
# ==========================================

def get_recommendations(user_id, df_master, cf_model, meta_db, user_emov, user_emoa, N):
    
    # 3.1. 후보 트랙 목록 준비
    candidate_tracks_df = df_master[['spotify_id', 'original_title', 'artist', 'valence', 'energy']].drop_duplicates(subset=['spotify_id']).copy()
    candidate_tracks_df['user_id'] = user_id
    
    # 3.2. 피처 계산을 위한 준비
    print(f"\n--- 💡 {user_id} 님을 위한 {len(candidate_tracks_df)}개 후보 트랙 피처 구축 시작 ---")
    
    # 트랙 인기도 사전 계산 (현재 후보 목록 내 출현 빈도 사용)
    track_popularity = df_master['spotify_id'].value_counts().to_dict()

    # 1. taste_score (CF 예측)
    print("   - 피쳐 1: 'taste_score' 계산 중...")
    candidate_tracks_df['taste_score'] = candidate_tracks_df.apply(
        lambda row: cf_model.predict(row['user_id'], row['spotify_id']).est,
        axis=1
    )

    # 2. emotion_score (사용자 입력 감성 기반)
    print("   - 피쳐 2: 'emotion_score' 계산 중...")
    distance = np.sqrt(
        (user_emov - candidate_tracks_df['valence'])**2 + 
        (user_emoa - candidate_tracks_df['energy'])**2
    )
    # 거리를 점수로 변환 (거리가 가까울수록 점수 높음)
    candidate_tracks_df['emotion_score'] = 1 / (1 + distance) 

    # 3. novelty_score (현재 데이터셋 내 인기도 기반)
    print("   - 피쳐 3: 'novelty_score' 계산 중...")
    
    def get_novelty_score_calculated(row, current_popularity):
        rating_count = current_popularity.get(row['spotify_id'], 0) 
        # 참신성 = 1 / (인기도 + 1)
        return 1 / (rating_count + 1)
    
    candidate_tracks_df['novelty_score'] = candidate_tracks_df.apply(
        lambda row: get_novelty_score_calculated(row, track_popularity), 
        axis=1
    )
    
    # 3.3. 최종 예측
    feature_columns = ['taste_score', 'emotion_score', 'novelty_score']
    X_candidates = candidate_tracks_df[feature_columns]
    
    print("   - 4. LambdaMART 최종 순위 점수 예측 중...")
    # LTR 모델을 사용하여 최종 점수 예측
    candidate_tracks_df['predicted_score'] = final_ranking_model.predict(X_candidates)
    
    # 3.4. 점수 기준 정렬 및 상위 N개 추출
    recommendations = candidate_tracks_df.sort_values(
        'predicted_score', 
        ascending=False
    ).head(N)
    
    return recommendations[['original_title', 'artist', 'predicted_score', 'taste_score', 'emotion_score', 'novelty_score', 'spotify_id']]

# ==========================================
# 4. 추천 실행 및 결과 출력
# ==========================================

if __name__ == "__main__":
    
    top_N_recommendations = get_recommendations(
        TARGET_USER_ID, 
        df_all_tracks, 
        final_cf_model, 
        track_meta_db, 
        user_emotion_v, 
        user_emotion_a, 
        N_RECOMMENDATIONS
    )

    print(f"\n==========================================")
    print(f"🥇 {TARGET_USER_ID} 님을 위한 맞춤 음악 추천 (N={N_RECOMMENDATIONS})")
    print(f"   (사용자 감성: V={user_emotion_v:.2f}, E={user_emotion_a:.2f})")
    print(f"==========================================")
    print(top_N_recommendations.to_string(index=False))
    print(f"==========================================")