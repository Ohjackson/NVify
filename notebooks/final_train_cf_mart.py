# Converted from final_train_cf_mart.ipynb
# Cell magics and shell commands are commented out.
# -*- coding: utf-8 -*-


# %% [markdown] cell 1
# <a href="https://colab.research.google.com/github/Ohjackson/NVify/blob/part%2Fcf_lambdamart_train/final_train_cf_mart.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% code cell 2
# !pip install surprise

# %% code cell 3
import pandas as pd
import numpy as np
import pickle
from surprise import Dataset, Reader, SVD

# 전처리된 마스터 데이터셋 로드
file_path = 'hybrid_drop1.csv'
try:
    df = pd.read_csv(file_path)
    print(f"'{file_path}' 로드 완료. (행: {len(df)})")
except FileNotFoundError:
    print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 이전 전처리 셀을 실행하세요.")
    raise

print("\n CF 모델 학습 시작 (Spotify ID 기반)...")
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(df[['user_id', 'spotify_id', 'rating']], reader)

trainset = data.build_full_trainset()
model_cf = SVD(n_factors=100, n_epochs=20, random_state=42, verbose=False)
model_cf.fit(trainset)

with open('cf_model_final.pkl', 'wb') as f:
    pickle.dump(model_cf, f)
print("--- 'cf_model_final.pkl' 저장 완료 ---")


print("\n트랙 메타 DB 구축 시작 (Spotify ID 기반)...")

# 트랙별 인기도(평가 횟수) 계산
rating_counts = df['spotify_id'].value_counts().reset_index()
rating_counts.columns = ['spotify_id', 'total_rating_count']

# 트랙별 V/A 정보 추출 (중복 제거)
track_info = df[['spotify_id', 'valence', 'energy']].drop_duplicates(subset=['spotify_id'])

# 인기도와 V/A를 병합
meta_df = pd.merge(track_info, rating_counts, on='spotify_id', how='left')
meta_df['total_rating_count'] = meta_df['total_rating_count'].fillna(0).astype(int)

# 딕셔너리 형태로 변환 (키가 Spotify ID)
track_meta_db = meta_df.set_index('spotify_id').to_dict('index')

with open('track_meta_db.pkl', 'wb') as f:
    pickle.dump(track_meta_db, f)

print(f"--- 'track_meta_db.pkl' 저장 완료 (총 {len(track_meta_db)}개 트랙) ---")
print("\n--- 2개 자산 생성 모두 완료 ---")

# %% code cell 4
"""
[최종 랭킹 모델] LambdaMART 학습 스크립트

본 스크립트는 감정 기반 음악 추천 시스템의 최종 랭킹을 담당하는
LambdaMART 모델('ranking_model_final.pkl')을 학습하고 저장합니다.

실행 전 필요 파일:
    - cf_model_final.pkl: 학습 완료된 Surprise SVD (CF) 모델
    - track_meta_db.pkl: 'total_rating_count'가 포함된 트랙 메타 딕셔너리
    - hybrid_drop1.csv: LTR 모델 학습을 위한 원본 데이터셋

주요 로직:
    1. 자산 로드: CF 모델과 메타 DB를 로드합니다.
    2. 데이터 로드: 'hybrid_drop1.csv'를 로드합니다.
    3. 피처 엔지니어링: 3가지 핵심 피처(taste, emotion, novelty)를 생성합니다.
    4. 모델 학습: LGBMRanker(LambdaMART)로 최종 모델을 학습합니다.
    5. 모델 저장: 'ranking_model_final.pkl'로 저장합니다.
"""
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split as sklearn_split
from surprise import SVD

# taste_score 예측을 위한 cf모델, novelty_score 계산을 위한 metadata를 가져
final_cf_model = pickle.load(open('cf_model_final.pkl', 'rb'))
track_meta_db = pickle.load(open('track_meta_db.pkl', 'rb'))
print("   - cf_model_final.pkl, track_meta_db.pkl 로드 완료")

# 학습 데이터셋 로드
print("학습 데이터셋 로딩 중 ---")
try:
    df_train = pd.read_csv('hybrid_drop1.csv')
    print(f"   - 학습 데이터(hybrid_drop1) 로드 완료: {len(df_train)}")
except FileNotFoundError:
    print("오류: 'hybrid_drop1.csv' 파일을 찾을 수 없습니다.")
    raise

print("\n-LTR 피쳐 엔지니어링 시작 ")
USER_EMOTION_V = 0.5 # 학습용 가상 감정
USER_EMOTION_A = 0.5


def build_ltr_features(dataframe, df_name, cf_model, meta_db, user_emotion_v, user_emotion_a):
  """
    LTR 학습을 위한 3가지 핵심 피처(taste, emotion, novelty)를 생성합니다.

    Args:
        dataframe (pd.DataFrame): 'user_id', 'spotify_id', 'valence', 'energy' 등이 포함된 원본 데이터.
        df_name (str): 로그 출력을 위한 데이터 프레임 이름 (예: "Train_DF").
        cf_model (surprise.SVD): .predict(uid, iid)가 가능한 학습된 CF 모델.
        meta_db (dict): 트랙 메타 정보가 담긴 딕셔너리 (key: spotify_id).
        user_emotion_v (float): 사용자의 (가상) Valence 값.
        user_emotion_a (float): 사용자의 (가상) Energy 값.

    Returns:
        pd.DataFrame: 'taste_score', 'emotion_score', 'novelty_score', 'label', 'query_id'가
                      추가되고 결측치가 제거된 LTR용 데이터 프레임.
    """
    print(f"--- {df_name} 피쳐 구축 시작 ---")
    print("   - 피쳐 1: 'taste_score' 계산 중...")

    # 로드해둔 CF모델을 사용하여 user_id, spotify_id을 바탕으로 예상 rating 계산
    dataframe['taste_score'] = dataframe.apply(
        lambda row: cf_model.predict(row['user_id'], row['spotify_id']).est,
        axis=1
    )

    print("   - 피쳐 2, 3: 'emotion_score', 'novelty_score' 계산 중...")

    def calculate_cb_scores(row, meta_db_ref, user_emov, user_emoa):
        track_id = row['spotify_id']
        meta = meta_db_ref.get(track_id)

        track_valence = row['valence']
        track_energy = row['energy']

        # 감정 거리 계산(유클리드 거리)
        distance = np.sqrt((user_emov - track_valence)**2 + (user_emoa - track_energy)**2)
        # 유클리드 거리의 역수로 유사도가 높을수록 점수가 높음
        emotion_score = 1 / (1 + distance)

        novelty_score = np.nan
        # total_rating_count는 특정 곡에 대해 평가가 매겨진 총 횟수(인기도)
        if meta and 'total_rating_count' in meta:
            # total_rating_count의 역수로 클수록 인기도가 낮은(신규성이 확보된) 점수
            novelty_score = 1 / (meta['total_rating_count'] + 1)

        return emotion_score, novelty_score, (row['rating'] >= 0.5)

    # feature_df로 emotion_score, novelty_score, label을 받아옴
    features_df = dataframe.apply(
        lambda row: calculate_cb_scores(row, meta_db, user_emotion_v, user_emotion_a),
        axis=1, result_type='expand'
    )

    # 원본 데이터 프레임에 emotion_score, novelty_score, label, query_id 컬럼을 추가
    dataframe['emotion_score'] = features_df[0]
    dataframe['novelty_score'] = features_df[1]

    print("   - 피쳐 4, 5: 'Label', 'Query' 정리 중...")
    dataframe['label'] = features_df[2].astype(int)
    # user_id 기준으로 그룹핑을 하고 문자열 id를 카테고리화 시키고 정수값으로 변경
    dataframe['query_id'] = dataframe['user_id'].astype('category').cat.codes

    # lambdaMART에 사용될 최종 dataset완성(taste_score, emotion_score, novelty_score, label가 컬럼으로 들어감)
    ltr_dataframe = dataframe.dropna(subset=['taste_score', 'emotion_score', 'novelty_score', 'label'])
    print(f"{df_name} LTR 피쳐 구축 완료: {len(ltr_dataframe)}개")
    return ltr_dataframe

# 학습 데이터에 대해서만 피쳐 구축 실행
ltr_train_df = build_ltr_features(df_train.copy(), "Train_DF", final_cf_model, track_meta_db, USER_EMOTION_V, USER_EMOTION_A)
del df_train # 메모리 확보

print("\n[자산 3] 최종 모델 (API 서빙용) 학습 시작 (평가 생략) ---")

feature_columns = ['taste_score', 'emotion_score', 'novelty_score']

# label 분
X_full = ltr_train_df[feature_columns]
y_full = ltr_train_df['label']
# 랭킹 모델에 필요한 그룹 정보 생성. ex) 첫 20개 행이 한 그룹, 그다음 15개 행이 다음 그룹이라고 인식
q_full = ltr_train_df.groupby('query_id').size().values

# 평가를 안 했으므로, 'n_estimators' 값을 500 등으로 고정
FIXED_N_ESTIMATORS = 500

final_ranking_model = lgb.LGBMRanker(
    objective='lambdarank',
    device='cpu',
    random_state=42,
    n_estimators=FIXED_N_ESTIMATORS, # 고정된 값 사용
    learning_rate=0.05,
    verbose = -1
)

# 전체 학습 데이터로 'fit'만 실행
print(f"   - {FIXED_N_ESTIMATORS}개 트리로 최종 모델 학습 중...")
final_ranking_model.fit(
    X=X_full,
    y=y_full,
    group=q_full
)

# [자산 3] 저장
with open('ranking_model_final.pkl', 'wb') as f:
    pickle.dump(final_ranking_model, f)
print("--- 'ranking_model_final.pkl' 저장 완료 ---")
