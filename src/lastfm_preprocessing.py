import json
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import re

# ------------------------------------------------------------
# 특수문자 및 이모지 제거 함수
# ------------------------------------------------------------
def clean_track_id(track_id):
    # 이모지 및 특수문자를 제거하고, 공백만 남도록 처리
    track_id = re.sub(r'[^\x00-\x7F]+', '', track_id)  # 이모지 제거
    track_id = re.sub(r'[^\w\s]', '', track_id)        # 특수문자 제거
    return track_id.strip().lower()

# ------------------------------------------------------------
# 노래 제목 및 가수 분리 함수 (⭐⭐ 수정됨: 파이프 '╎' 구분자 추가)
# ------------------------------------------------------------
def extract_title_artist(raw_track_string):
    # 1. 파이프 '╎' 기준으로 분리 시도 (Last.fm 포맷에서 흔함)
    if '╎' in raw_track_string:
        parts = raw_track_string.split('╎', 1) 
        title = parts[0].strip()
        artist = parts[1].strip()
        return title, artist

    # 2. 대시(-) 기준으로 분리 시도: '제목 - 가수' 형태 처리
    if ' - ' in raw_track_string:
        parts = raw_track_string.split(' - ', 1)
        title = parts[0].strip()
        artist = parts[1].strip()
        return title, artist
        
    # 3. 쉼표(,) 기준으로 분리 시도: '제목, 가수' 형태 처리
    if ',' in raw_track_string:
        parts = raw_track_string.split(',', 1) 
        # 단, 쉼표 분리는 제목에 쉼표가 있을 경우 부정확할 수 있음.
        # 일반적인 포맷이 아니라고 가정하고 분리하지 않고 전체를 제목으로 남겨둘 수도 있지만, 
        # 여기서는 일단 분리 로직 유지
        title = parts[0].strip()
        artist = parts[1].strip()
        return title, artist

    # 분리가 어려울 경우: 제목 = 원본, 가수 = 빈 값
    return raw_track_string.strip(), ""


# ------------------------------------------------------------
# 1️⃣ MUSIC INFO 전처리
# ------------------------------------------------------------
def preprocess_music_info(vads_path, out_path):
    print("[1] Music Info 전처리 시작")

    with open(vads_path, "r", encoding="utf-8") as f:
        item_vads = json.load(f)

    tracks = item_vads.get("Tracks", {})
    records = []
    for track, vads in tracks.items():
        if isinstance(vads, list) and len(vads) == 4:
            valence, arousal, dominance, sentiment = vads
            records.append({
                "track_id": track,
                "valence": valence,
                "energy": arousal,
                "dominance": dominance,
                "sentiment": sentiment
            })

    df_music = pd.DataFrame(records)
    print(f"   - 원본 트랙 수: {len(df_music):,}")

    # track_id 정리
    df_music['track_id'] = df_music['track_id'].apply(clean_track_id)

    # 정규화 후 중복된 track_id 제거
    original_count = len(df_music)
    # 중복된 track_id 중 첫 번째 레코드만 남깁니다.
    df_music.drop_duplicates(subset=['track_id'], keep='first', inplace=True) 
    
    print(f"   - 정규화 후 중복 제거된 트랙 수: {len(df_music):,} (제거: {original_count - len(df_music):,})")
    # `track_id`가 없는 데이터 제거
    df_music = df_music.dropna(subset=['track_id'])
    
    # ⭐ 최적화: track_id를 category 타입으로 변환하여 메모리 절약
    df_music['track_id'] = df_music['track_id'].astype('category')

    # 결측치 제거
    df_music = df_music.dropna(subset=["valence", "energy"])

    # 0~1 정규화
    scaler = MinMaxScaler()
    df_music[["valence", "energy"]] = scaler.fit_transform(df_music[["valence", "energy"]])

    # 감정 벡터 추가
    df_music["emotion_vector"] = df_music[["valence", "energy"]].values.tolist()

    df_music.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[1✅] Music info 저장 완료 → {out_path}\n")
    return df_music


# ------------------------------------------------------------
# 2️⃣ USER HISTORY 전처리 (rank → rating) (수정됨)
# ------------------------------------------------------------
def preprocess_user_history(top_path):
    print("[2] User Listening History 전처리 시작")

    with open(top_path, "r", encoding="utf-8") as f:
        top_tracks = json.load(f)

    user_records = []
    for user, tracks in top_tracks.items():
        n = len(tracks)
        if n <= 2:
            continue
        for rank, track_raw in enumerate(tracks, start=1):
            rating = 1 - (rank - 1) / (n - 1) if n > 1 else 1.0
            
            # ⭐ 새로운 컬럼: original_title, artist 추출
            title, artist = extract_title_artist(track_raw)
            
            user_records.append({
                "user_id": user,
                "track_id": track_raw, # 일단 원본을 저장
                "original_title": title, # Spotify 매핑용 제목
                "artist": artist,        # Spotify 매핑용 가수
                "rating": rating
            })

    df_user = pd.DataFrame(user_records)
    print(f"   - 유저 수: {df_user['user_id'].nunique():,}")
    print(f"   - 총 상호작용 수: {len(df_user):,}")

    # 1. track_id 정리 (병합 키로 사용될 클리닝된 ID 생성)
    # 기존 track_id 컬럼을 클리닝하여 병합 키로 사용합니다.
    df_user['track_id'] = df_user['track_id'].apply(clean_track_id)
    
    # 2. 빈 문자열 ("")을 NaN으로 변환 및 제거 (기존 수정 반영)
    df_user['track_id'] = df_user['track_id'].replace('', np.nan) 
    
    # `track_id`가 없는 데이터 제거
    df_user = df_user.dropna(subset=['track_id'])
    
    # 3. 최적화: category 타입 변환
    df_user['user_id'] = df_user['user_id'].astype('category')
    df_user['track_id'] = df_user['track_id'].astype('category')
    df_user['original_title'] = df_user['original_title'].astype('category') # 새 컬럼 추가
    df_user['artist'] = df_user['artist'].astype('category') # 새 컬럼 추가

    return df_user


# ------------------------------------------------------------
# 3️⃣ 최소 상호작용 필터 (유저≥3, 아이템≥5)
# ------------------------------------------------------------
def filter_min_interactions(df, min_user=3, min_item=5):
    print("[3] 최소 상호작용 필터 적용")

    initial_users = df['user_id'].nunique()
    initial_items = df['track_id'].nunique()
    initial_rows = len(df)
    
    # 필터링 반복: 필터링 후에도 기준 미달인 유저/아이템이 생길 수 있으므로 안정화될 때까지 반복
    while True:
        prev_rows = len(df)
        
        # 유저 필터
        ucnt = df["user_id"].value_counts()
        keep_users = ucnt[ucnt >= min_user].index
        df = df[df["user_id"].isin(keep_users)]

        # 아이템 필터
        icnt = df["track_id"].value_counts()
        keep_items = icnt[icnt >= min_item].index
        df = df[df["track_id"].isin(keep_items)]
        
        if len(df) == prev_rows:
            break # 더 이상 변화가 없으면 종료
            
    print(f"   - 초기 데이터: users={initial_users:,}, items={initial_items:,}, rows={initial_rows:,}")
    print(f"   - 최종 데이터: users={df['user_id'].nunique():,}, items={df['track_id'].nunique():,}, rows={len(df):,}")

    # 필터링 후 category 타입을 다시 설정하여 미사용 메모리 해제
    df['user_id'] = df['user_id'].cat.remove_unused_categories()
    df['track_id'] = df['track_id'].cat.remove_unused_categories()

    return df


# ------------------------------------------------------------
# 4️⃣ HYBRID 병합 (track_id 기준)
# ------------------------------------------------------------
def merge_datasets(user_df, music_df, out_path):
    print("[4] Hybrid 데이터 병합 시작")

    # 병합
    # track_id가 category 타입이므로, 효율적인 병합이 기대됨
    merged = user_df.merge(music_df, on="track_id", how="inner")
    print(f"   - 병합 후 shape: {merged.shape}")
    print(f"   - 유저 수: {merged['user_id'].nunique():,}")
    print(f"   - 트랙 수: {merged['track_id'].nunique():,}")
    
    # 병합된 데이터 저장
    # to_csv도 메모리 효율화 덕분에 빨라짐
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[4✅] Hybrid 데이터 저장 완료 → {out_path}\n")

    # 병합된 데이터를 반환
    return merged


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    # 실행 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    vads_path = os.path.join(BASE_DIR, "item_vads.json")
    top_path = os.path.join(BASE_DIR, "top_tracks.json")

    # 출력 파일 경로 설정
    out_music = os.path.join(BASE_DIR, "lastfm_music_emotion_clean.csv")
    out_user_min3 = os.path.join(BASE_DIR, "lastfm_user_ratings_min3.csv")
    out_hybrid_min3 = os.path.join(BASE_DIR, "lastfm_hybrid_min3.csv")

    print("============================================================")
    print(" LAST.FM JSON → CLEAN CSV PIPELINE START (Optimized)")
    print("============================================================")

    # 1. Music Info 전처리 및 저장 (track_id: category 타입 적용)
    df_music = preprocess_music_info(vads_path, out_music)
    
    # 2. User History 전처리 (user_id, track_id: category 타입 적용)
    df_user = preprocess_user_history(top_path)
    
    # 3. 최소 상호작용 필터링
    df_user_min = filter_min_interactions(df_user, min_user=3, min_item=5)
    df_user_min.to_csv(out_user_min3, index=False, encoding="utf-8-sig")

    # 4. Hybrid 데이터 병합 및 저장 (최적화된 타입 덕분에 병합 속도 개선)
    merged_data = merge_datasets(df_user_min, df_music, out_hybrid_min3)

    print("============================================================")
    print("✅ 모든 전처리 단계 완료 (Optimized)")
    print("============================================================")