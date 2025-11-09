# ==========================================================
# 🎵 NVify - Spotify ID 병합 및 최종 데이터셋 구축
# ==========================================================
#
# 📝 설명:
#   이전에 생성된 Last.fm <-> Spotify ID 매핑 딕셔너리를 사용하여 
#   사용자 상호작용 데이터(`track_id` 컬럼)에 `spotify_id` 컬럼을 추가하고, 
#   매핑에 실패한 레코드를 제거하여 최종 하이브리드 추천 데이터셋을 완성합니다.
#
# ----------------------------------------------------------
# 📁 파일 정보
# ----------------------------------------------------------
#
# ➡️ 입력 파일 (Input):
#   - lastfm_hybrid_min3.csv: 전처리 및 필터링된 사용자 데이터
#   - lastfm_to_spotify_mapping.pkl: Last.fm ID와 Spotify ID 간의 매핑 딕셔너리
#
# ⬅️ 출력 파일 (Output):
#   - test_data_with_spotify_id.csv: Spotify ID가 추가되고 정제된 최종 데이터셋
#
# 🛠️ 주요 라이브러리:
#   - pandas, pickle, numpy
#
# ==========================================
# 1. 설정 및 초기화
# ==========================================

import pandas as pd
import pickle
import numpy as np

# --- 1. 파일 경로 설정 ---
LASTFM_TEST_FILE = 'lastfm_hybrid_min3.csv'
MAPPING_FILE = 'lastfm_to_spotify_mapping.pkl'
OUTPUT_FILE = 'test_data_with_spotify_id.csv'

# ==========================================
# 2. 메인 실행 로직
# ==========================================

try:
    # 2. 데이터 및 매핑 파일 로드
    df_test_raw = pd.read_csv(LASTFM_TEST_FILE)
    print(f"✅ Last.fm 테스트 데이터 로드 완료. (총 {len(df_test_raw)}행)")
    
    with open(MAPPING_FILE, 'rb') as f:
        lastfm_to_spotify_mapping = pickle.load(f)
    print("✅ 매핑 파일 로드 완료.")

    # 3. Last.fm track_id를 Spotify ID로 변환하여 새로운 컬럼 ('spotify_id') 추가
    print("\n--- 4. Spotify ID 매핑 시작 ---")
    
    # 딕셔너리 형태의 매핑은 .map() 함수를 사용하는 것이 가장 빠르고 효율적입니다.
    df_test_raw['spotify_id'] = df_test_raw['track_id'].map(lastfm_to_spotify_mapping)
    
    # 4. 유효하지 않은 행 정리
    initial_rows = len(df_test_raw)
    
    # 매핑되지 않은 (즉, spotify_id가 NaN인) 행을 제거합니다.
    df_mapped = df_test_raw.dropna(subset=['spotify_id'])
    
    removed_rows = initial_rows - len(df_mapped)
    
    print(f"   - 매핑 후 유효 데이터: {len(df_mapped)}행")
    print(f"   - 매핑 실패/누락으로 제거된 데이터: {removed_rows}행")

    # 5. 최종 CSV 파일로 저장
    df_mapped.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n==========================================")
    print(f"🎉 성공! 매핑된 최종 CSV 파일이 '{OUTPUT_FILE}'로 저장되었습니다.")
    print(f"==========================================")
    
except FileNotFoundError as e:
    print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인하세요. ({e})")
except Exception as e:
    print(f"❌ 처리 중 예상치 못한 오류 발생: {e}")