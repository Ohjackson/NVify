import pandas as pd
import pickle
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import time

# ==========================================
# 1. 설정 및 초기화
# ==========================================

# ⚠️⚠️ IMPORTANT: Spotify API 키를 여기에 입력하거나 환경 변수를 사용하세요.
CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "YOUR_CLIENT_ID_HERE") 
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "YOUR_CLIENT_SECRET_HERE")

# 매핑할 데이터 파일 경로 (이전에 생성한 파일)
INPUT_CSV_PATH = 'lastfm_hybrid_min3.csv' 
# 최종 매핑 결과를 저장할 파일
OUTPUT_MAPPING_PATH = 'lastfm_to_spotify_mapping.pkl'


# ==========================================
# 2. 매핑 함수 정의
# ==========================================

def map_lastfm_to_spotify(title: str, artist: str, sp: spotipy.Spotify) -> str or None:
    """
    제목과 가수 정보를 사용하여 Spotify API를 검색하고 ID를 반환합니다.
    """
    # 검색 쿼리 생성
    if artist and artist != '':
        query = f'track:{title} artist:{artist}'
    else:
        query = f'track:{title}'
        
    try:
        # 가장 일치하는 트랙 1개만 검색
        results = sp.search(q=query, limit=1, type='track')
        
        if results and results['tracks']['items']:
            # Spotify ID 반환
            return f"spotify:track:{results['tracks']['items'][0]['id']}"
        else:
            return None
            
    except spotipy.SpotifyException as e:
        # API 레이트 리밋(Rate Limit) 오류 처리
        if e.http_status == 429:
            print("[WARNING] Rate limit hit. Waiting for 10 seconds before retrying...")
            time.sleep(10)
            # 재시도 (재귀 호출)
            return map_lastfm_to_spotify(title, artist, sp)
        # 기타 API 오류
        print(f"[ERROR] Spotify API failed for '{query}': {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unknown error: {e}")
        return None


# ==========================================
# 3. 메인 실행 로직
# ==========================================

if __name__ == "__main__":
    
    if CLIENT_ID == "YOUR_CLIENT_ID_HERE" or CLIENT_SECRET == "YOUR_CLIENT_SECRET_HERE":
        print("🛑 [FATAL] Spotify CLIENT_ID 또는 CLIENT_SECRET을 설정해 주세요.")
        raise SystemExit

    # 1. Spotify 인증 설정
    client_credentials_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID, 
        client_secret=CLIENT_SECRET
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    print("[INFO] Spotify API Client initialized successfully.")

    # 2. 데이터 로드 및 유니크 트랙 추출
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"🛑 [FATAL] 입력 파일 '{INPUT_CSV_PATH}'을 찾을 수 없습니다.")
        raise SystemExit

    # ⭐⭐ 유니크 트랙 추출 (강제 정규화 코드 없음)
    df_unique_tracks = df[['track_id', 'original_title', 'artist']].drop_duplicates(subset=['track_id'])
    
    total_tracks = len(df_unique_tracks)
    print(f"[INFO] 총 매핑할 유니크 트랙 수: {total_tracks:,}개")
    if total_tracks > 5000: 
        print(f"[WARNING] 유니크 트랙 수가 예상치({4601})보다 훨씬 많습니다. 이 문제가 진행률 초과의 원인입니다.")


    # 3. 매핑 수행 및 딕셔너리 생성
    mapping_dict = {}
    mapped_count = 0
    
    # index 변수를 직접 사용하여 순회합니다. (이전 로그 재현을 위함)
    for index, row in enumerate(df_unique_tracks.itertuples(), start=1):
        lastfm_tid = row.track_id
        title = row.original_title
        artist = row.artist
        
        # 매핑 실행
        spotify_tid = map_lastfm_to_spotify(title, artist, sp)
        
        if spotify_tid:
            mapping_dict[lastfm_tid] = spotify_tid
            mapped_count += 1
            
        # 진행 상황 출력
        # 여기서 index가 total_tracks를 초과하는 현상이 발생할 수 있습니다.
        if index % 100 == 0 or index == total_tracks:
            print(f"   - 진행률: {index}/{total_tracks} | 매핑 성공: {mapped_count} | 성공률: {mapped_count / index * 100:.2f}%")
            
        # API 레이트 리밋 방지를 위한 딜레이 (선택 사항)
        # time.sleep(0.01) 

    # 4. 결과 저장
    with open(OUTPUT_MAPPING_PATH, 'wb') as f:
        pickle.dump(mapping_dict, f)
        
    print("\n==============================================")
    print(f"✅ 매핑 완료: {mapped_count} / {total_tracks} 트랙")
    print(f"✅ 매핑 딕셔너리 저장 완료 → {OUTPUT_MAPPING_PATH}")
    print("==============================================")