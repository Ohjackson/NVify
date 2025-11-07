import pickle
import os

MAPPING_PATH = 'lastfm_to_spotify_mapping.pkl'

print(f"[INFO] 매핑 파일 로드 시작: {MAPPING_PATH}")

if not os.path.exists(MAPPING_PATH):
    print(f"🛑 [FATAL] 파일을 찾을 수 없습니다: {MAPPING_PATH}")
else:
    try:
        # 파일 로드
        with open(MAPPING_PATH, 'rb') as f:
            mapping_db = pickle.load(f)
        
        print("\n===== 매핑 데이터베이스 정보 =====")
        # 1. 딕셔너리 크기 확인 (총 매핑된 트랙 수)
        print(f"총 매핑된 트랙 수 (Key 개수): {len(mapping_db):,}")
        
        # 2. 키와 값의 유형 확인 (Last.fm ID vs Spotify ID)
        if mapping_db:
            # 딕셔너리의 첫 5개 항목 샘플링
            sample_items = list(mapping_db.items())[:5]
            
            print("\n샘플 5개 항목:")
            for lastfm_id, spotify_id in sample_items:
                print(f"  - Last.fm ID (Key): {lastfm_id}")
                print(f"    -> Spotify ID (Value): {spotify_id}")
                
            # 키와 값의 유형이 예상대로인지 확인
            first_key = next(iter(mapping_db))
            first_value = mapping_db[first_key]
            
            print("\n예상되는 구조 확인:")
            print(f"  - Key 유형 (Last.fm ID): {type(first_key)}")
            print(f"  - Value 유형 (Spotify ID): {type(first_value)}")
            
            # Spotify ID 포맷 확인
            if str(first_value).startswith("spotify:track:"):
                 print("  ✅ Value는 'spotify:track:' 포맷으로 보입니다.")
            else:
                 print("  ⚠️ Value 포맷이 예상과 다를 수 있습니다.")

        else:
            print("데이터베이스가 비어 있습니다.")
            
    except Exception as e:
        print(f"🛑 [ERROR] 파일 로드 중 오류 발생: {e}")