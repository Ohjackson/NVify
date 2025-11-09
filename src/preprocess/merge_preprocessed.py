import pandas as pd
import os

data_dir = "data"

user_path = os.path.join(data_dir, "user_ratings_min3.csv")      # 유저 필터링된 파일
music_path = os.path.join(data_dir, "music_emotion_clean.csv")   # 감정 피처 정제된 음악 데이터
output_path = os.path.join(data_dir, "hybrid_min3.csv")          # 최종 병합 결과

# 파일 불러오기
user_df = pd.read_csv(user_path)
music_df = pd.read_csv(music_path)

# 병합 (track_id 기준 inner join)
merged = user_df.merge(music_df, on="track_id", how="inner")

# 결과 확인
print("병합 전:", user_df.shape, music_df.shape)
print("병합 후:", merged.shape)
print("남은 유저 수:", merged["user_id"].nunique())
print("남은 트랙 수:", merged["track_id"].nunique())

# 저장
merged.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n 병합 완료 → {output_path}")
