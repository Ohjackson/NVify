import pandas as pd

# 기존 drop1 결과 불러오기
df = pd.read_csv("data/user_ratings_drop1.csv")

# 유저별 평가 곡 수 계산
user_counts = df['user_id'].value_counts()

# 3곡 이상 평가한 유저만 유지
valid_users = user_counts[user_counts >= 3].index
df_filtered = df[df['user_id'].isin(valid_users)]

# 통계 확인
print("필터 전:", df.shape)
print("필터 후:", df_filtered.shape)
print("남은 사용자 수:", df_filtered['user_id'].nunique())
print("남은 트랙 수:", df_filtered['track_id'].nunique())

# 저장
df_filtered.to_csv("data/user_ratings_min3.csv", index=False, encoding='utf-8-sig')
print("\n 저장 완료 → data/user_ratings_min3.csv")
