import pandas as pd

df = pd.read_csv("data/user_ratings_drop1.csv")

# 유저별 평가(청취)한 곡 수
user_counts = df['user_id'].value_counts()

print("전체 유저 수:", len(user_counts))
print("평균 평가 곡 수:", round(user_counts.mean(), 2))
print("중앙값:", user_counts.median())
print("최대 평가 곡 수:", user_counts.max())
print("평균 이상인 유저 비율:", (user_counts > user_counts.mean()).mean())

# 상위/하위 5개 예시
print("\n[상위 5명 유저별 곡 수]")
print(user_counts.head())

print("\n[하위 5명 유저별 곡 수]")
print(user_counts.tail())
