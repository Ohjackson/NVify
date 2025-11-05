import pandas as pd
import os

# 데이터 경로 설정

# data_path = "data/music_info.csv"

data_path = "data/user_listening_history.csv"
df = pd.read_csv(data_path)

print(df.shape)
print(df.head())

# 첫 번째 파일을 불러와 미리보기

print(df.shape)
print(df.head())
print(df.info())

print(df.isnull().sum().sort_values(ascending=False))
print(df.dtypes)

print(df.describe().T)

duplicates = df.duplicated().sum()
print("중복 행 수:", duplicates)

import matplotlib.pyplot as plt

num_cols = df.select_dtypes(include=["int", "float"]).columns

for col in num_cols:
    plt.figure()
    df[col].hist(bins=30)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

import seaborn as sns

corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    print(f"\n[{col}] 상위 10개 값:")
    print(df[col].value_counts().head(10))

summary_dir = os.path.dirname(data_path)          # data/
summary_path = os.path.join(summary_dir, "music_exploration_summary.csv")

df.describe(include='all').to_csv(summary_path, encoding='utf-8-sig')
print("요약 통계 저장 완료:", summary_path)
