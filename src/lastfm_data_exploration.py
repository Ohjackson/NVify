"""
============================================================
LAST.FM RAW JSON DATA EXPLORATION
============================================================

탐색 대상:
 - item_vads.json (감정 피처)
 - top_tracks.json (사용자-트랙 상호작용)
 - item_tags.json (태그 구조 요약)

기능:
 - JSON 구조 깊이 탐색
 - 항목 수, 키 구조, 예시 샘플 확인
 - 감정값 통계 (V-A-D-S)
 - 사용자 상호작용 통계
 - 태그 데이터 샘플 확인
============================================================
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------
# 1️⃣ item_vads.json — 감정 피처
# ------------------------------------------------------------
vads_path = os.path.join(DATA_DIR, "item_vads.json")
print(f"[INFO] Loading {vads_path}...")
with open(vads_path, "r", encoding="utf-8") as f:
    item_vads = json.load(f)

print(f"  ▶ Keys: {list(item_vads.keys())}")
for k in item_vads.keys():
    print(f"  - {k}: {len(item_vads[k])} entries")

# 트랙만 분석
track_vads = item_vads.get("Tracks", {})
records = []
for track_name, vads in track_vads.items():
    if isinstance(vads, list) and len(vads) == 4:
        valence, arousal, dominance, sentiment = vads
        records.append({
            "track": track_name,
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "sentiment": sentiment
        })

vads_df = pd.DataFrame(records)
print(f"[INFO] Valid track VADS entries: {len(vads_df):,}")
print(vads_df.head(5))

# 기본 통계
print("\n[INFO] VADS Descriptive Statistics:")
print(vads_df.describe().T)

# 시각화
for col in ["valence", "arousal", "dominance", "sentiment"]:
    plt.figure(figsize=(6, 4))
    sns.histplot(vads_df[col], bins=30, kde=True)
    plt.title(f"{col} distribution")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(vads_df[["valence", "arousal", "dominance", "sentiment"]].corr(), annot=True, cmap="coolwarm")
plt.title("VADS Feature Correlation")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 2️⃣ top_tracks.json — 사용자 상호작용
# ------------------------------------------------------------
top_path = os.path.join(DATA_DIR, "top_tracks.json")
print(f"\n[INFO] Loading {top_path}...")
with open(top_path, "r", encoding="utf-8") as f:
    top_tracks = json.load(f)

user_counts = {u: len(tracks) for u, tracks in top_tracks.items()}
print(f"[INFO] Users: {len(user_counts):,}")
print(f"  ▶ 평균 트랙 수: {pd.Series(user_counts).mean():.2f}")
print(f"  ▶ 최대 트랙 수: {pd.Series(user_counts).max()}")
print(f"  ▶ 최소 트랙 수: {pd.Series(user_counts).min()}")

plt.figure(figsize=(6, 4))
sns.histplot(pd.Series(user_counts), bins=30)
plt.title("Top Tracks per User Distribution")
plt.xlabel("# of Tracks per User")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 3️⃣ item_tags.json — 태그 구조 미리보기
# ------------------------------------------------------------
tag_path = os.path.join(DATA_DIR, "item_tags.json")
if os.path.exists(tag_path):
    print(f"\n[INFO] Loading {tag_path}...")
    with open(tag_path, "r", encoding="utf-8") as f:
        tag_data = json.load(f)
    print(f"  ▶ Keys: {list(tag_data.keys())}")
    for k in tag_data.keys():
        print(f"  - {k}: {len(tag_data[k])} entries")
    print("\n[Sample Artist Tags]")
    print(list(tag_data.get("Artists", list(tag_data.values())[0]).items())[:5])
else:
    print("\n[WARN] item_tags.json not found, skipping tag analysis.")

# ------------------------------------------------------------
# 4️⃣ Summary 저장
# ------------------------------------------------------------

vads_df.describe().to_csv("lastfm_vads_summary.csv", encoding="utf-8-sig")
pd.Series(user_counts).describe().to_csv("lastfm_user_track_summary.csv", encoding="utf-8-sig")

print("\nEDA 완료 — 요약 저장: data/summary/")
