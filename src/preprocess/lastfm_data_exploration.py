# ==========================================================
# 🎵 NVify - Last.fm 원시 데이터 탐색 및 분석 (EDA)
# ==========================================================
#
# 📝 설명:
#   Last.fm 원시 JSON 데이터셋(감정 피처, 사용자 상호작용, 태그 구조)의 
#   구조, 분포, 통계적 특성을 탐색하고 시각화합니다. 이 과정은 후속 전처리 및
#   모델 학습을 위한 데이터 이해도를 높이는 데 필수적입니다.
#
# ----------------------------------------------------------
# 📁 파일 정보
# ----------------------------------------------------------
#
# ➡️ 입력 파일 (Input):
#   - item_vads.json: 트랙 VADS(Valence, Arousal, Dominance, Sentiment) 점수
#   - top_tracks.json: 사용자별 Top Tracks 청취 기록
#   - item_tags.json (선택): 항목별 사용자 부여 태그 요약
#
# ⬅️ 출력 파일 (Output):
#   - lastfm_vads_summary.csv: VADS 피처의 기술 통계 요약
#   - lastfm_user_track_summary.csv: 사용자별 트랙 수 통계 요약
#   - [Images]: VADS 분포 및 상관관계, 사용자 상호작용 분포 히스토그램 (실행 시)
#
# 🛠️ 주요 라이브러리:
#   - pandas, matplotlib, seaborn, json
#
# ==========================================================
# 1. 설정 및 초기화
# ==========================================================

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================================
# 2. 핵심 로직: 데이터 탐색 (EDA)
# ==========================================================

# ------------------------------------------------------------
# 2.1 item_vads.json — 감정 피처
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

# 시각화 (실행 환경에서만 작동)
for col in ["valence", "arousal", "dominance", "sentiment"]:
    plt.figure(figsize=(6, 4))
    sns.histplot(vads_df[col], bins=30, kde=True)
    plt.title(f"{col} distribution")
    plt.tight_layout()
    # plt.show() # 문서화 목적상 주석 처리

plt.figure(figsize=(6, 5))
sns.heatmap(vads_df[["valence", "arousal", "dominance", "sentiment"]].corr(), annot=True, cmap="coolwarm")
plt.title("VADS Feature Correlation")
plt.tight_layout()
# plt.show() # 문서화 목적상 주석 처리


# ------------------------------------------------------------
# 2.2 top_tracks.json — 사용자 상호작용
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
# plt.show() # 문서화 목적상 주석 처리


# ------------------------------------------------------------
# 2.3 item_tags.json — 태그 구조 미리보기
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
    # tag_data가 비어있지 않다면 샘플을 출력합니다.
    if tag_data:
        first_key_data = list(tag_data.values())[0] 
        print(list(first_key_data.items())[:5])
else:
    print("\n[WARN] item_tags.json not found, skipping tag analysis.")

# ==========================================================
# 3. 메인 실행 로직: Summary 저장
# ==========================================================

vads_df.describe().to_csv("lastfm_vads_summary.csv", encoding="utf-8-sig")
pd.Series(user_counts).describe().to_csv("lastfm_user_track_summary.csv", encoding="utf-8-sig")

print("\nEDA 완료 — 요약 저장: lastfm_vads_summary.csv, lastfm_user_track_summary.csv")