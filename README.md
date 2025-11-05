# TwoOfUsTunes

두 사람의 감정(Valence–Arousal)을 결합해 **공통 분위기의 플레이리스트**를 추천하는 머신러닝 엔진

---

## 1. 서비스 목적 & 흐름

### 목적

* **감정 상태(Valence–Arousal, ±1 범위)**에 맞는 음악을 추천
* **2인 모드**에서 두 사람의 감정을 조합해 “둘이 들을” 공통 분위기 플레이리스트 생성
* 행동 피드백(스킵/좋아요/완청)을 받아 **점점 개인화**되는 추천 (가능하면)

### 기본 흐름

1. 사용자 입력 → 슬라이더(Valence, Arousal) + (선택) 컨텍스트/행동 로그
2. `state_estimator`가 감정 벡터 **E=(v,a)** 및 불확실성 산출
3. `recommender`가 곡 임베딩(va, tempo, energy, danceability 등)과 **유사도 기반 Top-K** 추천
4. (2인) `blend` 전략(**avg / intersect / union**)으로 공통 리스트 재랭킹
5. 사용 피드백을 `feedback`에서 수집 → 액티브 러닝 훅으로 업데이트 포인트 제공

---

## 2. 기술 스택

* **언어**: Python 3.10+
* **ML/데이터**: NumPy, Pandas, scikit-learn 
* **시각화/평가**: Matplotlib
* **실험**: 
* **CLI/도구**: 
* **패키징**: `requirements.txt` (또는 추후 `pyproject.toml` 전환 가능)

---

## 3. 설치 & 로컬 개발

### 요구 사항

* Python **3.10+**
* (오디오 피처 사용 시) **ffmpeg**, **librosa**가 사용하는 시스템 라이브러리
* macOS/Windows/Linux 모두 지원

### 설치

```bash
git clone https://github.com/<your-org>/TwoOfUsTunes.git
cd TwoOfUsTunes

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 데이터 준비(예시)



### 실행 예시 (CLI)


---

## 4. 파일 구조

```text
TwoOfUsTunes/
├─ data/
│  ├─ raw/                      # 원본 데이터 (gitignore 또는 LFS)
│  └─ processed/                # 표준화된 데이터셋 (tracks.parquet 등)
├─ features/
│  ├─ extract_audio.py          # 오디오 피처: MFCC/Chroma/Tempo...
│  ├─ extract_lyric.py          # 가사 임베딩: TF-IDF/BERT 등
│  └─ build_index.py            # (선택) FAISS/Annoy 인덱스
├─ engine/
│  ├─ recommender.py            # Top-K 추천, 다양성 제약, 재랭킹
│  ├─ blend.py                  # 2인 전략: avg/intersect/union
│  ├─ state_estimator.py        # 입력 융합 → (v,a,uncertainty)
│  ├─ feedback.py               # 스킵/좋아요/완청 로그 처리
│  ├─ models/                   # (선택) 회귀/분류 모델 아티팩트
│  └─ utils.py
├─ sensing/                     # (선택) 입력 플러그인
│  ├─ ser.py                    # 음성 감정(SER) - 로컬 처리
│  └─ fer.py                    # 표정 감정(FER) - 동의 필수
├─ eval/
│  ├─ offline_metrics.py        # NDCG/Hit/다양성 계산
│  ├─ plots.py                  # VA 맵/분포/지표 그래프
│  └─ ab_simulator.py           # (선택) 전략 비교 시뮬레이터
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_train_va_regressor.ipynb
│  └─ 03_eval_recommender.ipynb
├─ experiments/                 # 실험 설정/로그/결과
│  ├─ configs/
│  ├─ logs/
│  └─ results/
├─ results/                     # 사용자 실행 산출(csv 등)
├─ tests/
│  ├─ test_recommender.py
│  ├─ test_state_estimator.py
│  └─ test_utils.py
├─ cli.py
├─ requirements.txt
├─ README.md
└─ .gitignore
```

`.gitignore` 예시

```
data/raw/*
model_artifacts/*
results/*.csv
.venv/
__pycache__/
*.pyc
*.ipynb_checkpoints
*.mp3
*.wav
```

---

## 5. 브랜치 전략 (4개 고정)

* `main` : 릴리스/발표용. **직접 푸시/포스푸시 모두 금지**, 태그만 생성.
* `part/data` : 데이터·피처링 전용
* `part/model` : 모델·추천 전용
* `part/sensing` : 입력/감정추정 전용
* `part/exp` : 실험·평가·문서/CLI 전용


---

## 6. 커밋 컨벤션 (Conventional Commits)

**type 예시**

* `feat` 새 기능
* `fix` 버그 수정
* `docs` 문서 수정
* `refactor` 리팩터링(기능 변경 없음)
* `perf` 성능 개선
* `test` 테스트 관련
* `build` 빌드/의존성
* `chore` 자잘한 변경

**예시**

```
타입: ~내용~ - 브런치 이름

아래 예시입니다.


feat: add top-k cosine similarity ranking - data
fix: handle empty context gracefully - model

```

---

## 7. PR 규칙 & 템플릿

### 규칙

---

## 8. 개발 규칙(코드 스타일 & 테스트)

- 다른 파트 파일 건들지 않기 , 

---


## 9. Requirements (의존성)

`requirements.txt` (예시)

```
numpy
pandas
scikit-learn
librosa
faiss-cpu     # (선택) 대용량 인덱싱에 사용
matplotlib
scipy
tqdm
```

---

## 11. 라이선스

```
MIT License

Copyright (c) 2025 Ahn Jaehyun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---


