# NVify: Emotion-Aware Music Recommendation Pipeline

NVify combines emotional features (Valence/Energy) with user listening logs to recommend music. The repo is refactored so that a single `python main.py` runs preprocessing → optional Spotify mapping → CF (SVD) train/load → recommend → evaluate. CLI and logs default to Korean, but this README documents usage in English.

---

## Features
- Preprocessing: normalize two raw CSVs into standard processed artifacts under `data/processed/`
- Mapping (optional): run Spotify ID mapping if `tracks.csv` lacks `spotify_id`
- CF (SVD): train or load via adapters around the original notebook/scripts
- Recommend/Evaluate: reuse existing `recommender.py`, `recommend.py`, `recommend_evaluate.py`

---

## Project Structure
```
.
├── main.py                         # Orchestration (preprocess → mapping → CF → recommend → eval)
├── final_train_cf_mart.py          # CF(SVD) + LambdaMART training / artifacts CLI
├── requirements.txt                # Minimal deps (numpy<2, pandas, surprise, etc.)
├── config.yaml                     # Paths and default parameters
├── data/
│   ├── raw/                        # Raw CSV (Music Info.csv, User Listening History.csv)
│   ├── processed/                  # Processed artifacts (interactions, tracks, hybrid_*)
│   └── artifacts/                  # Model/evaluation artifacts (pkl, csv)
├── notebooks/
│   └── final_train_cf_mart.ipynb   # Reference notebook
├── docs/
│   └── dataset_schema.html         # Dataset schema (original)
├── src/
│   ├── preprocess/                 # Preprocessing scripts
│   ├── mapping/                    # Spotify mapping scripts (optional)
│   ├── cf/                         # SVD adapter
│   ├── serve/                      # Recommenders
│   └── eval/                       # Evaluation
├── preprocessing.md                # Preprocessing I/O + columns (detailed)
└── repreprocessing.md              # Original/target schema (full column lists)
```

---

## Requirements
- Python 3.9+ (virtualenv recommended)
- macOS + LightGBM: `brew install libomp`
- Install packages: `pip install -r requirements.txt`
  - Pins: `numpy<2`, `scikit-surprise`, `lightgbm`, `pandas`, `scikit-learn`, `tqdm`, `pyyaml`, `spotipy`

Virtualenv example
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

---

## Data Preparation
- Put two CSVs under `data/raw/`:
  - `Music Info.csv` (audio features and metadata; includes `spotify_id`, `valence`, `energy`)
  - `User Listening History.csv` (`user_id`, `track_id`, `playcount`)
- `main.py` copies them to `data/` for consistent relative paths.

See `repreprocessing.md` for complete schemas (all columns) and `preprocessing.md` for step-by-step I/O.

---

## Run
Run the full pipeline (preprocess → mapping → CF train/load → recommend → evaluate):

```
python3 main.py --user_id <USER_ID> [--valence 0.5] [--energy 0.5] [--top_k 10] [--eval]
```

Options (fall back to `config.yaml`):
- `--user_id`: target user ID for recommendation
- `--valence`, `--energy`: emotional preference (0–1)
- `--top_k`: number of recommendations
- `--eval`: run evaluation script

Env (optional):
- `NVIFY_TRAIN_MAX_ROWS`: cap training rows for CF/LTR (e.g., `50000`)

Artifacts
- Processed: `data/processed/{tracks.csv, interactions.csv, hybrid_drop1.csv}`
- Models: `data/artifacts/{cf_model_final.pkl, track_meta_db.pkl, ranking_model_final.pkl}`
- Output: `data/artifacts/recommendations.csv`

---

## Pipeline Overview
1) Preprocess (`src/preprocess/*.py`)
- `Music Info.csv` → `tracks.csv` (valence/energy ∈ [0,1], `emotion_vector`)
- `User Listening History.csv` → `interactions.csv` (log → percentile → rating ∈ [0,1])
- Merge → `hybrid_preprocessed.csv`; in Spotify-first mode → `hybrid_drop1.csv`

2) Mapping (optional)
- Skip if `tracks.csv` already has `spotify_id`
- Otherwise run `src/mapping/*` (Spotipy creds required)

3) CF (SVD) train/load
- `src/cf/svd_entry_shim.py` looks for model artifacts; if missing, runs `final_train_cf_mart.py`
- Input priority: `hybrid_drop1.csv` → `hybrid_preprocessed.csv` → `interactions.csv` (+ merge tracks)

4) Recommend / Evaluate
- Prefer `src/serve/recommender.py` (then `recommend.py`)
- Use `--eval` to run `src/eval/recommend_evaluate.py`

For detailed I/O and columns, see `preprocessing.md`.

---

## Configuration (config.yaml)
```
paths:
  raw_dir: data/raw
  processed_dir: data/processed
  artifacts_dir: data/artifacts

serve:
  top_k: 10

defaults:
  valence: 0.5
  energy: 0.5
```

---

## Troubleshooting
- Surprise/NumPy ABI issues
  - Symptom: `numpy.core.multiarray failed to import`
  - Fix: `pip install "numpy<2" scikit-surprise==1.1.3`

- LightGBM `libomp.dylib` not loaded
  - Fix: `brew install libomp`

- Spotify mapping credentials
  - If `tracks.csv` has `spotify_id`, mapping is skipped
  - Otherwise set `SPOTIPY_CLIENT_ID/SECRET` and run mapping scripts

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## References
- Schema: `repreprocessing.md`
- Preprocessing details: `preprocessing.md`
- Dataset schema (original): `docs/dataset_schema.html`
- Notebook: `notebooks/final_train_cf_mart.ipynb`
