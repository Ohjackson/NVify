# NVify: Emotion-Aware Music Recommendation System

NVify is a full-stack pipeline that ingests user listening histories plus track-level affective features (valence and energy) to generate personalized music recommendations. The project bundles data preprocessing, collaborative filtering (Surprise SVD), optional LambdaMART ranking, evaluation scripts, and a Flask UI so you can go from raw Kaggle CSVs to a running demo with a single command.

---

## Table of Contents
1. [Key Capabilities](#key-capabilities)
2. [Repository Layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Data Acquisition & Preparation](#data-acquisition--preparation)
6. [Environment Setup](#environment-setup)
7. [Pipeline in Depth](#pipeline-in-depth)
8. [Command Reference](#command-reference)
9. [Configuration](#configuration)
10. [Artifacts & Directory Conventions](#artifacts--directory-conventions)
11. [Web App Usage](#web-app-usage)
12. [Troubleshooting](#troubleshooting)
13. [Development Tips](#development-tips)
14. [License & References](#license--references)

---

## Key Capabilities
- **End-to-end orchestration**: `python main.py` runs preprocessing → Spotify ID enrichment → CF training or loading → recommendation → optional evaluation.
- **Rich preprocessing**: normalizes valence/energy to [0, 1], builds `emotion_vector`, and maps user play counts to percentile-based implicit ratings.
- **Model stack**:
  - Surprise SVD collaborative filtering as the baseline recommender.
  - Optional LightGBM LambdaMART ranker that blends CF taste scores with emotion/novelty signals.
- **Artifact management**: trained pickles, processed CSVs, and evaluation outputs are versioned under `data/artifacts/` and `data/processed/`.
- **Flask UI**: interactive dashboard for valence/energy sliders, Spotify previews, and ranked results.
- **Automation helpers**: `bootstrap.sh`, `make_processed.sh`, `train.sh`, and `start.sh` orchestrate environment setup and reproducible runs.

---

## Repository Layout
```
.
├── app.py                     # Flask application entry point
├── bootstrap.sh               # Create .venv, install deps (builds numpy from source)
├── config.yaml                # Default paths and hyperparameters
├── data/
│   ├── raw/                   # Required Kaggle CSVs (tracked): Music Info.csv, User Listening History.csv
│   ├── processed/             # Derived artifacts (tracks.csv, interactions.csv, hybrid_*)
│   └── artifacts/             # Model pickles, evaluation outputs
├── final_train_cf_mart.py     # Standalone CF + LambdaMART trainer
├── main.py                    # High-level pipeline orchestrator
├── make_processed.sh          # Idempotent processed-data builder (+ fallback placeholders)
├── notebooks/final_train_cf_mart.ipynb
├── requirements.txt           # `numpy<2`, `pandas<2.3`, `scikit-surprise`, etc.
├── src/
│   ├── preprocess/            # CSV normalization, merging, Kaggle download helpers
│   ├── mapping/               # Spotify ID enrichment utilities
│   ├── cf/                    # Surprise SVD adapter + lazy trainer
│   ├── serve/                 # CLI recommenders
│   └── eval/                  # Recommendation evaluation scripts
├── start.sh                   # Bootstrap + make_processed + launch Flask UI
├── templates/                 # HTML templates for the web app
├── train.sh                   # Opinionated training runner (forces artifacts refresh)
├── preprocessing.md           # Detailed preprocessing I/O spec
└── repreprocessing.md         # Original Kaggle schema documentation
```

---

## Prerequisites
- **OS**: Tested on macOS Sonoma (ARM). Linux works with equivalent packages.
- **Python**: 3.11 recommended. Surprise + LightGBM require a CPython build with shared OpenMP (`libomp` on macOS).
- **System dependencies**:
  - macOS: `brew install libomp python@3.11` (train.sh will install python@3.11 if missing).
  - Kaggle download: Kaggle account (for API terms) but no API token is required when using `kagglehub`.
- **Disk**: ~2 GB for raw CSVs + processed artifacts. `User Listening History.csv` alone is ~575 MB.
- **Network**: Needed once to fetch Python wheels and Kaggle data.

---

## Quick Start
1. **Clone & enter repo**
   ```bash
   git clone <your-fork-or-origin> NVify
   cd NVify
   ```
2. **Create/refresh the virtualenv** (builds NumPy from source to avoid macOS seatbelt crashes):
   ```bash
   bash bootstrap.sh
   ```
3. **Download Kaggle dataset** (run inside the repo):
   ```bash
   python - <<'PY'
   import kagglehub
   path = kagglehub.dataset_download('undefinenull/million-song-dataset-spotify-lastfm')
   print('Downloaded to:', path)
   PY
   cp "$path/Music Info.csv" data/raw/
   cp "$path/User Listening History.csv" data/raw/
   ```
4. **Generate processed CSVs** (tracks + interactions):
   ```bash
   bash make_processed.sh
   ```
   The script will attempt the full pipeline; if it cannot complete (e.g., low RAM), it at least creates placeholder CSVs so the web UI can boot.
5. **Train models & evaluate** (optional but recommended):
   ```bash
   bash train.sh        # forces SVD/LambdaMART retraining using .venv/.venv311
   ```
6. **Launch the Flask app**:
   ```bash
   ./start.sh           # boots venv, ensures processed data, runs app.py
   ```
   Visit the printed URL (default http://127.0.0.1:5000) and start exploring recommendations.

---

## Data Acquisition & Preparation
### Required CSVs (tracked in git)
| File | Location | Purpose |
| --- | --- | --- |
| `data/raw/Music Info.csv` | Kaggle dataset | Track metadata + audio features (`spotify_id`, `valence`, `energy`, artist/name, etc.) |
| `data/raw/User Listening History.csv` | Kaggle dataset | Long-form log of (`user_id`, `track_id`, `playcount`) pairs. |

Both files must exist before running `main.py`, `make_processed.sh`, or `train.sh`. They are added to version control so collaborators can run the pipeline without re-downloading (beware GitHub's 100 MB soft limit; consider storing large files in LFS if pushing upstream).

### Optional automatic download
`src/preprocess/data_pull.py` uses [`kagglehub`](https://github.com/Kaggle/kagglehub) to fetch `undefinenull/million-song-dataset-spotify-lastfm`. Trigger it manually with:
```bash
python -m kagglehub dataset download undefinenull/million-song-dataset-spotify-lastfm
```
The helper copies the CSVs into `data/raw/` if they are missing.

### Processed outputs
Running the preprocessing pipeline (either via `main.py` or `make_processed.sh`) produces:
- `data/processed/tracks.csv`: normalized track catalog; includes `emotion_vector` column `[valence, energy]` and stringified Spotify IDs.
- `data/processed/interactions.csv`: implicit rating matrix derived from listening history (log scaling + percentile normalization).
- `data/processed/hybrid_preprocessed.csv` and `data/processed/hybrid_drop1.csv`: joined datasets for CF/LTR.
These are consumed by `final_train_cf_mart.py` and the serving layer.

---

## Environment Setup
### Using `bootstrap.sh`
`bootstrap.sh` encapsulates the recommended setup:
1. Creates `.venv` (or uses `VENV_DIR` override) with the best available Python (prefers `/opt/homebrew/bin/python3.11`).
2. Activates the environment and upgrades `pip`.
3. Installs project requirements with `pip install --no-binary numpy -r requirements.txt` so NumPy is built from source (macOS wheels + Sandbox often segfault).

Run it anytime dependencies change:
```bash
bash bootstrap.sh
```

### Alternative manual setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install --no-binary numpy -r requirements.txt
```

### Additional helpers
- `train.sh`: ensures `.venv311` with Homebrew Python 3.11, installs deps, flips `config.yaml` to enable ranking, exports `NVIFY_FORCE_TRAIN=1`, and runs `python main.py --eval`.
- `make_processed.sh`: idempotently builds `tracks.csv`/`interactions.csv` (or provides placeholders) before the UI is launched.
- `start.sh`: wraps everything—bootstraps (if needed), generates processed data, then launches the Flask app with `.venv/bin/python`.

---

## Pipeline in Depth
1. **Preprocessing (`src/preprocess/`)**
   - `data_preprocessing_pipeline.py` reads the two raw CSVs under `data/`, cleans them, normalizes valence/energy via `MinMaxScaler`, creates `emotion_vector`, and converts playcounts to user-percentile ratings.
   - Outputs: `music_emotion_clean.csv`, `user_ratings_normalized.csv`, `hybrid_preprocessed.csv` (later copied into `data/processed/`).

2. **Spotify ID mapping (`src/mapping/`)** *(optional)*
   - Skipped if `tracks.csv` already contains `spotify_id`.
   - Otherwise, `merge_spotify_id.py` and `map_spotify_ids.py` merge Spotipy lookups using your credentials (`SPOTIPY_CLIENT_ID/SECRET`).

3. **Collaborative filtering (`src/cf/svd_entry_shim.py` + `final_train_cf_mart.py`)**
   - `load_model()` checks `data/artifacts/{cf_model_final.pkl, track_meta_db.pkl}`; if missing or `NVIFY_FORCE_TRAIN=1`, it runs `final_train_cf_mart.py`.
   - The script trains Surprise SVD (configurable factors/epochs) and builds a track metadata DB (valence/energy counts).
   - Optional LambdaMART ranking uses LightGBM with features `[taste_score, emotion_score, novelty_score]` per user query.

4. **Serving & evaluation (`src/serve/`, `src/eval/`)**
   - `src/serve/recommender.py` and `recommend.py` produce CSV outputs; `recommend_evaluate.py` computes metrics when `--eval` is passed to `main.py`.
   - `data/artifacts/recommendations.csv` stores the latest batch recommendations.

5. **Web layer (`app.py`)**
   - Loads artifacts, exposes UI controls (user_id, valence, energy, top_k), and embeds Spotify previews.
   - Honors env vars: `NVIFY_FORCE_TRAIN`, `NVIFY_ARTIFACTS_DIR`, etc.

---

## Command Reference
| Command | Description | Key Options |
| --- | --- | --- |
| `python main.py --user_id u123 --valence 0.6 --energy 0.4 --top_k 15 --eval` | Runs the entire pipeline. Skips preprocessing if `tracks.csv`/`interactions.csv` already exist. | `--config`, `--user_id`, `--valence`, `--energy`, `--top_k`, `--eval` |
| `python app.py` | Launches the Flask UI using the active interpreter (assumes deps/artifacts already exist). | Set `NVIFY_FORCE_TRAIN=0` to avoid retraining on startup. |
| `python final_train_cf_mart.py --input_csv data/processed/hybrid_drop1.csv --skip_ranking` | Re-trains CF (and optionally LambdaMART) outside of `main.py`. | `--max_rows`, `--n_factors`, `--n_epochs`, `--rank_estimators`, etc. |
| `bash make_processed.sh` | Builds `data/processed/{tracks,interactions}` via the main pipeline or placeholder fallbacks. | None |
| `bash train.sh [USER_ID]` | Forces retraining with optional user override; installs Homebrew Python/libomp if missing. | `NVIFY_FORCE_TRAIN`, `NVIFY_TRAIN_MAX_ROWS` env vars |
| `./start.sh` | Bootstraps, builds processed data (if needed), and launches the web app via `.venv/bin/python`. | `NVIFY_ARTIFACTS_DIR`, `NVIFY_FORCE_TRAIN`, etc. |

---

## Configuration
`config.yaml` defines defaults used by `main.py` and downstream modules:
```yaml
paths:
  raw_dir: data/raw              # where Music Info.csv and User Listening History.csv live
  processed_dir: data/processed  # where tracks/interactions/hybrid CSVs are written
  artifacts_dir: data/artifacts  # where models & outputs (pkl/csv) are stored

serve:
  top_k: 10                      # default number of recommendations in UI/CLI

defaults:
  valence: 0.5
  energy: 0.5

train:
  always_train: false            # set true or export NVIFY_FORCE_TRAIN=1 to retrain every run
  skip_ranking: true             # toggle to false (or NVIFY_SKIP_RANKING=0) to train LambdaMART
  max_rows: 50000                # 0 = use full dataset; env override NVIFY_TRAIN_MAX_ROWS
```
Additional environment variables:
- `NVIFY_FORCE_TRAIN` ("1" forces CF retraining even if artifacts exist; default "0" in app.py).
- `NVIFY_SKIP_RANKING` ("1" to bypass LambdaMART regardless of config).
- `NVIFY_TRAIN_MAX_ROWS` (caps training rows for faster iteration).
- `NVIFY_ARTIFACTS_DIR` (override artifact root; default `data/artifacts`).
- `NVIFY_USER_ID`, `NVIFY_VALENCE`, `NVIFY_ENERGY`, `NVIFY_TOP_K` are set internally when `main.py` runs.

---

## Artifacts & Directory Conventions
| Path | Contents |
| --- | --- |
| `data/processed/tracks.csv` | Track catalog with normalized valence/energy + `emotion_vector` (JSON-like list).
| `data/processed/interactions.csv` | Implicit ratings per (`user_id`, `track_id`).
| `data/processed/hybrid_drop1.csv` | Joined dataset (interactions × tracks) with Spotify IDs prioritized.
| `data/artifacts/cf_model_final.pkl` | Surprise SVD model (pickled). |
| `data/artifacts/track_meta_db.pkl` | Dict keyed by `spotify_id` containing valence, energy, rating counts. |
| `data/artifacts/ranking_model_final.pkl` | LightGBM LambdaMART model (if `skip_ranking=false`). |
| `data/artifacts/recommendations.csv` | Latest recommendation batch generated by `main.py`/`app.py`. |

Backing up `data/processed/` and `data/artifacts/` is enough to reproduce recommendations without re-running heavy preprocessing or training.

---

## Web App Usage
1. Ensure `.venv` is active or let `start.sh` handle it.
2. Run `./start.sh` (or `python app.py` if everything is already prepared).
3. Open the logged URL (default http://127.0.0.1:5000).
4. Provide inputs:
   - `user_id`: must exist in `interactions.csv`.
   - `Valence` / `Energy`: floats in [0, 1]; slider defaults come from `config.yaml`.
   - `Top-K`: number of items to display.
5. Submit to see ranked tracks, each containing Spotify preview embeds when `spotify_id` is available.

To prevent retraining every time the server starts, set `export NVIFY_FORCE_TRAIN=0` or edit `train.always_train` in `config.yaml`.

---

## Troubleshooting
| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'yaml'` when `start.sh` runs | `make_processed.sh` fell back to system Python (PEP 668 protected) | Ensure `.venv` is created (`bash bootstrap.sh`) and rerun start; already-generated processed CSVs will skip this path. |
| `numpy.core.multiarray failed to import` when importing `surprise` | Binary wheels compiled against NumPy 1.x on a NumPy 2.x runtime | Keep `numpy<2` (built from source) as pinned in `requirements.txt`. Reinstall via `pip install --no-binary numpy -r requirements.txt`. |
| Segmentation fault on macOS when running pandas/numpy | Mixing system Python with sandboxed Accelerate/vecLib | Always operate inside `.venv` and build NumPy from source (already handled by `bootstrap.sh`). |
| LightGBM complains about missing `libomp.dylib` | OpenMP runtime absent | `brew install libomp` (or install the appropriate package on Linux). |
| Spotify mapping step fails | Missing Spotipy credentials or `tracks.csv` lacks `spotify_id` | Set `SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET`, `SPOTIPY_REDIRECT_URI`; rerun `main.py` with `--eval` or `python src/mapping/map_spotify_ids.py`. |
| Kaggle download blocked | No network or Kaggle terms not accepted | Download manually from Kaggle and place the CSVs under `data/raw/`. |

---

## Development Tips
- **Formatting**: keep files ASCII; comments should explain non-obvious behavior (e.g., why NumPy must be built from source).
- **Testing**: for heavy scripts, prefer sampling via `NVIFY_TRAIN_MAX_ROWS=5000 python main.py ...` to iterate quickly.
- **Dirty worktree warning**: scripts intentionally avoid touching unrelated files. If you see unexpected changes you did not make, investigate before committing.
- **Large files**: GitHub rejects files >100 MB over HTTPS. If you plan to push the raw CSVs upstream, consider Git LFS or storing them elsewhere.

---

## License & References
- Licensed under the MIT License (see `LICENSE`).
- Kaggle dataset: [Million Song Dataset + Spotify + Last.fm (undefinenull)](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm).
- Original notebook: `notebooks/final_train_cf_mart.ipynb`.
- Supplemental docs: `preprocessing.md`, `repreprocessing.md`, `docs/dataset_schema.html`.

Happy recommending!
