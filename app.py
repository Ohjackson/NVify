from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from flask import Flask, render_template, request

# Ensure local imports work (src package)
REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Avoid accidental retraining when serving the app
os.environ.setdefault("NVIFY_FORCE_TRAIN", "0")

from src.cf.svd_entry_shim import load_model


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_pickle(path: Path):
    import pickle

    with path.open("rb") as fh:
        return pickle.load(fh)


def load_tracks(tracks_path: Path) -> pd.DataFrame:
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.csv not found at {tracks_path}")
    df = pd.read_csv(tracks_path)
    required = {"name", "artist", "valence", "energy"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"tracks.csv missing columns: {sorted(missing)}")
    # Prefer spotify_id if present, otherwise fallback to track_id; if absent, synthesize from track_id
    id_col = "spotify_id" if "spotify_id" in df.columns else "track_id"
    if id_col == "track_id" and "spotify_id" not in df.columns:
        df["spotify_id"] = df["track_id"].astype(str)
        id_col = "spotify_id"
    cols = [col for col in [id_col, "track_id", "name", "artist", "valence", "energy"] if col in df.columns]
    df = df[cols].dropna(subset=["name", "artist", "valence", "energy"])
    return df, id_col


def load_interaction_counts(path: Path) -> Optional[Dict[str, int]]:
    if not path.exists():
        return None
    # Only load track_id column to compute counts
    counts = pd.read_csv(path, usecols=["track_id"])["track_id"].value_counts()
    return counts.to_dict()


def compute_scores(
    model,
    user_id: str,
    tracks_df: pd.DataFrame,
    id_col: str,
    valence: float,
    energy: float,
    track_meta_db: Optional[dict],
    interaction_counts: Optional[Dict[str, int]],
) -> pd.DataFrame:
    df = tracks_df.copy()
    df["user_id"] = user_id

    # Taste score from CF model
    taste_scores: List[float] = []
    for iid in df[id_col].astype(str).tolist():
        try:
            est = model.predict(user_id, iid).est
        except Exception:
            est = 0.0
        taste_scores.append(est)
    df["taste_score"] = taste_scores

    # Emotion score (inverse distance)
    distance = ((df["valence"] - valence) ** 2 + (df["energy"] - energy) ** 2) ** 0.5
    df["emotion_score"] = 1.0 / (1.0 + distance)

    # Novelty score
    novelty: List[float] = []
    if track_meta_db and id_col == "spotify_id":
        for iid in df[id_col].astype(str):
            meta = track_meta_db.get(iid, {})
            cnt = meta.get("total_rating_count", 0)
            novelty.append(1.0 / (cnt + 1.0))
    elif interaction_counts is not None and "track_id" in df.columns:
        for iid in df["track_id"].astype(str):
            cnt = interaction_counts.get(iid, 0)
            novelty.append(1.0 / (cnt + 1.0))
    else:
        novelty = [1.0] * len(df)
    df["novelty_score"] = novelty

    return df


def blend_scores(df: pd.DataFrame, ranker) -> pd.DataFrame:
    if ranker is not None:
        feats = df[["taste_score", "emotion_score", "novelty_score"]]
        try:
            df["predicted_score"] = ranker.predict(feats)
            return df
        except Exception:
            pass
    df["predicted_score"] = (
        0.7 * df["taste_score"] + 0.2 * df["emotion_score"] + 0.1 * df["novelty_score"]
    )
    return df


def build_result_rows(df: pd.DataFrame, id_col: str, top_k: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    top = df.sort_values("predicted_score", ascending=False).head(top_k)
    for idx, (_, row) in enumerate(top.iterrows(), start=1):
        spotify_id = row.get("spotify_id") if id_col == "spotify_id" else None
        track_link = f"https://open.spotify.com/track/{spotify_id}" if spotify_id else None
        preview_url = (
            f"https://open.spotify.com/embed/track/{spotify_id}" if spotify_id else None
        )
        rows.append(
            {
                "rank": idx,
                "title": row.get("name", ""),
                "artist": row.get("artist", ""),
                "track_url": track_link,
                "preview_url": preview_url,
                "taste": row["taste_score"],
                "emotion": row["emotion_score"],
                "novelty": row["novelty_score"],
                "score": row["predicted_score"],
            }
        )
    return rows


# Bootstrap shared assets
CFG = load_config(REPO / "config.yaml")
PROCESSED_DIR = REPO / CFG["paths"]["processed_dir"]
ARTIFACTS_DIR = REPO / CFG["paths"]["artifacts_dir"]
TRACKS_PATH = PROCESSED_DIR / "tracks.csv"
INTERACTIONS_PATH = PROCESSED_DIR / "interactions.csv"
META_PATH = ARTIFACTS_DIR / "track_meta_db.pkl"
RANKING_PATH = ARTIFACTS_DIR / "ranking_model_final.pkl"

MODEL = load_model(CFG)
if MODEL is None:
    raise RuntimeError("CF model could not be loaded. Run training first (train.sh).")

TRACKS_DF, ID_COL = load_tracks(TRACKS_PATH)
INTERACTION_COUNTS = load_interaction_counts(INTERACTIONS_PATH)
TRACK_META_DB = load_pickle(META_PATH) if META_PATH.exists() else None
RANKER = load_pickle(RANKING_PATH) if RANKING_PATH.exists() else None

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    errors: List[str] = []
    results: List[Dict[str, Any]] | None = None

    default_valence = float(CFG["defaults"]["valence"])
    default_energy = float(CFG["defaults"]["energy"])
    default_topk = int(CFG["serve"]["top_k"])

    user_id = request.form.get("user_id", "u0")
    valence_raw = request.form.get("valence", str(default_valence))
    energy_raw = request.form.get("energy", str(default_energy))
    topk_raw = request.form.get("top_k", str(default_topk))

    if request.method == "POST":
        try:
            valence = max(0.0, min(1.0, float(valence_raw)))
        except ValueError:
            errors.append("Valence must be a number between 0 and 1.")
            valence = default_valence
        try:
            energy = max(0.0, min(1.0, float(energy_raw)))
        except ValueError:
            errors.append("Energy must be a number between 0 and 1.")
            energy = default_energy
        try:
            top_k = max(1, int(topk_raw))
        except ValueError:
            errors.append("Top-K must be an integer ≥ 1.")
            top_k = default_topk

        if not user_id.strip():
            errors.append("User ID cannot be empty.")

        if not errors:
            scored = compute_scores(
                MODEL,
                user_id.strip(),
                TRACKS_DF[[col for col in TRACKS_DF.columns if col in {ID_COL, "track_id", "name", "artist", "valence", "energy"}]],
                ID_COL,
                valence,
                energy,
                TRACK_META_DB,
                INTERACTION_COUNTS,
            )
            scored = blend_scores(scored, RANKER)
            results = build_result_rows(scored, ID_COL, top_k)
    else:
        valence = default_valence
        energy = default_energy
        top_k = default_topk

    return render_template(
        "index.html",
        user_id=user_id,
        valence=valence,
        energy=energy,
        top_k=top_k,
        errors=errors,
        results=results,
        has_spotify=(ID_COL == "spotify_id"),
    )


if __name__ == "__main__":
    app.run(debug=False)
