import sys
import os
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# Ensure local imports
REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.cf.svd_entry_shim import load_model


@st.cache_data(show_spinner=False)
def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def read_csv_safe(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


@st.cache_resource(show_spinner=False)
def load_pickle_safe(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def compute_scores(model, user_id: str, tracks_df: pd.DataFrame, id_col: str,
                   valence: float, energy: float, track_meta_db: dict | None,
                   interactions_df: pd.DataFrame | None) -> pd.DataFrame:
    df = tracks_df.copy()
    df["user_id"] = user_id

    # CF taste score
    # Surprise SVD .predict(uid, iid).est
    ests = []
    for iid in df[id_col].astype(str).tolist():
        try:
            est = model.predict(user_id, iid).est
        except Exception:
            est = 0.0
        ests.append(est)
    df["taste_score"] = ests

    # Emotion score: inverse Euclidean distance to (valence, energy)
    dist = ((df["valence"] - valence) ** 2 + (df["energy"] - energy) ** 2) ** 0.5
    df["emotion_score"] = 1.0 / (1.0 + dist)

    # Novelty score
    novelty = []
    if track_meta_db and id_col == "spotify_id":
        for iid in df[id_col].astype(str):
            meta = track_meta_db.get(iid, {})
            cnt = meta.get("total_rating_count", 0)
            novelty.append(1.0 / (cnt + 1.0))
    elif interactions_df is not None:
        counts = interactions_df.groupby("track_id").size().to_dict()
        for iid in df[id_col].astype(str):
            cnt = counts.get(iid, 0) if id_col == "track_id" else 0
            novelty.append(1.0 / (cnt + 1.0))
    else:
        novelty = [1.0] * len(df)
    df["novelty_score"] = novelty

    return df


def main():
    st.set_page_config(page_title="NVify Recommender", page_icon="🎵", layout="wide")
    st.title("🎵 NVify: Emotion-Aware Recommender")
    st.caption("Load trained artifacts (.pkl) and recommend by Valence/Energy")

    cfg_path = REPO / "config.yaml"
    if not cfg_path.exists():
        st.error("config.yaml not found.")
        st.stop()
    cfg = load_config(cfg_path)

    processed_dir = REPO / cfg["paths"]["processed_dir"]
    artifacts_dir = REPO / cfg["paths"]["artifacts_dir"]

    tracks_path = processed_dir / "tracks.csv"
    interactions_path = processed_dir / "interactions.csv"

    # UI controls
    with st.sidebar:
        st.header("Inputs")
        user_id = st.text_input("User ID", value="u0")
        valence = st.slider("Valence (0–1)", 0.0, 1.0, float(cfg["defaults"]["valence"]))
        energy = st.slider("Energy (0–1)", 0.0, 1.0, float(cfg["defaults"]["energy"]))
        top_k = st.slider("Top-K", 1, 50, int(cfg["serve"]["top_k"]))
        do_recommend = st.button("Recommend")

    # Load artifacts
    with st.spinner("Loading artifacts..."):
        model = load_model(cfg)
        if model is None:
            st.error("CF model not found or failed to load. Please run training first.")
            st.stop()

        # Optional artifacts
        cf_path = artifacts_dir / "cf_model_final.pkl"
        meta_path = artifacts_dir / "track_meta_db.pkl"
        rank_path = artifacts_dir / "ranking_model_final.pkl"

        track_meta_db = load_pickle_safe(meta_path) if meta_path.exists() else None
        ranker = load_pickle_safe(rank_path) if rank_path.exists() else None

        if not tracks_path.exists():
            st.error(f"Tracks CSV not found: {tracks_path}")
            st.stop()
        tracks_df = read_csv_safe(tracks_path)

        # Choose id column
        id_col = "spotify_id" if "spotify_id" in tracks_df.columns else "track_id"

        # Minimal columns check
        needed = [id_col, "name", "artist", "valence", "energy"]
        missing = [c for c in needed if c not in tracks_df.columns]
        if missing:
            st.error(f"tracks.csv missing columns: {missing}")
            st.stop()

        interactions_df = read_csv_safe(interactions_path) if interactions_path.exists() else None

    if do_recommend:
        with st.spinner("Scoring candidates..."):
            scored = compute_scores(model, user_id, tracks_df[needed], id_col,
                                    valence, energy, track_meta_db, interactions_df)

            # If ranker exists, predict; otherwise combine scores simply
            if ranker is not None:
                X = scored[["taste_score", "emotion_score", "novelty_score"]]
                try:
                    scored["predicted_score"] = ranker.predict(X)
                except Exception:
                    # Fallback simple blend
                    scored["predicted_score"] = (
                        0.7 * scored["taste_score"] + 0.2 * scored["emotion_score"] + 0.1 * scored["novelty_score"]
                    )
            else:
                scored["predicted_score"] = (
                    0.7 * scored["taste_score"] + 0.2 * scored["emotion_score"] + 0.1 * scored["novelty_score"]
                )

            out_cols = ["name", "artist", "predicted_score", "taste_score", "emotion_score", "novelty_score", id_col]
            recs = scored.sort_values("predicted_score", ascending=False)[out_cols].head(top_k)

        st.subheader("Recommendations")
        # Add Spotify link if possible
        if id_col == "spotify_id":
            recs = recs.copy()
            recs["spotify_link"] = recs[id_col].apply(lambda x: f"https://open.spotify.com/track/{x}")
        st.dataframe(recs, use_container_width=True)

        csv_bytes = recs.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="recommendations.csv", mime="text/csv")


if __name__ == "__main__":
    main()

