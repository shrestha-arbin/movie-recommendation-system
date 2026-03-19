from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


# -----------------------------
# Project root detection
# -----------------------------
def _find_project_root(start_dir: str) -> str:
    """
    Walk upward looking for a folder that contains BOTH:
      - 'data' directory
      - 'models' directory
    If not found, fall back to the parent of start_dir.
    """
    cur = os.path.abspath(start_dir)
    for _ in range(8):  # walk up max 8 levels
        data_dir = os.path.join(cur, "data")
        models_dir = os.path.join(cur, "models")
        if os.path.isdir(data_dir) and os.path.isdir(models_dir):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    # fallback (at least keep things relative)
    return os.path.abspath(os.path.join(start_dir, ".."))


PROJECT_ROOT = _find_project_root(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def _pick_existing(paths: List[str]) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of these files exist:\n" + "\n".join(paths))


def _safe_read_csv(paths: List[str]) -> pd.DataFrame:
    p = _pick_existing(paths)
    return pd.read_csv(p)


def _ensure_title_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable unique title key for UI lookup."""
    if "title_unique" in df.columns:
        return df
    if "title" not in df.columns or "movieId" not in df.columns:
        raise ValueError("movie_content_df must contain 'title' and 'movieId' columns.")
    df = df.copy()
    df["title_unique"] = df["title"].astype(str) + " [movieId=" + df["movieId"].astype(str) + "]"
    return df


def _minmax_norm(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - mn) / (mx - mn)


def _pick_svd_model_path() -> str:
    candidates = [
        os.path.join(MODELS_DIR, "svd_tuned.joblib"),
        os.path.join(MODELS_DIR, "svd_baseline.joblib"),
        os.path.join(MODELS_DIR, "svd_model.joblib"),
    ]
    return _pick_existing(candidates)


# -----------------------------
# Load everything
# -----------------------------
def load_data_and_models():
    """
    Returns:
      movie_content_df, ratings_df, movie_map,
      tfidf_vectorizer, tfidf_matrix,
      genome_nn, genome_indices_map,
      svd, user_rated_movies, all_movie_ids, meta
    """
    # ---- Data (try multiple locations) ----
    movie_content_paths = [
        os.path.join(DATA_DIR, "merged", "movie_content_final.csv"),
        os.path.join(DATA_DIR, "processed", "movie_content_final.csv"),
        os.path.join(DATA_DIR, "movie_content_final.csv"),
    ]
    ratings_paths = [
        os.path.join(DATA_DIR, "merged", "ratings_final.csv"),
        os.path.join(DATA_DIR, "processed", "ratings_final.csv"),
        os.path.join(DATA_DIR, "ratings_final.csv"),
    ]

    try:
        movie_content_df = _safe_read_csv(movie_content_paths)
        ratings_df = _safe_read_csv(ratings_paths)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n\n"
            "Fix:\n"
            "1) Run cleaning/preprocessing notebook to generate:\n"
            "   - data/merged/movie_content_final.csv\n"
            "   - data/merged/ratings_final.csv\n"
            "2) Or place them in data/processed/ or data/.\n"
        )

    movie_content_df = _ensure_title_unique(movie_content_df)

    keep_cols = [c for c in ["movieId", "title", "genres", "tmdbId", "title_unique"] if c in movie_content_df.columns]
    movie_map = movie_content_df[keep_cols].drop_duplicates("movieId").copy()

    # user rated map for filtering
    user_rated_movies: Dict[int, set] = {}
    if "userId" in ratings_df.columns and "movieId" in ratings_df.columns:
        user_rated_movies = (
            ratings_df.groupby("userId")["movieId"].apply(lambda s: set(map(int, s.values))).to_dict()
        )

    all_movie_ids = movie_map["movieId"].astype(int).unique().tolist()

    # ---- Models / artifacts ----
    tfidf_vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    tfidf_mat_path = os.path.join(MODELS_DIR, "tfidf_matrix.joblib")
    if not (os.path.exists(tfidf_vec_path) and os.path.exists(tfidf_mat_path)):
        raise FileNotFoundError(
            "Missing TF-IDF artifacts. Expected:\n"
            f"- {tfidf_vec_path}\n- {tfidf_mat_path}\n\n"
            "Fix: Run modelling.ipynb and save TF-IDF artifacts with joblib.dump(...)."
        )
    tfidf_vectorizer = joblib.load(tfidf_vec_path)
    tfidf_matrix = joblib.load(tfidf_mat_path)

    # Genome NN (optional)
    genome_nn = None
    genome_indices_map = None
    genome_nn_path = os.path.join(MODELS_DIR, "genome_nn.joblib")
    genome_indices_path = os.path.join(MODELS_DIR, "genome_indices_map.joblib")
    if os.path.exists(genome_nn_path) and os.path.exists(genome_indices_path):
        genome_nn = joblib.load(genome_nn_path)
        genome_indices_map = joblib.load(genome_indices_path)

    # Collaborative Filtering SVD (tuned preferred) - optional
    svd = None
    svd_path = None
    best_params = None
    try:
        svd_path = _pick_svd_model_path()
        try:
            svd = joblib.load(svd_path)
        except Exception as e:
            # Model may not deserialize if scikit-surprise isn't installed
            pass
        
        # best params (optional)
        params_path = os.path.join(MODELS_DIR, "svd_best_params.json")
        if os.path.exists(params_path):
            try:
                with open(params_path, "r", encoding="utf-8") as f:
                    best_params = json.load(f)
            except Exception:
                best_params = None
    except FileNotFoundError:
        # SVD model files don't exist
        pass

    meta = {"project_root": PROJECT_ROOT, "svd_path": svd_path, "svd_best_params": best_params}
    return (
        movie_content_df,
        ratings_df,
        movie_map,
        tfidf_vectorizer,
        tfidf_matrix,
        genome_nn,
        genome_indices_map,
        svd,
        user_rated_movies,
        all_movie_ids,
        meta,
    )


# -----------------------------
# Search
# -----------------------------
def search_title(movie_content_df: pd.DataFrame, query: str, top_k: int = 25) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q:
        return movie_content_df[["movieId", "title", "genres", "title_unique"]].head(top_k)

    mask = (
        movie_content_df["title_unique"].astype(str).str.lower().str.contains(q, na=False)
        | movie_content_df["title"].astype(str).str.lower().str.contains(q, na=False)
    )
    out = movie_content_df.loc[mask, ["movieId", "title", "genres", "title_unique"]].head(top_k)
    return out.reset_index(drop=True)


# -----------------------------
# Recommend: Text
# -----------------------------
def recommend_by_text(movie_content_df: pd.DataFrame, tfidf_matrix, title_unique: str, top_n: int = 10) -> pd.DataFrame:
    title_to_idx = pd.Series(movie_content_df.index, index=movie_content_df["title_unique"]).to_dict()
    if title_unique not in title_to_idx:
        raise KeyError(f"Title not found: {title_unique}")

    idx = title_to_idx[title_unique]
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = cosine_sim.argsort()[::-1][1 : top_n + 1]

    out = movie_content_df.loc[sim_indices, ["movieId", "title", "genres"]].copy()
    out["text_score"] = cosine_sim[sim_indices]
    return out.reset_index(drop=True)


# -----------------------------
# Recommend: Genome (optional)
# -----------------------------
def recommend_by_genome(
    movie_content_df: pd.DataFrame,
    genome_nn,
    genome_indices_map,
    title_unique: str,
    top_n: int = 10,
) -> Optional[pd.DataFrame]:
    if genome_nn is None or genome_indices_map is None:
        return None

    genome_cols = [c for c in movie_content_df.columns if c.startswith("tag_") or c.startswith("genome_")]
    if not genome_cols:
        genome_cols = [c for c in movie_content_df.columns if isinstance(c, str) and c.isdigit()]

    if not genome_cols:
        return None

    title_to_idx = pd.Series(movie_content_df.index, index=movie_content_df["title_unique"]).to_dict()
    if title_unique not in title_to_idx:
        return None

    idx = title_to_idx[title_unique]
    if float(movie_content_df.loc[idx, genome_cols].fillna(0).sum()) == 0.0:
        return None

    query_vec = movie_content_df.loc[idx, genome_cols].astype("float32").values.reshape(1, -1)
    distances, neighbor_pos = genome_nn.kneighbors(query_vec, n_neighbors=top_n + 1)

    rec_indices = genome_indices_map[np.array(neighbor_pos).flatten()]
    rec_indices = [int(i) for i in rec_indices if int(i) != int(idx)][:top_n]

    out = movie_content_df.loc[rec_indices, ["movieId", "title", "genres"]].copy()
    out["genome_score"] = (1.0 / (1.0 + distances.flatten()[1 : len(rec_indices) + 1])).astype(float)
    return out.reset_index(drop=True)


# -----------------------------
# Recommend: Collaborative Filtering
# -----------------------------
def recommend_by_cf(
    movie_map: pd.DataFrame,
    svd,
    user_rated_movies: Dict[int, set],
    all_movie_ids: List[int],
    user_id: int,
    top_n: int = 10,
    candidate_pool: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    # SVD model may not be available
    if svd is None:
        return pd.DataFrame(columns=["movieId", "title", "genres", "cf_score", "tmdbId"])
    
    rng = np.random.default_rng(random_state)

    rated = user_rated_movies.get(int(user_id), set())
    candidates = [mid for mid in all_movie_ids if mid not in rated]

    if len(candidates) == 0:
        return pd.DataFrame(columns=["movieId", "title", "genres", "cf_score", "tmdbId"])

    if len(candidates) > candidate_pool:
        candidates = rng.choice(candidates, size=candidate_pool, replace=False).tolist()

    preds = []
    try:
        for mid in candidates:
            est = float(svd.predict(int(user_id), int(mid)).est)
            preds.append((int(mid), est))
    except Exception:
        # SVD predict failed (e.g., scikit-surprise not available)
        return pd.DataFrame(columns=["movieId", "title", "genres", "cf_score", "tmdbId"])

    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:top_n]

    out = pd.DataFrame(top, columns=["movieId", "cf_score"])
    out = out.merge(movie_map, on="movieId", how="left")

    cols = [c for c in ["movieId", "title", "genres", "cf_score", "tmdbId"] if c in out.columns]
    return out[cols].reset_index(drop=True)


# -----------------------------
# Hybrid recommender
# -----------------------------
def recommend_hybrid_from_movie(
    movie_content_df: pd.DataFrame,
    movie_map: pd.DataFrame,
    tfidf_matrix,
    genome_nn,
    genome_indices_map,
    svd,
    user_rated_movies: Dict[int, set],
    all_movie_ids: List[int],
    title_unique: str,
    user_id: Optional[int] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    text_df = recommend_by_text(movie_content_df, tfidf_matrix, title_unique, top_n=50)
    genome_df = recommend_by_genome(movie_content_df, genome_nn, genome_indices_map, title_unique, top_n=50)

    cand = text_df[["movieId", "text_score"]].copy()

    if genome_df is not None and len(genome_df) > 0:
        cand = cand.merge(genome_df[["movieId", "genome_score"]], on="movieId", how="outer")
    else:
        cand["genome_score"] = np.nan

    if user_id is not None:
        cf_df = recommend_by_cf(
            movie_map, svd, user_rated_movies, all_movie_ids, int(user_id), top_n=200, candidate_pool=8000
        )
        if "cf_score" in cf_df.columns and len(cf_df) > 0:
            cand = cand.merge(cf_df[["movieId", "cf_score"]], on="movieId", how="outer")
        else:
            cand["cf_score"] = np.nan
    else:
        cand["cf_score"] = np.nan

    for c in ["text_score", "genome_score", "cf_score"]:
        cand[c] = cand[c].fillna(0.0).astype(float)

    cand["text_norm"] = _minmax_norm(cand["text_score"])
    cand["genome_norm"] = _minmax_norm(cand["genome_score"])
    cand["cf_norm"] = _minmax_norm(cand["cf_score"])

    if user_id is not None:
        cand["hybrid_score"] = 0.40 * cand["cf_norm"] + 0.40 * cand["text_norm"] + 0.20 * cand["genome_norm"]
    else:
        cand["hybrid_score"] = 0.70 * cand["text_norm"] + 0.30 * cand["genome_norm"] if cand["genome_norm"].sum() > 0 else cand["text_norm"]

    out = cand.sort_values("hybrid_score", ascending=False).head(top_n)
    out = out.merge(movie_map, on="movieId", how="left")

    cols = [c for c in ["movieId", "title", "genres", "hybrid_score", "tmdbId"] if c in out.columns]
    return out[cols].reset_index(drop=True)
