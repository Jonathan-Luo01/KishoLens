"""
semantic_match.py — Live inference for semantic genre/territory matching.

Loads pre-built genre centroids from disk and matches a user's text
to the closest genre using cosine similarity over sentence embeddings.

Gracefully returns None if centroids have not been built yet.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from kisholens.ml.build_centroids import (
    embed_texts,
    load_centroids_from_disk,
)

# Module-level centroid cache: {data_dir: {"genre": (centroids, meta), "territory": (centroids, meta)}}
_centroid_cache: dict[str, dict[str, tuple[np.ndarray, dict]]] = {}

# Default centroid location (relative to project root)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)


def _load_with_cache(data_dir: str) -> tuple[Optional[np.ndarray], Optional[dict], Optional[np.ndarray], Optional[dict]]:
    """
    Load centroids from disk, caching per data_dir so repeated calls within
    the same process do not re-read files.
    """
    if data_dir not in _centroid_cache:
        g_centroids, g_meta = load_centroids_from_disk("genre", data_dir)
        t_centroids, t_meta = load_centroids_from_disk("territory", data_dir)
        if g_centroids is not None and t_centroids is not None:
            _centroid_cache[data_dir] = {
                "genre": (g_centroids, g_meta),
                "territory": (t_centroids, t_meta)
            }
        else:
            return None, None, None, None
    entry = _centroid_cache[data_dir]
    return entry["genre"][0], entry["genre"][1], entry["territory"][0], entry["territory"][1]


def match_semantic(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
    data_dir: str = DEFAULT_DATA_DIR,
) -> Optional[dict]:
    """
    Embed `text` and compute cosine similarity against all genre and territory centroids.

    Returns None if centroids have not been built (data files absent).

    Returns a dict:
    {
        "genre":                str,    # canonical genre with highest similarity
        "genre_confidence":     float,  # highest genre cosine similarity score
        "genre_scores": [               # ALL genres, sorted descending by score
            {"genre": str, "score": float},
            ...
        ],
        "territory":            str,    # territory with highest similarity
        "territory_confidence": float,  # highest territory cosine similarity score
        "territory_scores": [           # ALL territories, sorted descending by score
            {"territory": str, "score": float},
            ...
        ]
    }
    """
    g_centroids, g_meta, t_centroids, t_meta = _load_with_cache(data_dir)
    if g_centroids is None or g_meta is None or t_centroids is None or t_meta is None:
        return None

    # Embed the input text
    embedding = embed_texts([text], model_name=model_name)  # (1, 384)

    # Cosine similarity using pure numpy:
    emb_norm = np.linalg.norm(embedding)
    if emb_norm == 0:
        emb_norm = 1.0
    norm_emb = embedding / emb_norm

    # 1. Genre similarity
    g_norms = np.linalg.norm(g_centroids, axis=1, keepdims=True)
    g_safe_norms = np.where(g_norms == 0, 1.0, g_norms)
    norm_g_centroids = g_centroids / g_safe_norms
    g_sims = np.dot(norm_emb, norm_g_centroids.T)[0]

    # 2. Territory similarity
    t_norms = np.linalg.norm(t_centroids, axis=1, keepdims=True)
    t_safe_norms = np.where(t_norms == 0, 1.0, t_norms)
    norm_t_centroids = t_centroids / t_safe_norms
    t_sims = np.dot(norm_emb, norm_t_centroids.T)[0]

    # Build sorted genre scores list
    genres = g_meta["genres"]
    genre_scores = []
    for i in range(len(genres)):
        val = float(g_sims[i])
        if val != val:
            val = 0.0
        val = max(-1.0, min(1.0, val))
        genre_scores.append({
            "genre": genres[i],
            "score": val,
        })
    genre_scores.sort(key=lambda x: x["score"], reverse=True)

    # Build sorted territory scores list
    territories = t_meta["territories"]
    territory_scores = []
    for i in range(len(territories)):
        val = float(t_sims[i])
        if val != val:
            val = 0.0
        val = max(-1.0, min(1.0, val))
        territory_scores.append({
            "territory": territories[i],
            "score": val,
        })
    territory_scores.sort(key=lambda x: x["score"], reverse=True)

    return {
        "genre": genre_scores[0]["genre"],
        "genre_confidence": genre_scores[0]["score"],
        "genre_scores": genre_scores,
        "territory": territory_scores[0]["territory"],
        "territory_confidence": territory_scores[0]["score"],
        "territory_scores": territory_scores,
    }
