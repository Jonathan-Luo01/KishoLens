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
    GENRE_TERRITORIES,
)

# Module-level centroid cache: (data_dir -> (centroids, meta))
_centroid_cache: dict[str, tuple[np.ndarray, dict]] = {}

# Default centroid location (relative to project root)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)


def _load_with_cache(data_dir: str) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Load centroids from disk, caching per data_dir so repeated calls within
    the same process do not re-read files.
    """
    if data_dir not in _centroid_cache:
        centroids, meta = load_centroids_from_disk(data_dir)
        if centroids is not None:
            _centroid_cache[data_dir] = (centroids, meta)
        else:
            return None, None
    return _centroid_cache[data_dir]


def match_semantic(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
    data_dir: str = DEFAULT_DATA_DIR,
) -> Optional[dict]:
    """
    Embed `text` and compute cosine similarity against all genre centroids.

    Returns None if centroids have not been built (data files absent).

    Returns a dict:
    {
        "genre":      str,    # canonical genre with highest similarity
        "territory":  str,    # territory for that genre
        "confidence": float,  # highest cosine similarity score [0, 1]
        "scores": [           # ALL genres, sorted descending by score
            {"genre": str, "territory": str, "score": float},
            ...
        ]
    }
    """
    centroids, meta = _load_with_cache(data_dir)
    if centroids is None or meta is None:
        return None

    genres: list[str] = meta["genres"]
    territories: list[str] = meta["territories"]

    # Embed the input text
    embedding = embed_texts([text], model_name=model_name)  # (1, 384)

    # Cosine similarity: (1, 384) @ (384, G) → (1, G)
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(embedding, centroids)[0]  # (G,)

    # Build sorted scores list
    scores = [
        {
            "genre": genres[i],
            "territory": territories[i],
            "score": float(sims[i]),
        }
        for i in range(len(genres))
    ]
    scores.sort(key=lambda x: x["score"], reverse=True)

    best = scores[0]
    return {
        "genre": best["genre"],
        "territory": best["territory"],
        "confidence": best["score"],
        "scores": scores,
    }
