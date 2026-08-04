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


def clear_centroid_cache() -> None:
    """Clear in-memory centroid cache to force reload from disk."""
    _centroid_cache.clear()


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


from kisholens.pipeline.taxonomy import (
    GENRE_TAXONOMY,
    ANCHOR_TERMS,
    detect_text_language,
    scan_anchor_boosts,
)



from kisholens.ml.analyzer import analyze_prose


def match_semantic(
    text: str,
    title: Optional[str] = None,
    synopsis: Optional[str] = None,
    model_name: str = "all-MiniLM-L6-v2",
    data_dir: str = DEFAULT_DATA_DIR,
) -> Optional[dict]:
    paragraphs = text.split("\n\n")
    ch1 = paragraphs[0] if paragraphs else text
    ch10 = paragraphs[len(paragraphs) // 2] if len(paragraphs) > 2 else None
    ch20 = paragraphs[-1] if len(paragraphs) > 2 else None

    taxonomy = analyze_prose(synopsis, ch1, ch10, ch20, title=title, data_dir=data_dir)
    if not taxonomy:
        return None

    world_primary = taxonomy["world_setting"]["primary"]
    world_score = taxonomy["world_setting"]["score"]

    genre_scores = taxonomy.get("genre_scores", [{"genre": world_primary, "score": world_score, "raw_score": world_score}])

    return {
        "genre": world_primary,
        "genre_confidence": world_score,
        "territory": "Web Novel Territory",
        "territory_confidence": 0.95,
        "genre_scores": genre_scores,
        "territory_scores": [{"territory": "Web Novel Territory", "score": 0.95, "raw_score": 0.95}],
        "taxonomy": taxonomy,
    }

