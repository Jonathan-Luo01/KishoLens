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


import re

ANCHOR_TERMS: dict[str, list[str]] = {
    "Fantasy": [
        r"\barchmage\b", r"\bwizard\b", r"\bsorcerer\b", r"\belf\b", r"\belven\b",
        r"\bdragon\b", r"\bmagic\b", r"\bspell\b", r"\bspire\b", r"\bgoblin\b", r"\borc\b",
        r"\bkingdom\b", r"\bpaladin\b", r"\bhigh fantasy\b", r"\bmonolith\b", r"\bwyrm\b",
        r"\bmana\b", r"\brealm\b", r"\barcane\b"
    ],
    "Isekai": [
        r"\breincarnat", r"\btransmigrat", r"\breborn\b", r"\btruck-kun\b",
        r"\banother world\b", r"\bsummoned\b", r"\botome\b", r"\bvillainess\b",
        r"\botherworld\b", r"\bregressor\b", r"\bsecond chance\b"
    ],
    "Cultivation": [
        r"\bdantian\b", r"\bqi\b", r"\btribulation\b", r"\bsect\b",
        r"\bimmortal\b", r"\bdao\b", r"\bxianxia\b", r"\bwuxia\b",
        r"\bbreakthrough\b", r"\bcultivat", r"\bpill refining\b", r"\bspirit herb\b",
        r"\bmeridian\b", r"\bgolden core\b", r"\bnascent soul\b"
    ],
    "Progression Fantasy": [
        r"\bstatus window\b", r"\bstat point", r"\blevel up\b", r"\bdungeon\b",
        r"\bsystem notification\b", r"\blitrpg\b", r"\bstat screen\b",
        r"\bleveling\b", r"\bexperience points\b", r"\bexp\b", r"\bclass rank\b",
        r"\bweak to strong\b", r"\blevel system\b", r"\baccelerated growth\b",
        r"\bvrmmo\b", r"\bvrmmorpg\b", r"\blevel \d+\b", r"\bdivine stats\b"
    ],
    "Mystery": [
        r"\binspector\b", r"\bdetective\b", r"\bcyanide\b", r"\bpoison\b",
        r"\bwhodunit\b", r"\bmurder victim\b", r"\bsuspect\b", r"\bforensic\b",
        r"\bcrime scene\b", r"\bhomicide\b", r"\bwatson\b", r"\bholmes\b",
        r"\bclue\b", r"\bcase\b", r"\binterrogat"
    ],
    "Sci-Fi": [
        r"\bspaceship\b", r"\bstarship\b", r"\bcybernetic\b", r"\bandroid\b",
        r"\bwarp drive\b", r"\balien\b", r"\bgalaxy\b", r"\bquantum\b",
        r"\bcyberpunk\b", r"\bteleport\b", r"\bspacecraft\b", r"\binterstellar\b",
        r"\bsci-fi\b", r"\bscience fiction\b"
    ],
    "Horror": [
        r"\bhaunted\b", r"\bdemonic\b", r"\bghost\b", r"\bspecter\b",
        r"\bmacabre\b", r"\bpossession\b", r"\beeldritch\b", r"\bnightmare\b",
        r"\bterrifying\b", r"\bcreepy\b", r"\bcorpse\b", r"\bvictim\b"
    ],
    "Romance": [
        r"\bblush", r"\bheartbeat\b", r"\bconfession\b", r"\bfluster",
        r"\bfirst kiss\b", r"\bsecret crush\b", r"\belope\b", r"\btrue love\b"
    ],
    "Action / Adventure": [
        r"\bswordfight\b", r"\bbattlefield\b", r"\bambush\b", r"\bexpedition\b",
        r"\bquest\b", r"\bcombat\b", r"\bblade\b", r"\bwarrior\b", r"\berupted\b",
        r"\bunleashed\b", r"\broar\b"
    ],
    "Drama": [
        r"\btragedy\b", r"\bbetrayal\b", r"\bconflict\b", r"\bfamily feud\b",
        r"\btearful\b", r"\bsorrow\b", r"\bheartbreak\b", r"\bvictim\b"
    ],
    "Comedy": [
        r"\bhilarious\b", r"\blaugh\b", r"\babsurd\b", r"\bprank\b",
        r"\bchuckle\b", r"\bparody\b", r"\bsarcastic\b"
    ],
    "Slice of Life": [
        r"\bcozy\b", r"\bcafe\b", r"\btea shop\b", r"\beveryday\b",
        r"\bpeaceful\b", r"\bneighborhood\b", r"\bschool life\b"
    ],
    "Historical": [
        r"\bvictorian\b", r"\bmedieval\b", r"\bdynasty\b", r"\bempire\b",
        r"\bmonarch\b", r"\bking\b", r"\bqueen\b", r"\bduke\b"
    ],
    "Supernatural": [
        r"\bparanormal\b", r"\bspirit\b", r"\bphantom\b", r"\bcurse\b",
        r"\bexorcist\b", r"\boccult\b", r"\barcane\b", r"\bmana\b", r"\bcrystal orb\b"
    ]
}


def scan_anchor_boosts(text: str) -> dict[str, float]:
    """
    Scan text for high-confidence macro-genre anchor terms and return boost multipliers.
    """
    low_text = text.lower()
    boosts: dict[str, float] = {}
    for genre, patterns in ANCHOR_TERMS.items():
        matches = set()
        for pat in patterns:
            if re.search(pat, low_text):
                matches.add(pat)
        if matches:
            boosts[genre] = min(0.30, 0.10 * len(matches))
    return boosts


def match_semantic(
    text: str,
    title: Optional[str] = None,
    synopsis: Optional[str] = None,
    model_name: str = "all-MiniLM-L6-v2",
    data_dir: str = DEFAULT_DATA_DIR,
) -> Optional[dict]:
    g_centroids, g_meta, t_centroids, t_meta = _load_with_cache(data_dir)
    if g_centroids is None or g_meta is None or t_centroids is None or t_meta is None:
        return None

    # Embed the input text using 3-window weighted pooling
    embedding = embed_texts([text], model_name=model_name)  # (1, 384)

    # Cosine similarity using pure numpy:
    emb_norm = np.linalg.norm(embedding)
    if emb_norm == 0:
        emb_norm = 1.0
    norm_emb = embedding / emb_norm

    # 1. Genre similarity with common-space mean centroid subtraction
    g_norms = np.linalg.norm(g_centroids, axis=1, keepdims=True)
    g_safe_norms = np.where(g_norms == 0, 1.0, g_norms)
    norm_g_centroids = g_centroids / g_safe_norms

    # Global mean centroid vector across all 17 genre centroids in g_centroids
    g_mean = norm_g_centroids.mean(axis=0, keepdims=True)
    g_sub = norm_g_centroids - g_mean

    # Compute dot product against centered genre centroids
    g_sims = np.dot(norm_emb, g_sub.T)[0]

    # Baseline noise subtraction across 17 genre similarities so cross-genre noise drops below 0
    g_sims = g_sims - np.mean(g_sims)

    # 2. Territory similarity
    t_norms = np.linalg.norm(t_centroids, axis=1, keepdims=True)
    t_safe_norms = np.where(t_norms == 0, 1.0, t_norms)
    norm_t_centroids = t_centroids / t_safe_norms
    t_sims = np.dot(norm_emb, norm_t_centroids.T)[0]

    # Scan anchor boosts across title, synopsis, and text
    full_scan_text = f"{title or ''} {synopsis or ''} {text}".strip()
    anchor_boosts = scan_anchor_boosts(full_scan_text)

    # Build sorted genre scores list
    genres = g_meta["genres"]
    genre_scores = []
    for i in range(len(genres)):
        g_name = genres[i]
        val = float(g_sims[i])
        if val != val:
            val = 0.0
        val = val + anchor_boosts.get(g_name, 0.0)
        val = max(-1.0, min(1.0, val))
        genre_scores.append({
            "genre": g_name,
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
