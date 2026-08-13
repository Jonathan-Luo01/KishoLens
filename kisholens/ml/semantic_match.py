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


def classify_territory(
    text: str,
    features: Optional[dict] = None,
    world_genre: str = "Mystery",
    lang: str = "en",
    data_dir: str = DEFAULT_DATA_DIR,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict:
    """
    Classifies Territory (Classic Literature Territory vs Web Novel Territory) using
    a hybrid model: 60% Stylistic Syntax & Pacing, 30% 384D Territory Corpus Embedding,
    and 10% Genre Affinity Prior.
    """
    from kisholens.ml.build_centroids import GENRE_TERRITORIES, embed_texts
    from kisholens.ml.features import extract_english_features

    if features is None:
        if lang == "en":
            features = extract_english_features(text)
        else:
            features = {}

    # 1. Territory Corpus Embedding Signal (45%)
    _, _, t_centroids, t_meta = _load_with_cache(data_dir)
    if t_centroids is not None and len(t_centroids) >= 2:
        emb = embed_texts([text], model_name=model_name)[0]
        c_classic = t_centroids[0]
        c_web = t_centroids[1]

        norm_emb = float(np.linalg.norm(emb))
        norm_c = float(np.linalg.norm(c_classic))
        norm_w = float(np.linalg.norm(c_web))

        sim_c = float(np.dot(emb, c_classic) / (norm_emb * norm_c)) if (norm_emb > 0 and norm_c > 0) else 0.0
        sim_w = float(np.dot(emb, c_web) / (norm_emb * norm_w)) if (norm_emb > 0 and norm_w > 0) else 0.0

        exp_c = float(np.exp(sim_c * 8.0))
        exp_w = float(np.exp(sim_w * 8.0))
        denom = exp_c + exp_w
        emb_classic_prob = exp_c / denom if denom > 0 else 0.5
        emb_web_prob = 1.0 - emb_classic_prob
    else:
        emb_classic_prob = 0.5
        emb_web_prob = 0.5

    # 2. Stylistic Syntax & Web Tropes Signal (40%)
    sl = float(features.get("avg_sentence_len", 15.0) or 15.0)
    para_density = float(features.get("avg_sentences_per_paragraph", 2.0) or 2.0)
    depth = float(features.get("dep_tree_depth", 5.0) or 5.0)

    # Calibrated to corpus distributions: Classic ~21w, para ~7, depth ~6.4; Web Novel ~13w, para ~2, depth ~5.0
    s_sl = max(0.0, min(1.0, (sl - 11.0) / 10.0))
    s_para = max(0.0, min(1.0, (para_density - 1.5) / 4.5))
    s_depth = max(0.0, min(1.0, (depth - 4.5) / 2.2))

    base_style_classic = 0.35 * s_para + 0.35 * s_depth + 0.30 * s_sl

    # Detect distinctive web fiction / serialized novel tropes and lexical patterns
    text_lower = text.lower()
    web_markers = [
        "[", "]", "level ", "hp", "mp", "mana", "system", "reincarnat", "transmigrat",
        "cheat skill", "stats", "cultivat", "dao", "sect", "dungeon", "truck-kun",
        "archmage", "quest", "grimoire", "holographic"
    ]
    marker_hits = sum(1 for m in web_markers if m in text_lower)
    web_marker_discount = min(0.45, marker_hits * 0.12)

    style_classic_prob = max(0.05, min(0.95, base_style_classic - web_marker_discount))
    style_web_prob = 1.0 - style_classic_prob

    # 3. Genre Affinity Prior Signal (15%)
    prior_territory = GENRE_TERRITORIES.get(world_genre, "Classic Literature Territory")
    genre_classic_prob = 1.0 if prior_territory == "Classic Literature Territory" else 0.0
    genre_web_prob = 1.0 - genre_classic_prob

    # Composite Calculation: 45% Embedding + 40% Style + 15% Genre Prior
    final_classic = float(0.45 * emb_classic_prob + 0.40 * style_classic_prob + 0.15 * genre_classic_prob)
    final_web = float(1.0 - final_classic)

    top_t = "Classic Literature Territory" if final_classic >= final_web else "Web Novel Territory"
    top_conf = max(final_classic, final_web)

    scores = [
        {"territory": "Classic Literature Territory", "score": round(final_classic, 4), "raw_score": round(final_classic, 4)},
        {"territory": "Web Novel Territory", "score": round(final_web, 4), "raw_score": round(final_web, 4)},
    ] if final_classic >= final_web else [
        {"territory": "Web Novel Territory", "score": round(final_web, 4), "raw_score": round(final_web, 4)},
        {"territory": "Classic Literature Territory", "score": round(final_classic, 4), "raw_score": round(final_classic, 4)},
    ]

    return {
        "territory": top_t,
        "territory_confidence": round(top_conf, 4),
        "territory_scores": scores,
        "territory_breakdown": {
            "stylistic": {"Classic Literature Territory": round(style_classic_prob, 4), "Web Novel Territory": round(style_web_prob, 4)},
            "embedding": {"Classic Literature Territory": round(emb_classic_prob, 4), "Web Novel Territory": round(emb_web_prob, 4)},
            "genre_prior": {"Classic Literature Territory": round(genre_classic_prob, 4), "Web Novel Territory": round(genre_web_prob, 4)},
        },
    }


def match_semantic(
    text: str,
    title: Optional[str] = None,
    synopsis: Optional[str] = None,
    features: Optional[dict] = None,
    model_name: str = "all-MiniLM-L6-v2",
    data_dir: str = DEFAULT_DATA_DIR,
    use_regex_boost: bool = True,
) -> Optional[dict]:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 3:
        ch1 = paragraphs[0] if len(paragraphs) > 0 else text
        ch10 = paragraphs[1] if len(paragraphs) > 1 else None
        ch20 = paragraphs[2] if len(paragraphs) > 2 else None
    else:
        n = len(paragraphs)
        ch1 = "\n\n".join(paragraphs[:n // 3])
        ch10 = "\n\n".join(paragraphs[n // 3: 2 * n // 3])
        ch20 = "\n\n".join(paragraphs[2 * n // 3:])

    taxonomy = analyze_prose(synopsis, ch1, ch10, ch20, title=title, data_dir=data_dir, use_regex_boost=use_regex_boost)
    if not taxonomy:
        return None

    world_primary = taxonomy["world_setting"]["primary"]
    world_score = taxonomy["world_setting"]["score"]

    genre_scores = taxonomy.get("genre_scores", [{"genre": world_primary, "score": world_score, "raw_score": world_score}])

    # Prose-driven multi-signal territory classification
    t_res = classify_territory(
        text=text,
        features=features,
        world_genre=world_primary,
        data_dir=data_dir,
        model_name=model_name,
    )

    return {
        "genre": world_primary,
        "genre_confidence": world_score,
        "territory": t_res["territory"],
        "territory_confidence": t_res["territory_confidence"],
        "genre_scores": genre_scores,
        "territory_scores": t_res["territory_scores"],
        "territory_breakdown": t_res.get("territory_breakdown", {}),
        "taxonomy": taxonomy,
    }



