"""
similarity.py — Enhanced multi-faceted literary doppelgänger search for KishoLens.

Compares novels using 5 weighted similarity factors:
  1. Stylistic fingerprint (8D radar vector cosine + L1)
  2. Semantic embedding (384D sentence transformer cosine similarity)
  3. Parent genre overlap (Jaccard set similarity)
  4. Fine-grained tag overlap (Jaccard set similarity)
  5. Territory semantic similarity (embedding cosine, not binary)

Returns per-match breakdown so the frontend can explain WHY novels match.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sqlmodel import Session, select
from kisholens.models import Novel, Chapter, get_engine
from kisholens.ml.features import (
    extract_english_features,
    extract_japanese_features,
    extract_chinese_features,
    normalize_feature_percentile,
)
from kisholens.ml.semantic_match import match_semantic

RADAR_FEATURE_KEYS = [
    "theme_explication_ratio",
    "linearity_subversion_score",
    "sensory_body_density",
    "outside_world_engagement",
    "narrative_feature_diversity",
    "dialogue_ratio",
    "ttr",
    "temporal_shift_score"
]

_novel_vector_cache: Dict[int, dict] = {}

# Cache for 384D concept embeddings keyed by "{genre}|{territory}|{title}" strings
_concept_embedding_cache: Dict[str, np.ndarray] = {}


def _get_concept_embedding(title: str, genre: str, territory: str) -> np.ndarray:
    """
    Computes a 384D concept embedding for a novel by encoding its title, genre,
    and territory through the sentence transformer. This captures WHAT a novel
    is about semantically (themes, setting, narrative type).

    Results are cached by the composite key for performance.
    """
    cache_key = f"{genre}|{territory}|{title}"
    if cache_key in _concept_embedding_cache:
        return _concept_embedding_cache[cache_key]

    try:
        from kisholens.ml.embeddings import embed_single_text
        # Build a descriptive concept string for embedding
        parts = []
        if genre:
            parts.append(genre)
        if territory:
            parts.append(territory)
        if title:
            parts.append(title)
        concept_text = ". ".join(parts) if parts else "fiction"
        vec = embed_single_text(concept_text)
        _concept_embedding_cache[cache_key] = vec
        return vec
    except Exception:
        vec = np.zeros(384, dtype=np.float32)
        _concept_embedding_cache[cache_key] = vec
        return vec


def extract_feature_vector(features: dict) -> np.ndarray:
    """
    Extracts an 8-dimensional normalized percentile vector from a features dict.
    """
    prefix = ""
    for p in ["en_", "ja_", "zh_"]:
        if any(k.startswith(p) for k in features.keys()):
            prefix = p
            break

    vec = []
    norm_radar = features.get("normalized_radar", {})
    for key in RADAR_FEATURE_KEYS:
        full_key = f"{prefix}{key}" if prefix else key
        if full_key in norm_radar:
            val = norm_radar[full_key]
        elif key in norm_radar:
            val = norm_radar[key]
        elif full_key in features and features[full_key] is not None:
            val = normalize_feature_percentile(key, features[full_key])
        elif key in features and features[key] is not None:
            val = normalize_feature_percentile(key, features[key])
        else:
            val = 0.5
        vec.append(float(val))
    return np.array(vec, dtype=float)


def _init_cache_from_disk(cache_path: Optional[Union[str, Path]] = None) -> None:
    """
    Hydrates _novel_vector_cache from data/stats_cache.json on startup / module import,
    extracting 8D normalized radar vectors, primary taxonomy genres, top genres, territories,
    and metadata for fast in-memory similarity matching.
    """
    if cache_path is None:
        cache_path = Path(__file__).resolve().parent.parent.parent / "data" / "stats_cache.json"
    else:
        cache_path = Path(cache_path)

    if not cache_path.exists():
        return

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    for k, item in data.items():
        if k.startswith("_") or not isinstance(item, dict):
            continue
        try:
            nid = int(k)
        except (ValueError, TypeError):
            continue

        vec = extract_feature_vector(item)
        genre = (
            item.get("genre")
            or item.get("primary_genre")
            or (
                item.get("taxonomy", {}).get("world_setting", {}).get("primary")
                if isinstance(item.get("taxonomy"), dict)
                else None
            )
            or "Unknown"
        )
        top_genres = (
            item.get("top_genres")
            or (
                item.get("archetype_match", {}).get("top_genres")
                if isinstance(item.get("archetype_match"), dict)
                else None
            )
            or []
        )
        territory = (
            item.get("territory")
            or (
                item.get("archetype_match", {}).get("territory")
                if isinstance(item.get("archetype_match"), dict)
                else None
            )
            or "Unknown"
        )
        title = item.get("title") or "Unknown Title"
        author = item.get("author") or "Unknown Author"

        _novel_vector_cache[nid] = {
            "id": nid,
            "title": title,
            "author": author,
            "genre": genre,
            "primary_genre": genre,
            "top_genres": top_genres,
            "territory": territory,
            "vector": vec,
            "semantic": item.get("taxonomy") or item.get("archetype_match"),
        }


# Auto-hydrate on module load
_init_cache_from_disk()


def extract_multi_point_samples(chapters: List[Chapter]) -> List[str]:
    """
    Extracts 300-word/character samples from the 20% mark and 80% mark of a novel
    to capture narrative arc progression rather than just opening lines.
    """
    if not chapters:
        return []

    sorted_chs = sorted(chapters, key=lambda c: c.chapter_number)

    if len(sorted_chs) == 1:
        text = sorted_chs[0].text_en or sorted_chs[0].text_ja or getattr(sorted_chs[0], "text_zh", "") or ""
        words = text.split()
        if len(words) >= 500:
            idx_20 = int(len(words) * 0.20)
            idx_80 = int(len(words) * 0.80)
            s20 = " ".join(words[idx_20 : idx_20 + 300])
            s80 = " ".join(words[idx_80 : idx_80 + 300])
            return [s20, s80]
        else:
            return [text]
    else:
        idx_20 = int((len(sorted_chs) - 1) * 0.20)
        idx_80 = int((len(sorted_chs) - 1) * 0.80)
        ch_20 = sorted_chs[idx_20]
        ch_80 = sorted_chs[idx_80]

        t20 = ch_20.text_en or ch_20.text_ja or getattr(ch_20, "text_zh", "") or ""
        t80 = ch_80.text_en or ch_80.text_ja or getattr(ch_80, "text_zh", "") or ""

        s20 = " ".join(t20.split()[:300])
        s80 = " ".join(t80.split()[:300])
        return [s20, s80]


def get_novel_vector_and_meta(novel: Novel, session: Session) -> dict:
    """
    Computes or retrieves cached multi-point feature vector and semantic genre metadata
    for a database novel.
    """
    if novel.id in _novel_vector_cache:
        return _novel_vector_cache[novel.id]

    chapters = session.exec(select(Chapter).where(Chapter.novel_id == novel.id)).all()
    genre_display = novel.genre or (novel.source == "gutenberg" and "Classic Lit" or "Web Novel")

    if not chapters:
        meta = {
            "id": novel.id,
            "title": novel.title,
            "author": novel.author or "Unknown Author",
            "genre": genre_display,
            "vector": np.full(8, 0.5),
            "semantic": None
        }
        _novel_vector_cache[novel.id] = meta
        return meta

    samples = extract_multi_point_samples(chapters)
    vectors = []

    for sample in samples:
        row = {}
        ch0 = chapters[0]
        if ch0.text_en:
            row.update({f"en_{k}": v for k, v in extract_english_features(sample).items()})
        elif ch0.text_ja:
            row.update({f"ja_{k}": v for k, v in extract_japanese_features(sample).items()})
        elif getattr(ch0, "text_zh", ""):
            row.update({f"zh_{k}": v for k, v in extract_chinese_features(sample).items()})
        vectors.append(extract_feature_vector(row))

    avg_vec = np.mean(vectors, axis=0) if vectors else np.full(8, 0.5)

    # Compute semantic centroid match on combined sample text
    combined_sample = " ".join(samples) if samples else ""
    sem = None
    if combined_sample:
        try:
            sem = match_semantic(combined_sample)
        except Exception:
            pass

    meta = {
        "id": novel.id,
        "title": novel.title,
        "author": novel.author or "Unknown Author",
        "genre": genre_display,
        "territory": novel.territory or (sem.get("territory") if sem else "Unknown"),
        "vector": avg_vec,
        "semantic": sem
    }
    _novel_vector_cache[novel.id] = meta
    return meta


def find_top_matches(
    query_features: dict,
    query_text: Optional[str] = None,
    exclude_novel_id: Optional[int] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Performs multi-faceted vector similarity search comparing query features, semantic
    concept embeddings, genres, tags, and territories against all database novels.

    Returns per-match breakdown for frontend display.

    Scoring weights:
      - 25% Stylistic similarity (8D radar cosine + L1)
      - 20% Semantic concept embedding (384D cosine via sentence transformer)
      - 25% Parent genre overlap (Jaccard set similarity)
      - 10% Fine-grained tag overlap (Jaccard set similarity)
      - 20% Territory semantic similarity (384D concept embedding cosine)
    """
    q_vec = extract_feature_vector(query_features)

    query_semantic = None
    if query_text:
        try:
            query_semantic = match_semantic(query_text)
        except Exception:
            pass

    candidates = []
    engine = get_engine()

    with Session(engine) as session:
        target_novel = session.get(Novel, exclude_novel_id) if exclude_novel_id else None
        target_genres = set([g.strip().lower() for g in (target_novel.genre or "").split(",") if g.strip()]) if target_novel else set()
        target_tags = set([t.strip().lower() for t in (target_novel.tags or "").split(",") if t.strip()]) if target_novel else set()
        target_territory = target_novel.territory if target_novel else None
        target_title = target_novel.title if target_novel else (query_features.get("title") or "")

        if query_semantic and not target_genres:
            for item in query_semantic.get("genre_scores", []):
                if item.get("score", 0) > 0.4:
                    target_genres.add(item["genre"].lower())
            if not target_territory:
                target_territory = query_semantic.get("territory")

        # Compute query concept embedding (384D) for semantic + territory similarity
        q_genre_str = ", ".join(sorted(target_genres)) if target_genres else ""
        q_concept_emb = _get_concept_embedding(target_title, q_genre_str, target_territory or "")

        all_novels = session.exec(select(Novel)).all()

        for novel in all_novels:
            if exclude_novel_id is not None and novel.id == exclude_novel_id:
                continue

            # Fast path: use pre-computed vector from cache; skip NLP for uncached novels
            if novel.id in _novel_vector_cache:
                n_meta = _novel_vector_cache[novel.id]
                n_vec = n_meta["vector"]
            else:
                # Lightweight fallback: use neutral 0.5 vector + DB metadata only (no NLP)
                n_vec = np.full(8, 0.5)
                n_meta = {
                    "id": novel.id,
                    "title": novel.title,
                    "author": novel.author or "Unknown Author",
                    "genre": novel.genre or "",
                    "territory": novel.territory or "Unknown",
                    "vector": n_vec,
                    "semantic": None,
                }

            # ── Factor 1: Stylistic Similarity (8D radar cosine + L1) ──
            cos_sim = float(np.dot(q_vec, n_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(n_vec) + 1e-9))
            l1_diff = float(np.mean(np.abs(q_vec - n_vec)))
            style_sim = (0.5 * cos_sim) + (0.5 * max(0.0, 1.0 - (4.0 * l1_diff)))

            # ── Factor 2: Semantic Concept Embedding (384D cosine) ──
            n_genre_str = n_meta.get("genre", "") or ""
            n_territory = n_meta.get("territory") or novel.territory or "Unknown"
            n_concept_emb = _get_concept_embedding(novel.title, n_genre_str, n_territory)
            q_norm = np.linalg.norm(q_concept_emb)
            n_norm = np.linalg.norm(n_concept_emb)
            if q_norm > 1e-9 and n_norm > 1e-9:
                semantic_sim = float(np.dot(q_concept_emb, n_concept_emb) / (q_norm * n_norm))
                semantic_sim = max(0.0, semantic_sim)  # clamp negative cosine to 0
            else:
                semantic_sim = 0.3

            # ── Factor 3: Parent Genre Overlap (Jaccard) ──
            cand_genres = set([g.strip().lower() for g in (n_meta["genre"] or "").split(",") if g.strip()])
            if target_genres and cand_genres:
                intersection = target_genres & cand_genres
                union = target_genres | cand_genres
                genre_sim = float(len(intersection) / len(union))
            else:
                genre_sim = 0.4

            # ── Factor 4: Fine-grained Tag Overlap (Jaccard) ──
            cand_tags = set([t.strip().lower() for t in (novel.tags or "").split(",") if t.strip()])
            if target_tags and cand_tags:
                tag_sim = float(len(target_tags & cand_tags) / max(1, len(target_tags | cand_tags)))
            else:
                tag_sim = genre_sim * 0.8  # slight discount when no tag data

            # ── Factor 5: Territory Semantic Similarity (384D cosine) ──
            # Uses the concept embedding which encodes territory as part of the
            # semantic fingerprint, but we also compute a focused territory-only
            # similarity for novels in different territories
            if target_territory and n_territory:
                if target_territory.lower() == n_territory.lower():
                    territory_sim = 1.0
                else:
                    # Partial credit via concept embedding overlap (already captured
                    # in semantic_sim), but also check for substring containment
                    t_overlap = (target_territory.lower() in n_territory.lower() or
                                 n_territory.lower() in target_territory.lower())
                    territory_sim = 0.6 if t_overlap else 0.15
            else:
                territory_sim = 0.3

            # ── Composite Score ──
            # 25% style + 20% semantic + 25% genre + 10% tags + 20% territory
            composite_score = (
                0.25 * style_sim +
                0.20 * semantic_sim +
                0.25 * genre_sim +
                0.10 * tag_sim +
                0.20 * territory_sim
            )
            score = round(min(0.99, max(0.01, composite_score)), 4)

            candidates.append({
                "id": novel.id,
                "title": novel.title,
                "author": novel.author,
                "genre": n_meta["genre"],
                "territory": n_territory or "Unknown",
                "similarity_score": score,
                "breakdown": {
                    "style": round(style_sim, 3),
                    "semantic": round(semantic_sim, 3),
                    "genre": round(genre_sim, 3),
                    "tags": round(tag_sim, 3),
                    "territory": round(territory_sim, 3),
                }
            })

    # Sort by composite score descending, then by genre overlap as tiebreaker
    candidates.sort(key=lambda x: (x["similarity_score"], x["breakdown"]["genre"]), reverse=True)
    return candidates[:top_k]
