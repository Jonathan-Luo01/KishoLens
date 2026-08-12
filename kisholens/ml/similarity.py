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
import re
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


def _get_concept_embedding(
    genre: str = "",
    territory: str = "",
    title: str = "",
    author: str = ""
) -> np.ndarray:
    """
    Computes a 384D concept embedding for a novel by encoding its genre
    and territory through the sentence transformer. This captures WHAT a novel
    is about semantically (themes, setting, narrative type).

    Results are cached by the composite key for high performance.
    """
    cache_key = f"{genre.strip()}|{territory.strip()}"
    if cache_key in _concept_embedding_cache:
        return _concept_embedding_cache[cache_key]

    try:
        from kisholens.ml.embeddings import embed_single_text
        parts = []
        if genre and genre.strip():
            parts.append(genre.strip())
        if territory and territory.strip():
            parts.append(territory.strip())
        concept_text = ". ".join(parts) if parts else "fiction"
        vec = embed_single_text(concept_text)
        _concept_embedding_cache[cache_key] = vec
        return vec
    except Exception:
        vec = np.zeros(384, dtype=np.float32)
        _concept_embedding_cache[cache_key] = vec
        return vec


def _extract_genre_list(genres_val: Any) -> List[str]:
    """Helper to extract clean genre name strings from string, list of strings, or list of dicts."""
    names: List[str] = []
    if not genres_val:
        return names
    if isinstance(genres_val, str):
        for part in genres_val.split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in names:
                names.append(cleaned)
    elif isinstance(genres_val, list):
        for g in genres_val:
            if isinstance(g, dict) and "genre" in g:
                cleaned = str(g["genre"]).strip()
                if cleaned and cleaned not in names:
                    names.append(cleaned)
            elif isinstance(g, str):
                cleaned = g.strip()
                if cleaned and cleaned not in names:
                    names.append(cleaned)
    return names


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
        tags = item.get("tags") or ""

        _novel_vector_cache[nid] = {
            "id": nid,
            "title": title,
            "author": author,
            "genre": genre,
            "primary_genre": genre,
            "top_genres": top_genres,
            "territory": territory,
            "tags": tags,
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
            "primary_genre": genre_display,
            "top_genres": [genre_display],
            "territory": novel.territory or "Unknown",
            "tags": novel.tags or "",
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
        "primary_genre": genre_display,
        "top_genres": [genre_display],
        "territory": novel.territory or (sem.get("territory") if sem else "Unknown"),
        "tags": novel.tags or "",
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
      - 30% Stylistic similarity (8D radar cosine + L1)
      - 35% Genre similarity & primary genre affinity (Jaccard + primary match bonus)
      - 20% Semantic concept embedding (384D sentence transformer cosine similarity)
      -  5% Fine-grained tag overlap (Jaccard set similarity)
      - 10% Territory semantic similarity
    """
    if len(_novel_vector_cache) == 0:
        _init_cache_from_disk()

    q_vec = extract_feature_vector(query_features)

    query_semantic = None
    if query_text:
        try:
            query_semantic = match_semantic(query_text)
        except Exception:
            pass

    # Extract target metadata from DB / cache if exclude_novel_id is set
    target_novel_meta = _novel_vector_cache.get(exclude_novel_id) if exclude_novel_id else None
    target_title = (target_novel_meta.get("title") if target_novel_meta else query_features.get("title")) or ""
    raw_target_tags = (target_novel_meta.get("tags") if target_novel_meta else query_features.get("tags")) or ""
    target_tags = set(
        [t.strip().lower() for t in raw_target_tags.split(",") if t.strip()]
    ) if raw_target_tags else set()

    # Extract query primary genre and top genres
    q_primary_genre = None
    q_genres: List[str] = []

    if query_semantic:
        if isinstance(query_semantic.get("taxonomy"), dict):
            ws = query_semantic["taxonomy"].get("world_setting", {})
            if isinstance(ws, dict) and ws.get("primary"):
                q_primary_genre = ws["primary"]
        if not q_primary_genre:
            q_primary_genre = query_semantic.get("closest_trope") or query_semantic.get("genre")
        if not q_primary_genre and query_semantic.get("top_genres"):
            first_tg = query_semantic["top_genres"][0]
            q_primary_genre = first_tg["genre"] if isinstance(first_tg, dict) else str(first_tg)
        if not q_primary_genre and query_semantic.get("genre_scores"):
            q_primary_genre = query_semantic["genre_scores"][0].get("genre")

        for g_item in query_semantic.get("top_genres", []):
            g_name = g_item["genre"] if isinstance(g_item, dict) else str(g_item)
            if g_name and g_name not in q_genres:
                q_genres.append(g_name)
        for g_item in query_semantic.get("genre_scores", []):
            if isinstance(g_item, dict) and g_item.get("score", 0) > 0.4:
                g_name = g_item.get("genre")
                if g_name and g_name not in q_genres:
                    q_genres.append(g_name)

    if not q_primary_genre:
        if isinstance(query_features.get("archetype_match"), dict):
            am = query_features["archetype_match"]
            q_primary_genre = am.get("closest_trope") or am.get("genre")
            for g_item in am.get("top_genres", []):
                g_name = g_item["genre"] if isinstance(g_item, dict) else str(g_item)
                if g_name and g_name not in q_genres:
                    q_genres.append(g_name)
        if not q_primary_genre:
            q_primary_genre = query_features.get("primary_genre") or query_features.get("genre")

    if target_novel_meta and not q_primary_genre:
        q_primary_genre = target_novel_meta.get("primary_genre") or target_novel_meta.get("genre")
        for g in _extract_genre_list(target_novel_meta.get("top_genres")) + _extract_genre_list(target_novel_meta.get("genre")):
            if g not in q_genres:
                q_genres.append(g)

    if q_primary_genre and q_primary_genre not in q_genres:
        q_genres.insert(0, q_primary_genre)

    target_genres = set(g.lower().strip() for g in q_genres if g and g.strip())
    q_primary_genre_lower = q_primary_genre.lower().strip() if q_primary_genre else ""

    # Territory detection
    target_territory = None
    if target_novel_meta:
        target_territory = target_novel_meta.get("territory")
    elif query_features.get("territory"):
        target_territory = query_features.get("territory")
    elif query_text:
        try:
            from kisholens.ml.semantic_match import _load_with_cache, DEFAULT_DATA_DIR
            from kisholens.ml.embeddings import embed_single_text
            g_c, g_m, t_c, t_m = _load_with_cache(DEFAULT_DATA_DIR)
            if t_c is not None and t_m is not None:
                q_txt_emb = embed_single_text(query_text[:500])
                t_sims = np.dot(t_c, q_txt_emb) / (np.linalg.norm(t_c, axis=1) * np.linalg.norm(q_txt_emb) + 1e-9)
                target_territory = t_m["territories"][int(np.argmax(t_sims))]
        except Exception:
            target_territory = query_semantic.get("territory") if query_semantic else None

    # Query concept embedding (384D)
    try:
        q_genre_str = q_primary_genre or (", ".join(sorted(target_genres)) if target_genres else "")
        q_concept_emb = _get_concept_embedding(genre=q_genre_str, territory=target_territory or "")
    except Exception:
        q_concept_emb = np.zeros(384, dtype=np.float32)

    q_norm = float(np.linalg.norm(q_concept_emb))

    candidates: List[Dict[str, Any]] = []

    # Iterate cached novel metadata or DB items
    candidate_items = list(_novel_vector_cache.values())
    if not candidate_items:
        engine = get_engine()
        with Session(engine) as session:
            all_novels = session.exec(select(Novel)).all()
            for novel in all_novels:
                candidate_items.append(get_novel_vector_and_meta(novel, session))

    for n_meta in candidate_items:
        nid = n_meta["id"]
        if exclude_novel_id is not None and nid == exclude_novel_id:
            continue

        n_vec = n_meta.get("vector")
        if n_vec is None:
            n_vec = np.full(8, 0.5)

        # ── Factor 1: Stylistic Similarity (30%) ──
        cos_sim = float(np.dot(q_vec, n_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(n_vec) + 1e-9))
        l1_diff = float(np.mean(np.abs(q_vec - n_vec)))
        style_sim = float(np.clip((0.5 * cos_sim) + (0.5 * max(0.0, 1.0 - (4.0 * l1_diff))), 0.0, 1.0))

        # ── Factor 2: Genre Similarity & Primary Affinity (35%) ──
        cand_genre_names = (
            _extract_genre_list(n_meta.get("top_genres"))
            + _extract_genre_list(n_meta.get("primary_genre"))
            + _extract_genre_list(n_meta.get("genre"))
        )
        cand_genres = set(g.lower() for g in cand_genre_names if g)
        cand_primary = (n_meta.get("primary_genre") or n_meta.get("genre") or "").lower().strip()

        intersection = target_genres & cand_genres
        union = target_genres | cand_genres
        jaccard = float(len(intersection) / max(1, len(union)))

        if q_primary_genre_lower and cand_primary and q_primary_genre_lower == cand_primary:
            genre_sim = 0.85 + 0.15 * jaccard
        elif q_primary_genre_lower and q_primary_genre_lower in cand_genres:
            genre_sim = 0.60 + 0.25 * jaccard
        elif target_genres and cand_genres:
            genre_sim = 0.20 * jaccard
        else:
            genre_sim = 0.40

        genre_sim = float(np.clip(genre_sim, 0.0, 1.0))

        # ── Factor 3: Semantic Concept Embedding (20%) ──
        n_territory = n_meta.get("territory") or "Unknown"
        title_str = n_meta.get("title", "")
        author_str = n_meta.get("author", "")

        n_concept_emb = _get_concept_embedding(genre=cand_primary, territory=n_territory)
        n_norm = float(np.linalg.norm(n_concept_emb))
        if q_norm > 1e-9 and n_norm > 1e-9:
            sem_raw = float(np.clip(np.dot(q_concept_emb, n_concept_emb) / (q_norm * n_norm), 0.0, 1.0))
        else:
            sem_raw = 0.40

        # Generic token overlap between candidate title/author words (>= 4 chars) and query text
        if query_text:
            q_lower = query_text.lower()
            title_tokens = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", title_str.lower()) if w not in {"with", "from", "that", "this", "into", "over", "about"}]
            author_tokens = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", author_str.lower()) if w not in {"unknown", "author"}]
            title_matches = sum(1 for w in set(title_tokens) if w in q_lower)
            author_matches = sum(1 for w in set(author_tokens) if w in q_lower)
            token_overlap = min(0.60, 0.30 * title_matches + 0.20 * author_matches)
            semantic_sim = float(np.clip(0.40 * sem_raw + token_overlap, 0.0, 1.0))
        else:
            semantic_sim = float(np.clip(sem_raw, 0.0, 1.0))

        # ── Factor 4: Fine-grained Tag Overlap (5%) ──
        cand_tags = set([t.strip().lower() for t in (n_meta.get("tags") or "").split(",") if t.strip()])
        if target_tags and cand_tags:
            tag_sim = float(len(target_tags & cand_tags) / max(1, len(target_tags | cand_tags)))
        else:
            tag_sim = float(genre_sim * 0.80)
        tag_sim = float(np.clip(tag_sim, 0.0, 1.0))

        # ── Factor 5: Territory Semantic Similarity (10%) ──
        if target_territory and target_territory != "Unknown" and n_territory != "Unknown":
            if target_territory.lower() == n_territory.lower():
                territory_sim = 1.0
            elif target_territory.lower() in n_territory.lower() or n_territory.lower() in target_territory.lower():
                territory_sim = 0.60
            else:
                territory_sim = 0.20
        else:
            territory_sim = 0.50

        # ── Composite Score (strictly normalized weights sum to 1.00) ──
        composite_score = (
            0.30 * style_sim
            + 0.35 * genre_sim
            + 0.20 * semantic_sim
            + 0.05 * tag_sim
            + 0.10 * territory_sim
        )
        score = round(min(0.99, max(0.01, composite_score)), 4)

        # Generate clean human-readable match reasons (no emojis)
        reasons = []
        if style_sim >= 0.88:
            reasons.append("Similar prose style & sentence structure")
        elif style_sim >= 0.75:
            reasons.append("Comparable sentence cadence")

        if q_primary_genre_lower and cand_primary and q_primary_genre_lower == cand_primary:
            cand_primary_clean = n_meta.get("primary_genre") or n_meta.get("genre")
            reasons.append(f"Matching primary archetype: {cand_primary_clean}")
        elif q_primary_genre_lower and q_primary_genre_lower in cand_genres:
            reasons.append(f"Shared genre: {q_primary_genre}")
        elif genre_sim >= 0.60:
            reasons.append("Strong genre overlap")

        if semantic_sim >= 0.80:
            reasons.append("Closely aligned plot premise & themes")
        elif semantic_sim >= 0.65:
            reasons.append("Thematic narrative overlap")

        if territory_sim >= 0.85 and n_territory and n_territory != "Unknown":
            if "classic" in n_territory.lower():
                reasons.append("Shared Classic Literature tradition")
            elif "web" in n_territory.lower():
                reasons.append("Shared Web Novel territory")

        if tag_sim >= 0.60:
            reasons.append("Overlapping narrative tropes")

        if not reasons:
            reasons.append("Overall stylistic and structural affinity")

        candidates.append({
            "id": nid,
            "title": title_str,
            "author": author_str,
            "genre": n_meta.get("genre", ""),
            "territory": n_territory or "Unknown",
            "similarity_score": score,
            "reasons": reasons,
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


