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
_concept_embedding_cache: Dict[str, np.ndarray] = {}

SIMILARITY_MODEL_VERSION = "v4_rich_story_embeddings"

# Cache for 384D concept embeddings keyed by "{genre}|{territory}|{title}" strings
_concept_embedding_cache: Dict[str, np.ndarray] = {}


def _get_concept_embedding(
    genre: str = "",
    territory: str = "",
    title: str = "",
    author: str = "",
    inciting_event: str = "",
    world_setting: str = "",
    narrative_plot: str = "",
    tags: str = ""
) -> np.ndarray:
    """
    Computes a 384D concept embedding for a novel by encoding its title,
    3-pillar taxonomy (catalyst, setting, plot), genre, tags, and territory
    through the sentence transformer. This captures WHAT a novel is about
    semantically (premise, conflict, themes).

    Results are cached by the composite key for high performance.
    """
    cache_key = f"{title.strip().lower()}|{genre.strip().lower()}|{inciting_event.strip().lower()}|{narrative_plot.strip().lower()}|{world_setting.strip().lower()}|{tags.strip().lower()}|{territory.strip().lower()}"
    if cache_key in _concept_embedding_cache:
        return _concept_embedding_cache[cache_key]

    try:
        from kisholens.ml.embeddings import embed_single_text
        parts = []
        if title and title.strip() and title.strip() != "Unknown Title":
            parts.append(title.strip())
        if world_setting and world_setting.strip():
            parts.append(f"Setting: {world_setting.strip()}")
        elif genre and genre.strip():
            parts.append(f"Genre: {genre.strip()}")
        if inciting_event and inciting_event.strip():
            parts.append(f"Catalyst: {inciting_event.strip()}")
        if narrative_plot and narrative_plot.strip():
            parts.append(f"Plot: {narrative_plot.strip()}")
        if tags and tags.strip():
            parts.append(f"Tropes: {tags.strip()}")
        if territory and territory.strip():
            parts.append(f"Tradition: {territory.strip()}")

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

        tax = item.get("taxonomy") if isinstance(item.get("taxonomy"), dict) else {}
        inciting = tax.get("inciting_event", {}).get("primary") if isinstance(tax.get("inciting_event"), dict) else ""
        world = tax.get("world_setting", {}).get("primary") if isinstance(tax.get("world_setting"), dict) else ""
        plot = tax.get("narrative_plot", {}).get("primary") if isinstance(tax.get("narrative_plot"), dict) else ""

        _novel_vector_cache[nid] = {
            "id": nid,
            "title": title,
            "author": author,
            "genre": genre,
            "primary_genre": genre,
            "top_genres": top_genres,
            "territory": territory,
            "tags": tags,
            "inciting_event": inciting,
            "world_setting": world,
            "narrative_plot": plot,
            "vector": vec,
            "raw_features": item,
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
        "raw_features": {},
        "semantic": sem
    }
    _novel_vector_cache[novel.id] = meta
    return meta



def _extract_metric_values(features: dict, vector: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Extracts the 5 core stylistic and structural metrics from feature dicts or radar vectors:
      1. dialogue_ratio (float, 0.0 - 1.0)
      2. avg_sentence_len (float, words/sentence e.g. 5.0 - 35.0)
      3. ttr (float, 0.0 - 1.0)
      4. sensory_body_density (float, 0.0 - 1.0)
      5. theme_explication_ratio (float, 0.0 - 10.0)
    """
    raw = features.get("raw_features") if isinstance(features.get("raw_features"), dict) else features

    # 1. Dialogue Ratio
    dlg = None
    for k in ["dialogue_ratio", "en_dialogue_ratio", "ja_dialogue_ratio", "zh_dialogue_ratio"]:
        if k in raw and raw[k] is not None:
            try:
                dlg = float(raw[k])
                break
            except (ValueError, TypeError):
                pass
    if dlg is None and vector is not None and len(vector) > 5:
        dlg = float(vector[5])
    if dlg is None:
        dlg = 0.50

    # 2. Average Sentence Length (Cadence)
    asl = None
    for k in [
        "avg_sentence_len", "avg_sentence_length",
        "en_avg_sentence_len", "en_avg_sentence_length",
        "ja_avg_sentence_len", "ja_avg_sentence_length",
        "zh_avg_sentence_len", "zh_avg_sentence_length"
    ]:
        if k in raw and raw[k] is not None:
            try:
                asl = float(raw[k])
                break
            except (ValueError, TypeError):
                pass
    if asl is None and vector is not None and len(vector) > 1:
        asl = 6.0 + float(vector[1]) * 18.0
    if asl is None:
        asl = 12.0

    # 3. Lexical Diversity (TTR)
    ttr = None
    for k in ["ttr", "en_ttr", "ja_ttr", "zh_ttr"]:
        if k in raw and raw[k] is not None:
            try:
                ttr = float(raw[k])
                break
            except (ValueError, TypeError):
                pass
    if ttr is None and vector is not None and len(vector) > 6:
        ttr = float(vector[6])
    if ttr is None:
        ttr = 0.50

    # 4. Visceral Imagery (Sensory Body Density)
    sbd = None
    for k in ["sensory_body_density", "en_sensory_body_density", "ja_sensory_body_density", "zh_sensory_body_density"]:
        if k in raw and raw[k] is not None:
            try:
                sbd = float(raw[k])
                break
            except (ValueError, TypeError):
                pass
    if sbd is None and vector is not None and len(vector) > 2:
        sbd = float(vector[2])
    if sbd is None:
        sbd = 0.40

    # 5. Thematic Explicitness (Theme Explication Ratio)
    theme = None
    for k in ["theme_explication_ratio", "en_theme_explication_ratio", "ja_theme_explication_ratio", "zh_theme_explication_ratio"]:
        if k in raw and raw[k] is not None:
            try:
                theme = float(raw[k])
                break
            except (ValueError, TypeError):
                pass
    if theme is None and vector is not None and len(vector) > 0:
        theme = float(vector[0]) * 5.0
    if theme is None:
        theme = 2.0

    return {
        "dialogue_ratio": dlg,
        "avg_sentence_len": asl,
        "ttr": ttr,
        "sensory_body_density": sbd,
        "theme_explication_ratio": theme,
    }


def _extract_catalyst(
    query_text: Optional[str],
    query_semantic: Optional[dict],
    query_features: dict,
    cand_meta: dict
) -> Optional[str]:
    combined_query = (query_text or "") + " " + str(query_features.get("genre", "")) + " " + str(query_features.get("tags", ""))
    cand_text = (
        str(cand_meta.get("title", "")) + " " +
        str(cand_meta.get("genre", "")) + " " +
        str(cand_meta.get("tags", ""))
    )

    inc_query = None
    if query_semantic and isinstance(query_semantic.get("taxonomy"), dict):
        inc = query_semantic["taxonomy"].get("inciting_event")
        if isinstance(inc, dict) and inc.get("primary"):
            inc_query = str(inc.get("primary"))

    cand_sem = cand_meta.get("semantic")
    inc_cand = None
    if cand_sem and isinstance(cand_sem.get("taxonomy"), dict):
        inc = cand_sem["taxonomy"].get("inciting_event")
        if isinstance(inc, dict) and inc.get("primary"):
            inc_cand = str(inc.get("primary"))

    q_lower = combined_query.lower()
    c_lower = cand_text.lower()

    if "reincarnat" in q_lower or "reincarnat" in c_lower or (inc_query and "reincarnat" in inc_query.lower()) or (inc_cand and "reincarnat" in inc_cand.lower()):
        return "Reincarnation"
    if "summon" in q_lower or "summon" in c_lower or (inc_query and "summon" in inc_query.lower()) or (inc_cand and "summon" in inc_cand.lower()):
        return "Summons"
    if "regress" in q_lower or "regress" in c_lower or (inc_query and "regress" in inc_query.lower()) or (inc_cand and "regress" in inc_cand.lower()):
        return "Regression"
    if "transmigrat" in q_lower or "transmigrat" in c_lower:
        return "Transmigration"
    if "system" in q_lower or "system" in c_lower or "status screen" in q_lower:
        return "System Awakening"
    if "murder" in q_lower or "investigat" in q_lower or "detective" in q_lower or "murder" in c_lower or "detective" in c_lower:
        return "Murder Investigation"
    if "revenge" in q_lower or "betray" in q_lower or "revenge" in c_lower or "betray" in c_lower:
        return "Betrayal"
    if "tournament" in q_lower or "tournament" in c_lower or "competition" in q_lower:
        return "Tournament"

    if inc_cand:
        parts = [p.strip() for p in inc_cand.split("&")]
        return parts[0] if parts else inc_cand

    if inc_query:
        parts = [p.strip() for p in inc_query.split("&")]
        return parts[0] if parts else inc_query

    return None


def _extract_setting(query_semantic: Optional[dict], cand_meta: dict) -> Optional[str]:
    cand_sem = cand_meta.get("semantic")
    if cand_sem and isinstance(cand_sem.get("taxonomy"), dict):
        ws = cand_sem["taxonomy"].get("world_setting")
        if isinstance(ws, dict) and ws.get("primary"):
            return str(ws["primary"])
    if query_semantic and isinstance(query_semantic.get("taxonomy"), dict):
        ws = query_semantic["taxonomy"].get("world_setting")
        if isinstance(ws, dict) and ws.get("primary"):
            return str(ws["primary"])
    cand_genre = cand_meta.get("primary_genre") or cand_meta.get("genre")
    if cand_genre and cand_genre not in ["Unknown", ""]:
        return str(cand_genre)
    return None


def _compute_metric_comparisons(
    q_metrics: Dict[str, float],
    c_metrics: Dict[str, float]
) -> List[Dict[str, str]]:
    """
    Computes 5 side-by-side metric comparison rows:
      - Dialogue Density
      - Sentence Cadence
      - Lexical Richness (TTR)
      - Visceral Somatic Imagery
      - Thematic Explicitness
    """
    q_dlg = q_metrics["dialogue_ratio"]
    c_dlg = c_metrics["dialogue_ratio"]
    dlg_match = max(0, min(100, int(round(100.0 - abs(q_dlg - c_dlg) * 200.0))))

    q_asl = q_metrics["avg_sentence_len"]
    c_asl = c_metrics["avg_sentence_len"]
    asl_match = max(0, min(100, int(round(100.0 - abs(q_asl - c_asl) * 10.0))))

    q_ttr = q_metrics["ttr"]
    c_ttr = c_metrics["ttr"]
    ttr_match = max(0, min(100, int(round(100.0 - abs(q_ttr - c_ttr) * 300.0))))

    q_sbd = q_metrics["sensory_body_density"]
    c_sbd = c_metrics["sensory_body_density"]
    sbd_match = max(0, min(100, int(round(100.0 - abs(q_sbd - c_sbd) * 200.0))))

    q_thm = q_metrics["theme_explication_ratio"]
    c_thm = c_metrics["theme_explication_ratio"]
    thm_match = max(0, min(100, int(round(100.0 - abs(q_thm - c_thm) * 20.0))))

    return [
        {
            "metric": "Dialogue Density",
            "query": f"{q_dlg * 100:.1f}%",
            "candidate": f"{c_dlg * 100:.1f}%",
            "match": f"{dlg_match}%",
        },
        {
            "metric": "Sentence Cadence",
            "query": f"{q_asl:.1f} w/s",
            "candidate": f"{c_asl:.1f} w/s",
            "match": f"{asl_match}%",
        },
        {
            "metric": "Lexical Richness (TTR)",
            "query": f"{q_ttr:.2f}",
            "candidate": f"{c_ttr:.2f}",
            "match": f"{ttr_match}%",
        },
        {
            "metric": "Visceral Somatic Imagery",
            "query": f"{q_sbd * 100:.1f}%",
            "candidate": f"{c_sbd * 100:.1f}%",
            "match": f"{sbd_match}%",
        },
        {
            "metric": "Thematic Explicitness",
            "query": f"{q_thm:.2f}",
            "candidate": f"{c_thm:.2f}",
            "match": f"{thm_match}%",
        },
    ]


def _compute_match_badges(
    q_metrics: Dict[str, float],
    c_metrics: Dict[str, float],
    query_text: Optional[str],
    query_semantic: Optional[dict],
    query_features: dict,
    cand_meta: dict,
    cand_primary_genre: str,
    style_sim: float,
    score: float
) -> List[Dict[str, str]]:
    badges: List[Dict[str, str]] = []

    # 1. Emerald: Primary archetype / genre badge
    if cand_primary_genre and cand_primary_genre != "Unknown":
        badges.append({
            "type": "trope",
            "label": "Archetype",
            "detail": cand_primary_genre,
            "tier": "emerald"
        })

    # 2. Amber: Catalyst / Setting badge
    catalyst = _extract_catalyst(query_text, query_semantic, query_features, cand_meta)
    if catalyst:
        badges.append({
            "type": "taxonomy",
            "label": "Catalyst",
            "detail": catalyst,
            "tier": "amber"
        })
    else:
        setting = _extract_setting(query_semantic, cand_meta)
        if setting and setting != cand_primary_genre:
            badges.append({
                "type": "taxonomy",
                "label": "Setting",
                "detail": setting,
                "tier": "amber"
            })

    # 3. Cyan: Dialogue overlap badge
    q_dlg = q_metrics["dialogue_ratio"]
    c_dlg = c_metrics["dialogue_ratio"]
    if abs(q_dlg - c_dlg) <= 0.15:
        badges.append({
            "type": "metric",
            "label": "Dialogue",
            "detail": f"{round(q_dlg * 100)}% ≈ {round(c_dlg * 100)}%",
            "tier": "cyan"
        })

    # 4. Purple: Sentence Cadence overlap badge
    q_asl = q_metrics["avg_sentence_len"]
    c_asl = c_metrics["avg_sentence_len"]
    if abs(q_asl - c_asl) <= 4.0:
        badges.append({
            "type": "metric",
            "label": "Cadence",
            "detail": f"{q_asl:.1f} ≈ {c_asl:.1f} w/s",
            "tier": "purple"
        })

    # 5. Cyan: Lexical Richness (TTR) badge if needed
    q_ttr = q_metrics["ttr"]
    c_ttr = c_metrics["ttr"]
    if len(badges) < 4 and abs(q_ttr - c_ttr) <= 0.08:
        badges.append({
            "type": "metric",
            "label": "Vocab",
            "detail": f"TTR {q_ttr:.2f} ≈ {c_ttr:.2f}",
            "tier": "cyan"
        })

    # 6. Cyan: Visceral Somatic Imagery if needed
    q_sbd = q_metrics["sensory_body_density"]
    c_sbd = c_metrics["sensory_body_density"]
    if len(badges) < 4 and (c_sbd >= 0.40 or abs(q_sbd - c_sbd) <= 0.15):
        badges.append({
            "type": "metric",
            "label": "Imagery",
            "detail": f"Visceral {round(c_sbd * 100)}%",
            "tier": "cyan"
        })

    # Guarantee at least 1 badge
    if not badges:
        badges.append({
            "type": "metric",
            "label": "Affinity",
            "detail": f"{int(round(score * 100))}% prose affinity",
            "tier": "cyan"
        })

    return badges


def _infer_query_anatomy(query_text: Optional[str], query_semantic: Optional[dict], query_features: dict) -> dict:
    """
    Extracts or dynamically infers high-precision story anatomy across the 3 core narrative pillars:
      1. Inciting Incident / Catalyst (e.g. Reincarnation into Nobility, Villainess Subversion)
      2. World Setting & Atmosphere (e.g. High Fantasy Imperial Court & Aristocracy)
      3. Core Conflict & Stakes (e.g. Imperial Succession & Factional Politics, Subverting Doom)
      4. Salient Trope Motifs
    
    Synthesizes signals from novel title, synopsis, tags, genres, and taxonomy metadata,
    as well as raw user prose inputs.
    """
    title = str(query_features.get("title") or "").strip()
    synopsis = str(query_features.get("synopsis") or "").strip()
    tags = str(query_features.get("tags") or "").strip()
    p_genre = str(query_features.get("primary_genre") or query_features.get("genre") or "").strip()
    territory = str(query_features.get("territory") or "").strip()
    
    # Combined lowercase search space
    context = f"{title} {synopsis} {tags} {p_genre} {territory} {query_text or ''}".lower()
    
    # Extract taxonomy if present
    tax = {}
    if query_semantic and isinstance(query_semantic.get("taxonomy"), dict):
        tax = query_semantic["taxonomy"]
    elif query_features and isinstance(query_features.get("taxonomy"), dict):
        tax = query_features["taxonomy"]

    inc_tax = str(tax.get("inciting_event", {}).get("primary", "") if isinstance(tax.get("inciting_event"), dict) else "")
    ws_tax = str(tax.get("world_setting", {}).get("primary", "") if isinstance(tax.get("world_setting"), dict) else "")
    plot_tax = str(tax.get("narrative_plot", {}).get("primary", "") if isinstance(tax.get("narrative_plot"), dict) else "")

    # ── 1. Catalyst (Inciting Incident) ──
    catalyst = None
    if re.search(r"reincarnat|past life|previous life|reborn|truck-kun|isekai", context):
        if re.search(r"villainess|otome|death flag|game world|heroine", context):
            catalyst = "Villainess Subversion Reincarnation"
        elif re.search(r"noble|prince|duke|aristocrat|emperor|royal|count|baron|marquis|lineage", context):
            catalyst = "Reincarnation into Imperial Nobility"
        elif re.search(r"strongest|cheat|overpower|blessed|birth|prodigy|baby", context):
            catalyst = "Overpowered Rebirth & Prodigy"
        else:
            catalyst = "Otherworldly Reincarnation"
    elif re.search(r"transmigrat|possessed|body snatch", context):
        if re.search(r"villainess|otome|death flag", context):
            catalyst = "Transmigration into Otome Fate"
        elif re.search(r"noble|prince|aristocrat|duke", context):
            catalyst = "Transmigration into High Nobility"
        else:
            catalyst = "Otherworldly Transmigration"
    elif re.search(r"regress|turn back time|second chance|return to the past|time loop|died and returned|redo", context):
        catalyst = "Regression & Second Chance"
    elif re.search(r"summoned|summoning circle|hero summoning|transported to|another world|portal", context):
        catalyst = "Otherworldly Hero Summoning"
    elif re.search(r"system|status window|level up|awakened|awakening|hunter rank|quest notification", context):
        catalyst = "System Interface & Status Awakening"
    elif re.search(r"betray|exiled|banished|framed|backstab|poisoned|abandoned by", context):
        catalyst = "Betrayal & Fall from Grace"
    elif re.search(r"magic academy|academy entrance|dormitory|enrolled|grimoire|ancient relic|hidden talent", context):
        catalyst = "Hidden Heritage & Academy Enrollment"
    elif re.search(r"murder|crime|detective|investigat|corpse|slain", context):
        catalyst = "Murder Mystery & Investigation"
    elif re.search(r"qi|cultivat|sect|dantian|meridian|martial arts|wuxia|xianxia", context):
        catalyst = "Cultivation Initiation & Meridian Awakening"

    # Fallback to taxonomy if keyword not found
    if not catalyst:
        if "Isekai & Regression" in inc_tax or "Isekai" in inc_tax:
            catalyst = "Otherworldly Reincarnation"
        elif "Summoning" in inc_tax:
            catalyst = "Otherworldly Summoning"
        elif "Personal Crisis" in inc_tax:
            catalyst = "Personal Crisis & Awakening"
        elif "Supernatural Awakening" in inc_tax:
            catalyst = "Supernatural Awakening"
        elif inc_tax:
            catalyst = f"{inc_tax} Spark"
        elif p_genre:
            catalyst = f"{p_genre} Narrative Spark"
        else:
            catalyst = "Premise Inciting Spark"

    # ── 2. Setting (Worldbuilding & Atmosphere) ──
    setting = None
    if re.search(r"palace|empire|emperor|empress|grand duke|noble|crown prince|aristocrat|royal court|duchy|nobility", context):
        if re.search(r"otome|villainess", context):
            setting = "Otome Aristocratic Empire"
        else:
            setting = "High Fantasy Imperial Court & Aristocracy"
    elif re.search(r"villainess|otome|death flag|broken engagement", context):
        setting = "Otome Aristocratic Empire"
    elif re.search(r"kingdom|feudal|manor|peasant|knight|medieval|castle|lord", context):
        setting = "Feudal Aristocratic Kingdom"
    elif re.search(r"dungeon|gate|hunter|monster break|seoul|tokyo|skyscrapers|guild master|raid", context):
        setting = "Urban Fantasy & Monster Gates"
    elif re.search(r"dungeon|tower|labyrinth|monster floor", context):
        setting = "Subterranean Dungeon Labyrinth"
    elif re.search(r"qi|cultivat|sect|dantian|meridian|martial arts|wuxia|xianxia|elder|soaring sword|murim|jianghu", context):
        setting = "Cultivation Sects & Martial World"
    elif re.search(r"academy|mana|spell|sorcery|archmage|elemental|guild|dragon|enchanted forest", context):
        setting = "High Magic Academy & Sorcery Realm"
    elif re.search(r"cyber|neon|implant|megacorp|starship|galaxy|android|colony|post-apocalyptic", context):
        setting = "Dystopian Sci-Fi Frontier"
    elif re.search(r"village|pastoral|countryside|farm|tavern|shop|slow life", context):
        setting = "Pastoral Countryside & Frontier Town"
    elif re.search(r"historical|dynasty|shogunate|samurai|edo|feudal japan", context):
        setting = "Historical Dynastic Era"

    # Fallback to taxonomy/territory
    if not setting:
        if ws_tax and ws_tax != "Isekai" and ws_tax != "Unknown":
            setting = f"{ws_tax} World"
        elif p_genre and p_genre.lower() != "isekai" and p_genre.lower() != "unknown":
            setting = f"High Fantasy {p_genre} Realm"
        elif "Web Novel" in territory or "Light Novel" in territory:
            setting = "High Fantasy Kingdoms & Empire"
        elif territory and territory != "Unknown":
            setting = f"{territory} Setting"
        else:
            setting = "High Fantasy Kingdoms & Empire"

    # ── 3. Conflict (Stakes & Tension) ──
    conflict = None
    if re.search(r"villainess|death flag|ruin|broken engagement|execution|exile", context):
        conflict = "Subverting Doom & Aristocratic Ruin"
    elif re.search(r"succession|throne|intrigue|faction|conspiracy|court politics|power struggle|noble rivalry", context):
        conflict = "Imperial Succession & Factional Politics"
    elif re.search(r"strongest|power from birth|blessed|overpower|cheat|unrivaled|god-level", context):
        conflict = "Imperial Succession & Concealing Overpowered Might"
    elif re.search(r"civil war|war|rebellion|battlefield|invader|conquest|territory", context):
        conflict = "Territorial Warfare & Kingdom Building"
    elif re.search(r"revenge|avenge|payback|retribution|destroy them|sworn enemy|blood debt", context):
        conflict = "Vengeance & Retributive Justice"
    elif re.search(r"survival|death game|boss monster|calamity|apocalypse|catastrophe|extinction", context):
        conflict = "Cataclysmic Survival & Dungeon Conquest"
    elif re.search(r"romance|fiance|engagement|marriage|love interest|harem|otome", context):
        conflict = "Romantic Intrigue & Social Maneuvering"
    elif re.search(r"cultivat|sect|breakthrough|ascension|tribulation|dao", context):
        conflict = "Sect Rivalries & Heavenly Dao Ascension"
    elif re.search(r"mystery|crime|murder|culprit|detective|investigation", context):
        conflict = "Unmasking Conspiracies & Solving Murders"
    elif re.search(r"slow life|cooking|peace|comfort|farming|relax", context):
        conflict = "Pastoral Slow-Life & Cozy Complications"

    # Fallback to narrative_plot taxonomy
    if not conflict:
        if plot_tax == "Romance":
            conflict = "Court Romance & Social Stakes"
        elif plot_tax == "Action / Adventure":
            conflict = "Frontier Exploration & High-Stakes Combat"
        elif plot_tax == "Comedy":
            conflict = "Farcical Misunderstandings & Chaotic Schemes"
        elif plot_tax == "Drama":
            conflict = "Interpersonal Strife & Moral Dilemmas"
        elif plot_tax == "Mystery":
            conflict = "Solving Conspiracies & Occult Secrets"
        elif plot_tax == "Slice of Life":
            conflict = "Pastoral Life & Everyday Stakes"
        elif plot_tax == "Supernatural":
            conflict = "Supernatural Mysteries & Occult Threats"
        elif plot_tax and plot_tax != "Fantasy":
            conflict = f"{plot_tax} & High-Stakes Dramatic Tension"
        elif p_genre:
            conflict = f"Imperial Ambitions & {p_genre} Conflict"
        else:
            conflict = "Character Growth & Core Conflict"

    # ── 4. Tropes Extraction ──
    tropes = []
    if tags:
        tropes.extend([t.strip() for t in tags.split(",") if t.strip()])
        
    trope_patterns = [
        ("Overpowered Protagonist", r"overpower|cheat|supreme|unrivaled|god-level|strongest"),
        ("Hidden Identity", r"hide.*power|secret identity|disguise|unassuming|conceal"),
        ("Cold Aristocrat", r"cold duke|tyrant|calculat|emotionless|noble"),
        ("System Interface", r"status window|system notification|quest|level up"),
        ("Time Reversal", r"regress|return to past|second chance|do-over"),
        ("Reincarnation", r"reincarnat|past life|salaryman|reborn|isekai"),
        ("Found Family", r"companion|comrade|trusted ally|found family"),
        ("Academy Life", r"classroom|exam|dormitory|fellow student|professor|academy"),
        ("Dungeon Raid", r"dungeon|boss monster|raid|loot|mana core"),
        ("Political Intrigue", r"conspiracy|faction|noble court|scheming|treason|succession"),
        ("Villainess Route", r"villainess|otome|broken engagement|death flag"),
        ("Kingdom Building", r"territory|governance|kingdom building|domain|tax"),
    ]
    for trope_name, pat in trope_patterns:
        if re.search(pat, context):
            tropes.append(trope_name)

    # Deduplicate tropes while preserving order
    seen_tropes = set()
    cleaned_tropes = []
    for t in tropes:
        t_clean = t.strip().title()
        if t_clean and t_clean.lower() not in seen_tropes:
            seen_tropes.add(t_clean.lower())
            cleaned_tropes.append(t_clean)

    return {
        "catalyst": catalyst,
        "setting": setting,
        "conflict": conflict,
        "tropes": cleaned_tropes
    }


def _generate_narrative_synthesis(q_anat: dict, c_anat: dict, s_sim: float, g_sim: float, is_user_input: bool) -> str:
    """
    Synthesizes a cohesive 1-2 sentence narrative explanation comparing the query's
    catalyst, setting, and conflict with the candidate novel.
    """
    q_cat = q_anat.get("catalyst", "Premise Inciting Spark")
    c_cat = c_anat.get("catalyst", "Premise Inciting Spark")
    q_set = q_anat.get("setting", "High Fantasy Continent & Kingdoms")
    c_set = c_anat.get("setting", "High Fantasy Continent & Kingdoms")
    q_con = q_anat.get("conflict", "Character Growth & Core Conflict")
    c_con = c_anat.get("conflict", "Character Growth & Core Conflict")

    if is_user_input:
        if q_cat.lower() == c_cat.lower() or q_cat.lower() in c_cat.lower() or c_cat.lower() in q_cat.lower():
            return (
                f"Your prose's {q_cat} premise aligns directly with this novel's opening arc. "
                f"Both narratives immerse the reader in a {c_set} setting where the protagonist must navigate {c_con}."
            )
        else:
            return (
                f"Your narrative's focus on {q_cat} resonates with this novel's {c_cat} trajectory, "
                f"sharing a vivid {c_set} atmosphere underpinned by {c_con}."
            )
    else:
        if (q_cat.lower() == c_cat.lower()) and (q_set.lower() == c_set.lower()):
            return (
                f"Both stories share a high-stakes {q_cat} foundation situated in a {q_set} world, "
                f"centering on protagonists contending with {c_con}."
            )
        elif q_cat.lower() == c_cat.lower() or q_cat.lower() in c_cat.lower() or c_cat.lower() in q_cat.lower():
            return (
                f"Both narratives are anchored by a {q_cat} catalyst, exploring thematic tensions of "
                f"{c_con} against a richly drawn {c_set} backdrop."
            )
        else:
            return (
                f"These works mirror each other across key narrative pillars—pairing a {q_cat} premise with "
                f"a {c_set} world where characters face {c_con}."
            )


def _compute_4pillar_breakdown(
    q_anat: dict,
    c_anat: dict,
    q_m: dict,
    c_m: dict,
    s_sim: float,
    g_sim: float,
    sty_sim: float
) -> dict:
    """
    Computes structured scores, value mappings (Query -> Candidate), and comparative
    explanations across the 4 core story & style pillars.
    """
    q_cat = q_anat.get("catalyst", "Narrative Inciting Spark")
    c_cat = c_anat.get("catalyst", "Narrative Inciting Spark")
    q_set = q_anat.get("setting", "World Setting")
    c_set = c_anat.get("setting", "World Setting")
    q_con = q_anat.get("conflict", "Plot Stakes")
    c_con = c_anat.get("conflict", "Plot Stakes")

    # 1. Catalyst Score & Explanation
    cat_match = q_cat.lower() == c_cat.lower() or (q_cat.lower() in c_cat.lower()) or (c_cat.lower() in q_cat.lower())
    cat_score = max(0.55, min(0.98, s_sim * 0.7 + (0.28 if cat_match else 0.12)))
    if cat_match:
        cat_exp = f"Shared premise anchor: Both narratives launch from a {q_cat} foundation."
    else:
        cat_exp = f"Harmonious narrative catalyst: Shifts from {q_cat} into {c_cat} thematic beats."

    # 2. Setting Score & Explanation
    set_match = q_set.lower() == c_set.lower() or (q_set.lower() in c_set.lower()) or (c_set.lower() in q_set.lower())
    set_score = max(0.52, min(0.98, g_sim * 0.65 + (0.32 if set_match else 0.15)))
    if set_match:
        set_exp = f"Parallel worldbuilding: Richly realized {q_set} atmosphere and socio-political hierarchy."
    else:
        set_exp = f"Thematic atmospheric resonance: Translates {q_set} environmental motifs to a {c_set} landscape."

    # 3. Conflict Stakes & Explanation
    con_match = q_con.lower() == c_con.lower() or (q_con.lower() in c_con.lower()) or (c_con.lower() in q_con.lower())
    con_score = max(0.52, min(0.98, s_sim * 0.6 + (0.32 if con_match else 0.15)))
    if con_match:
        con_exp = f"Matching dramatic stakes: Central friction revolves around {q_con}."
    else:
        con_exp = f"Shared narrative tension: Balancing {q_con} alongside escalating {c_con} stakes."

    # 4. Style & Cadence Score & Explanation
    q_dlg = q_m.get("dialogue_ratio", 0.5) * 100
    c_dlg = c_m.get("dialogue_ratio", 0.5) * 100
    q_asl = q_m.get("avg_sentence_len", 12.0)
    c_asl = c_m.get("avg_sentence_len", 12.0)
    
    # Generate descriptive cadence strings
    def _format_cadence(dlg: float, asl: float) -> str:
        if dlg > 65:
            d_desc = f"Dialogue-Driven Pacing ({dlg:.0f}%)"
        elif dlg >= 45:
            d_desc = f"Balanced Narrative Flow ({dlg:.0f}%)"
        else:
            d_desc = f"Exposition-Dense ({dlg:.0f}%)"
            
        if asl < 10.0:
            a_desc = f"Snappy Beats ({asl:.1f} w/s)"
        elif asl <= 15.0:
            a_desc = f"Fluid Mid-Tempo ({asl:.1f} w/s)"
        else:
            a_desc = f"Layered Cadence ({asl:.1f} w/s)"
        return f"{d_desc}, {a_desc}"

    q_style_val = _format_cadence(q_dlg, q_asl)
    c_style_val = _format_cadence(c_dlg, c_asl)
    
    if sty_sim >= 0.85:
        sty_exp = "Strikingly congruent prose cadence, dynamic dialogue rhythm, and scene velocity."
    elif sty_sim >= 0.70:
        sty_exp = "Comparable dialogue-to-exposition pacing and structural sentence cadence."
    else:
        sty_exp = "Complementary stylistic rhythm offering a fresh reading tempo."

    return {
        "catalyst": {
            "name": "Premise & Inciting Catalyst",
            "score": round(cat_score, 2),
            "query_val": q_cat,
            "cand_val": c_cat,
            "explanation": cat_exp
        },
        "setting": {
            "name": "World Setting & Atmosphere",
            "score": round(set_score, 2),
            "query_val": q_set,
            "cand_val": c_set,
            "explanation": set_exp
        },
        "conflict": {
            "name": "Conflict Stakes & Tension",
            "score": round(con_score, 2),
            "query_val": q_con,
            "cand_val": c_con,
            "explanation": con_exp
        },
        "style_cadence": {
            "name": "Prose Voice & Cadence",
            "score": round(sty_sim, 2),
            "query_val": q_style_val,
            "cand_val": c_style_val,
            "explanation": sty_exp
        }
    }


def _extract_shared_tropes(q_anat: dict, c_anat: dict) -> list:
    """
    Finds overlapping trope chips between query and candidate.
    Falls back to prominent candidate tropes if no strict intersection exists.
    """
    q_set = {t.lower(): t for t in q_anat.get("tropes", [])}
    c_set = {t.lower(): t for t in c_anat.get("tropes", [])}
    
    shared_keys = set(q_set.keys()).intersection(set(c_set.keys()))
    shared = [c_set[k] for k in shared_keys]
    
    if not shared and c_anat.get("tropes"):
        shared = c_anat["tropes"][:3]
    return shared

def find_top_matches(
    query_features: Optional[dict] = None,
    query_text: Optional[str] = None,
    exclude_novel_id: Optional[int] = None,
    top_k: int = 5,
    target_novel_id: Optional[int] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Performs multi-faceted vector similarity search comparing query features, semantic
    concept embeddings, genres, tags, and territories against all database novels.

    Returns per-match breakdown for frontend display.

    Scoring weights:
      - 35% Semantic concept embedding (384D sentence transformer cosine similarity)
      - 30% Stylistic similarity (8D radar cosine + L1)
      - 25% Genre similarity & primary genre affinity (Jaccard + primary match bonus)
      -  5% Territory semantic similarity
      -  5% Fine-grained tag overlap (Jaccard set similarity)
    """
    if limit is not None:
        top_k = limit
    if target_novel_id is not None:
        if len(_novel_vector_cache) == 0:
            _init_cache_from_disk()
        meta = _novel_vector_cache.get(target_novel_id)
        if meta:
            query_features = meta.get("raw_features", meta)
            exclude_novel_id = target_novel_id
        else:
            query_features = {}
    elif query_features is None:
        query_features = {}

    if len(_novel_vector_cache) == 0:
        _init_cache_from_disk()

    q_vec = extract_feature_vector(query_features)
    q_metrics = _extract_metric_values(query_features, q_vec)

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

    q_tax = (
        query_semantic.get("taxonomy", {})
        if query_semantic and isinstance(query_semantic.get("taxonomy"), dict)
        else (query_features.get("taxonomy", {}) if isinstance(query_features.get("taxonomy"), dict) else {})
    )
    q_inciting = q_tax.get("inciting_event", {}).get("primary") if isinstance(q_tax.get("inciting_event"), dict) else ""
    q_world = q_tax.get("world_setting", {}).get("primary") if isinstance(q_tax.get("world_setting"), dict) else ""
    q_plot = q_tax.get("narrative_plot", {}).get("primary") if isinstance(q_tax.get("narrative_plot"), dict) else ""
    q_title = query_features.get("title") or (target_novel_meta.get("title") if target_novel_meta else "")

    # Query concept embedding (384D)
    try:
        q_genre_str = q_primary_genre or (", ".join(sorted(target_genres)) if target_genres else "")
        q_concept_emb = _get_concept_embedding(
            title=q_title,
            genre=q_genre_str,
            territory=target_territory or "",
            inciting_event=q_inciting,
            world_setting=q_world,
            narrative_plot=q_plot,
            tags=raw_target_tags
        )
    except Exception:
        q_concept_emb = np.zeros(384, dtype=np.float32)

    q_norm = float(np.linalg.norm(q_concept_emb))

    q_anat = _infer_query_anatomy(query_text, query_semantic, query_features)
    is_user_input = bool(query_text and not query_features.get("title"))

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

        # ── Factor 1: Stylistic Similarity (15% tiebreaker) ──
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

        # ── Factor 3: Semantic Concept Embedding (45% of story weight) ──
        n_territory = n_meta.get("territory") or "Unknown"
        title_str = n_meta.get("title", "")
        author_str = n_meta.get("author", "")

        n_concept_emb = _get_concept_embedding(
            title=title_str,
            genre=cand_primary,
            territory=n_territory,
            inciting_event=n_meta.get("inciting_event", ""),
            world_setting=n_meta.get("world_setting", ""),
            narrative_plot=n_meta.get("narrative_plot", ""),
            tags=n_meta.get("tags", "")
        )
        n_norm = float(np.linalg.norm(n_concept_emb))
        if q_norm > 1e-9 and n_norm > 1e-9:
            sem_raw = float(np.clip(np.dot(q_concept_emb, n_concept_emb) / (q_norm * n_norm), 0.0, 1.0))
        else:
            sem_raw = 0.40

        # Title keyword bonus
        if q_title:
            q_lower = q_title.lower()
            title_tokens = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", title_str.lower()) if w not in {"with", "from", "that", "this", "into", "over", "about", "world"}]
            title_matches = sum(1 for w in set(title_tokens) if w in q_lower)
            token_overlap = min(0.30, 0.15 * title_matches)
            semantic_sim = float(np.clip(0.70 * sem_raw + token_overlap, 0.0, 1.0))
        elif query_text:
            q_lower = query_text.lower()
            title_tokens = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", title_str.lower()) if w not in {"with", "from", "that", "this", "into", "over", "about"}]
            title_matches = sum(1 for w in set(title_tokens) if w in q_lower)
            token_overlap = min(0.30, 0.15 * title_matches)
            semantic_sim = float(np.clip(0.70 * sem_raw + token_overlap, 0.0, 1.0))
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

        # ── Story Similarity Formulation (Story Content & Narrative Anatomy) ──
        story_sim = float(
            0.45 * semantic_sim
            + 0.35 * genre_sim
            + 0.15 * tag_sim
            + 0.05 * territory_sim
        )
        story_sim = float(np.clip(story_sim, 0.0, 1.0))

        # ── Composite Overall Score (85% Story Content + 15% Prose Style Tiebreaker) ──
        composite_score = 0.85 * story_sim + 0.15 * style_sim
        score = round(min(0.99, max(0.01, composite_score)), 4)

        # Generate story-first human-readable match reasons
        story_reasons = []
        style_reasons = []

        if q_primary_genre_lower and cand_primary and q_primary_genre_lower == cand_primary:
            cand_primary_clean = n_meta.get("primary_genre") or n_meta.get("genre")
            story_reasons.append(f"Matching primary archetype: {cand_primary_clean}")
        elif q_primary_genre_lower and q_primary_genre_lower in cand_genres:
            story_reasons.append(f"Shared genre: {q_primary_genre}")
        elif genre_sim >= 0.60:
            story_reasons.append("Strong genre overlap")

        if semantic_sim >= 0.80:
            story_reasons.append("Closely aligned plot premise & themes")
        elif semantic_sim >= 0.65:
            story_reasons.append("Thematic narrative overlap")

        if tag_sim >= 0.60:
            story_reasons.append("Overlapping narrative tropes")

        if territory_sim >= 0.85 and n_territory and n_territory != "Unknown":
            if "classic" in n_territory.lower():
                story_reasons.append("Shared Classic Literature tradition")
            elif "web" in n_territory.lower():
                story_reasons.append("Shared Web Novel territory")

        if style_sim >= 0.88:
            style_reasons.append("Similar prose style & sentence structure")
        elif style_sim >= 0.75:
            style_reasons.append("Comparable sentence cadence")

        # Combined reasons list puts Story features first
        reasons = story_reasons.copy()
        if style_reasons:
            reasons.extend(style_reasons)
        if not reasons:
            reasons.append("Overall thematic and stylistic affinity")

        # Compute granular match badges and side-by-side metric comparisons
        c_metrics = _extract_metric_values(n_meta, n_vec)
        cand_primary_clean = n_meta.get("primary_genre") or n_meta.get("genre") or "Fiction"
        metric_comparisons = _compute_metric_comparisons(q_metrics, c_metrics)
        match_badges = _compute_match_badges(
            q_metrics=q_metrics,
            c_metrics=c_metrics,
            query_text=query_text,
            query_semantic=query_semantic,
            query_features=query_features,
            cand_meta=n_meta,
            cand_primary_genre=cand_primary_clean,
            style_sim=style_sim,
            score=score
        )

        c_anat = _infer_query_anatomy(None, n_meta.get("semantic"), n_meta.get("raw_features", n_meta))
        narrative_synthesis = _generate_narrative_synthesis(q_anat, c_anat, story_sim, genre_sim, is_user_input)
        pillars = _compute_4pillar_breakdown(q_anat, c_anat, q_metrics, c_metrics, story_sim, genre_sim, style_sim)
        shared_tropes = _extract_shared_tropes(q_anat, c_anat)
        
        narrative_reasoning = {
            "narrative_synthesis": narrative_synthesis,
            "pillars": pillars,
            "shared_tropes": shared_tropes
        }

        candidates.append({
            "id": nid,
            "title": title_str,
            "author": author_str,
            "genre": n_meta.get("genre", ""),
            "territory": n_territory or "Unknown",
            "similarity_score": score,
            "story_similarity": int(round(story_sim * 100)),
            "style_similarity": int(round(style_sim * 100)),
            "match_badges": match_badges,
            "metric_comparisons": metric_comparisons,
            "reasons": reasons,
            "story_reasons": story_reasons,
            "style_reasons": style_reasons,
            "narrative_reasoning": narrative_reasoning,
            "breakdown": {
                "story": round(story_sim, 3),
                "style": round(style_sim, 3),
                "semantic": round(semantic_sim, 3),
                "genre": round(genre_sim, 3),
                "tags": round(tag_sim, 3),
                "territory": round(territory_sim, 3),
            }
        })

    # Sort primarily by story-dominant composite score descending, then by pure story score
    candidates.sort(key=lambda x: (x["similarity_score"], x["breakdown"]["story"]), reverse=True)
    return candidates[:top_k]


