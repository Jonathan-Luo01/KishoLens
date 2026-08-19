import re
import uuid
import random
import warnings
import os
import json
from typing import Dict, Any
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select, func

from kisholens.pipeline.main import run_etl

from kisholens.models import Novel, Chapter, get_engine
from kisholens.ml.features import (
    extract_english_features,
    extract_japanese_features,
    extract_chinese_features,
    match_archetype,
    normalize_feature_percentile,
    split_paragraphs,
    detect_language,
    _init_nlp_resources,
)
from kisholens.ml.semantic_match import match_semantic
from kisholens.ml.sentiment_arc import compute_kishotenketsu_quantile_arc
from kisholens.ml.similarity import find_top_matches
from kisholens.storage.r2 import sync_from_r2

import sqlite3

# Pre-computed cache paths
DATA_CACHE_PATH = "data/stats_cache.json"
STATS_DB_PATH = "data/stats_cache.sqlite"
VECTOR_CACHE_PATH = "data/vector_cache.json"
ARC_CACHE_PATH = "data/arc_cache.json"

_cached_novel_stats: Dict[int, Any] = {}
_cached_novel_arcs: Dict[int, Any] = {}


def _get_stats_db_conn():
    if not os.path.exists(STATS_DB_PATH):
        if not os.path.exists(DATA_CACHE_PATH):
            sync_from_r2()
        if os.path.exists(DATA_CACHE_PATH):
            _build_sqlite_stats_cache()
        else:
            return None
    try:
        return sqlite3.connect(STATS_DB_PATH, check_same_thread=False)
    except Exception as e:
        print(f"[CACHE ERROR] Could not connect to SQLite stats DB: {e}")
        return None


def _build_sqlite_stats_cache():
    if not os.path.exists(DATA_CACHE_PATH):
        return
    import gc
    try:
        print(f"[CACHE] Indexing {DATA_CACHE_PATH} into lightweight SQLite database {STATS_DB_PATH}...")
        conn = sqlite3.connect(STATS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS stats (id INTEGER PRIMARY KEY, title TEXT, author TEXT, genre TEXT, territory TEXT, data TEXT)")
        with open(DATA_CACHE_PATH, "r", encoding="utf-8") as f:
            stats_data = json.load(f)
        rows = [
            (int(k), v.get("title", ""), v.get("author", ""), v.get("genre", ""), v.get("territory", ""), json.dumps(v))
            for k, v in stats_data.items()
        ]
        del stats_data
        gc.collect()
        cursor.executemany("INSERT OR REPLACE INTO stats VALUES (?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
        conn.close()
        del rows
        gc.collect()
        print(f"[CACHE] Successfully indexed novels into SQLite cache.")
    except Exception as e:
        print(f"[CACHE WARN] Could not build SQLite cache from JSON: {e}")


def _load_disk_cache():
    if os.path.exists(STATS_DB_PATH):
        print(f"[CACHE] Connected to lightweight SQLite index at {STATS_DB_PATH} (< 5MB RAM).")
        return
    if os.path.exists(DATA_CACHE_PATH):
        _build_sqlite_stats_cache()

def _save_disk_cache():
    if not _cached_novel_stats:
        print("[CACHE WARN] Refusing to overwrite disk cache with empty dict.")
        return
    try:
        existing = {}
        if os.path.exists(DATA_CACHE_PATH):
            with open(DATA_CACHE_PATH, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = {}
        for k, v in _cached_novel_stats.items():
            existing[str(k)] = v
        with open(DATA_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False)
    except Exception as e:
        print(f"[CACHE WARN] Could not save disk cache: {e}")

def _load_arc_disk_cache():
    if os.path.exists(ARC_CACHE_PATH):
        try:
            with open(ARC_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    _cached_novel_arcs[int(k)] = v
            print(f"[CACHE] Loaded {len(_cached_novel_arcs)} pre-computed novel arcs from disk cache.")
        except Exception as e:
            print(f"[CACHE WARN] Could not load arc disk cache: {e}")

def _save_arc_disk_cache():
    if not _cached_novel_arcs:
        print("[CACHE WARN] Refusing to overwrite arc disk cache with empty dict.")
        return
    try:
        existing = {}
        if os.path.exists(ARC_CACHE_PATH):
            with open(ARC_CACHE_PATH, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = {}
        for k, v in _cached_novel_arcs.items():
            existing[str(k)] = v
        with open(ARC_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False)
    except Exception as e:
        print(f"[CACHE WARN] Could not save arc disk cache: {e}")


def _load_vector_disk_cache():
    """Load pre-computed novel feature vectors into similarity._novel_vector_cache."""
    from kisholens.ml.similarity import _novel_vector_cache
    import numpy as np
    if os.path.exists(VECTOR_CACHE_PATH):
        try:
            with open(VECTOR_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                entry = dict(v)
                entry["vector"] = np.array(entry["vector"], dtype=float)
                _novel_vector_cache[int(k)] = entry
            print(f"[CACHE] Loaded {len(_novel_vector_cache)} novel vectors from disk vector cache.")
        except Exception as e:
            print(f"[CACHE WARN] Could not load vector cache: {e}")

def _save_novel_to_vector_cache(novel_id: int, title: str, author: str, genre: str, territory: str, feature_vec):
    """Persist a single novel's vector to the vector cache JSON (incremental save)."""
    from kisholens.ml.similarity import _novel_vector_cache
    import numpy as np
    entry = {
        "id": novel_id,
        "title": title,
        "author": author or "Unknown Author",
        "genre": genre or "",
        "territory": territory or "Unknown",
        "vector": feature_vec.tolist() if hasattr(feature_vec, 'tolist') else list(feature_vec),
        "semantic": None,
    }
    _novel_vector_cache[novel_id] = {**entry, "vector": np.array(entry["vector"], dtype=float)}
    # Incremental save: update only this novel's entry in the JSON file
    try:
        existing = {}
        if os.path.exists(VECTOR_CACHE_PATH):
            with open(VECTOR_CACHE_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing[str(novel_id)] = entry
        with open(VECTOR_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False)
    except Exception as e:
        print(f"[CACHE WARN] Could not save vector cache entry: {e}")

# Immediately load pre-computed disk caches upon module import
_load_disk_cache()
_load_arc_disk_cache()
_load_vector_disk_cache()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fast, non-blocking lifespan that allows instant binding to Cloud Run port 8080."""
    print("[SERVER] KishoLens API server ready and listening on port.")
    yield

app = FastAPI(title="KishoLens API", lifespan=lifespan)

# Add CORS middleware so the Astro frontend (Cloudflare Pages, GitHub Pages, localhost) can fetch data
cors_env = os.getenv("CORS_ORIGINS")
origins = [o.strip() for o in cors_env.split(",") if o.strip()] if cors_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.pages\.dev|https://.*\.github\.io|http://localhost:.*|http://127\.0\.0\.1:.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = get_engine()


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "kisholens"}


@app.get("/api/novels")
def get_novels():
    try:
        results = []
        try:
            with Session(engine) as session:
                stmt = (
                    select(Novel, func.count(Chapter.id).label("chapter_count"))
                    .outerjoin(Chapter, Novel.id == Chapter.novel_id)
                    .group_by(Novel.id)
                )
                results = session.exec(stmt).all()
        except Exception:
            results = []

        if results:
            return [
                {
                    "id": novel.id,
                    "title": novel.title,
                    "author": novel.author,
                    "source": novel.source,
                    "chapter_count": chapter_count,
                    "genre": novel.genre,
                    "territory": novel.territory,
                }
                for novel, chapter_count in results
            ]
        
        # Fallback to SQLite stats cache if raw SQLite table is not populated
        conn = _get_stats_db_conn()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, author, genre, territory FROM stats")
            rows = cursor.fetchall()
            if rows:
                return [
                    {
                        "id": r[0],
                        "title": r[1] or f"Novel #{r[0]}",
                        "author": r[2] or "Unknown Author",
                        "source": "cache",
                        "chapter_count": 1,
                        "genre": r[3] or "",
                        "territory": r[4] or "Unknown",
                    }
                    for r in rows
                ]

        if _cached_novel_stats:
            return [
                {
                    "id": nid,
                    "title": stats.get("title", f"Novel #{nid}"),
                    "author": stats.get("author", "Unknown Author"),
                    "source": "cache",
                    "chapter_count": stats.get("en_sentence_count", 1) or 1,
                    "genre": stats.get("genre", ""),
                    "territory": stats.get("territory", "Unknown"),
                }
                for nid, stats in _cached_novel_stats.items()
            ]
        return []
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database connection error: {str(e)}"
        )


@app.get("/api/db/stats")
def get_db_stats():
    """Returns database overview statistics including total novels, source counts, territory counts, and genre breakdown."""
    try:
        novels = []
        try:
            with Session(engine) as session:
                novels = session.exec(select(Novel)).all()
        except Exception:
            novels = []

        genres_list = [
            "Action / Adventure", "Comedy", "Drama", "Fantasy", "Horror",
            "Historical", "Sci-Fi", "Philosophy", "Mystery", "Tragedy",
            "Supernatural", "Poetry", "Romance", "Slice of Life",
            "Cultivation", "Isekai", "Progression Fantasy"
        ]

        if novels:
            total_novels = len(novels)
            by_source = {}
            by_territory = {}
            by_genre = {g: 0 for g in genres_list}

            for novel in novels:
                raw_src = novel.source or "unknown"
                src = raw_src.split("/")[0].lower() if "/" in raw_src else raw_src.lower()
                by_source[src] = by_source.get(src, 0) + 1

                terr = novel.territory or ("Classic Literature Territory" if src == "gutenberg" else "Web Novel Territory")
                by_territory[terr] = by_territory.get(terr, 0) + 1

                n_g = (novel.genre or "").lower()
                for g in genres_list:
                    if g.lower() in n_g:
                        by_genre[g] += 1

            return {
                "total_novels": total_novels,
                "by_source": by_source,
                "by_territory": by_territory,
                "by_genre": by_genre,
            }

        # Query from SQLite stats cache
        conn = _get_stats_db_conn()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT genre, territory FROM stats")
            rows = cursor.fetchall()
            if rows:
                total_novels = len(rows)
                by_source = {"web novel": 0, "gutenberg": 0}
                by_territory = {}
                by_genre = {g: 0 for g in genres_list}

                for genre_val, terr_val in rows:
                    terr = terr_val or "Web Novel Territory"
                    by_territory[terr] = by_territory.get(terr, 0) + 1
                    if "Classic" in terr:
                        by_source["gutenberg"] += 1
                    else:
                        by_source["web novel"] += 1

                    n_g = (genre_val or "").lower()
                    for g in genres_list:
                        if g.lower() in n_g:
                            by_genre[g] += 1

                return {
                    "total_novels": total_novels,
                    "by_source": by_source,
                    "by_territory": by_territory,
                    "by_genre": by_genre,
                }

        return {"total_novels": 0, "by_source": {}, "by_territory": {}, "by_genre": {}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/novels/{novel_id}")
def get_novel(novel_id: int):
    try:
        with Session(engine) as session:
            novel = session.get(Novel, novel_id)
            if not novel:
                raise HTTPException(status_code=404, detail="Novel not found")

            chapters = session.exec(
                select(Chapter)
                .where(Chapter.novel_id == novel_id)
                .order_by(Chapter.chapter_number)
            ).all()

            return {
                "id": novel.id,
                "title": novel.title,
                "author": novel.author,
                "source": novel.source,
                "chapters": [
                    {
                        "id": ch.id,
                        "novel_id": ch.novel_id,
                        "chapter_number": ch.chapter_number,
                        "title": ch.title,
                        "text_ja": ch.text_ja,
                        "text_en": ch.text_en,
                        "text_zh": getattr(ch, "text_zh", ""),
                    }
                    for ch in chapters
                ]
            }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Database connection error: {str(e)}"
        )


@app.get("/api/stats/sources")
def get_stats_sources():
    try:
        with Session(engine) as session:
            novels = session.exec(select(Novel)).all()
            novels_map = {n.id: n for n in novels}
            chapters = session.exec(select(Chapter)).all()

            features_list = []
            for ch in chapters:
                novel = novels_map.get(ch.novel_id)
                if not novel:
                    continue
                row = {
                    "source": novel.source
                }
                if ch.text_en:
                    en_feat = extract_english_features(ch.text_en)
                    row.update({f"en_{k}": v for k, v in en_feat.items()})
                if ch.text_ja:
                    ja_feat = extract_japanese_features(ch.text_ja)
                    row.update({f"ja_{k}": v for k, v in ja_feat.items()})
                text_zh = getattr(ch, "text_zh", "")
                if text_zh:
                    zh_feat = extract_chinese_features(text_zh)
                    row.update({f"zh_{k}": v for k, v in zh_feat.items()})
                features_list.append(row)

            if not features_list:
                return []

            # Group rows by source platform
            grouped = {}
            for row in features_list:
                src = row["source"]
                if src not in grouped:
                    grouped[src] = []
                grouped[src].append(row)

            # Calculate average for each numeric metric key per source
            agg_records = []
            for src, rows in grouped.items():
                agg = {"source": src}
                # Gather all keys present in the rows for this source
                keys = set()
                for r in rows:
                    keys.update(k for k in r.keys() if k != "source")

                for k in sorted(keys):
                    vals = [r[k] for r in rows if k in r and r[k] is not None]
                    agg[k] = sum(vals) / len(vals) if vals else 0.0
                agg["archetype_match"] = match_archetype(agg)
                agg_records.append(agg)

            return agg_records
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database connection error: {str(e)}"
        )


class IngestRequest(BaseModel):
    dataset_name: str
    num_records: int


SIMILARITY_MODEL_VERSION = "v4_rich_story_embeddings"


@app.get("/api/novels/{novel_id}/stats")
def get_novel_stats(novel_id: int):
    global _cached_novel_stats
    if novel_id in _cached_novel_stats:
        stats = _cached_novel_stats[novel_id]
        if (
            stats.get("similarity_version") != SIMILARITY_MODEL_VERSION
            or not stats.get("top_matches")
            or not any(m.get("match_badges") for m in stats.get("top_matches", []))
        ):
            from kisholens.ml.similarity import find_top_matches
            stats["top_matches"] = find_top_matches(stats, exclude_novel_id=novel_id, top_k=5)
            stats["similarity_version"] = SIMILARITY_MODEL_VERSION
            _save_disk_cache()
        return stats

    # Fast query from lightweight SQLite stats index
    conn = _get_stats_db_conn()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM stats WHERE id = ?", (novel_id,))
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
        except Exception as e:
            print(f"[CACHE WARN] SQLite query error for novel {novel_id}: {e}")

    try:
        with Session(engine) as session:
            novel = session.get(Novel, novel_id)
            if not novel:
                raise HTTPException(status_code=404, detail="Novel not found")

            chapters = session.exec(
                select(Chapter)
                .where(Chapter.novel_id == novel_id)
            ).all()

            # Fast sampling for stats API: sample up to 6 evenly-spaced chapters across the book
            sorted_chs = sorted(chapters, key=lambda c: c.chapter_number)
            if len(sorted_chs) > 6:
                step = (len(sorted_chs) - 1) / 5.0
                sampled_chs = [sorted_chs[int(round(i * step))] for i in range(6)]
            else:
                sampled_chs = sorted_chs

            features_list = []
            for ch in sampled_chs:
                row = {}
                if ch.text_en:
                    en_feat = extract_english_features(ch.text_en)
                    row.update({f"en_{k}": v for k, v in en_feat.items()})
                if ch.text_ja:
                    ja_feat = extract_japanese_features(ch.text_ja)
                    row.update({f"ja_{k}": v for k, v in ja_feat.items()})
                text_zh = getattr(ch, "text_zh", "")
                if text_zh:
                    zh_feat = extract_chinese_features(text_zh)
                    row.update({f"zh_{k}": v for k, v in zh_feat.items()})
                if row:
                    features_list.append(row)

            if not features_list:
                return {}

            keys = set()
            for r in features_list:
                keys.update(r.keys())

            agg = {}
            for k in sorted(keys):
                vals = [r[k] for r in features_list if k in r and r[k] is not None]
                agg[k] = sum(vals) / len(vals) if vals else None

            # Add percentile normalized radar scores to agg
            normalized_radar = {}
            for k in sorted(keys):
                if agg[k] is not None:
                    base_k = re.sub(r'^(en_|ja_|zh_)', '', k)
                    normalized_radar[k] = normalize_feature_percentile(base_k, agg[k])
            agg["normalized_radar"] = normalized_radar

            # Dynamically compute pacing paragraph lengths for the barcodes
            paragraph_lengths = []
            for ch in sorted(chapters, key=lambda c: c.chapter_number):
                text = ch.text_en or ch.text_ja or getattr(ch, "text_zh", "") or ""
                paragraphs = split_paragraphs(text)
                for p in paragraphs:
                    if ch.text_en:
                        word_count = len(re.findall(r'\b\w+\b', p))
                    else:
                        word_count = len([c for c in p if not c.isspace()])
                    if word_count > 0:
                        paragraph_lengths.append(word_count)
            
            # Limit to first 100 paragraphs for dashboard display
            agg["pacing"] = paragraph_lengths[:100]

            # Compute semantic genre and territory on-the-fly using first, middle, and last chapters
            sorted_chs = sorted(chapters, key=lambda c: c.chapter_number)
            if not sorted_chs:
                text = ""
            elif len(sorted_chs) == 1:
                text = sorted_chs[0].text_en or sorted_chs[0].text_ja or getattr(sorted_chs[0], "text_zh", "") or ""
            elif len(sorted_chs) == 2:
                text1 = sorted_chs[0].text_en or sorted_chs[0].text_ja or getattr(sorted_chs[0], "text_zh", "") or ""
                text2 = sorted_chs[1].text_en or sorted_chs[1].text_ja or getattr(sorted_chs[1], "text_zh", "") or ""
                text = text1 + "\n\n" + text2
            else:
                ch_beg = sorted_chs[0]
                ch_mid = sorted_chs[len(sorted_chs) // 2]
                ch_end = sorted_chs[-1]
                text_beg = ch_beg.text_en or ch_beg.text_ja or getattr(ch_beg, "text_zh", "") or ""
                text_mid = ch_mid.text_en or ch_mid.text_ja or getattr(ch_mid, "text_zh", "") or ""
                text_end = ch_end.text_en or ch_end.text_ja or getattr(ch_end, "text_zh", "") or ""
                text = text_beg + "\n\n" + text_mid + "\n\n" + text_end

            semantic = match_semantic(text, title=novel.title, synopsis=getattr(novel, "synopsis", None), features=agg) if text else None

            if agg:
                # Detect language dynamically for baselines
                lang = "en"
                if chapters:
                    first_ch = chapters[0]
                    if not first_ch.text_en:
                        if first_ch.text_ja:
                            lang = "ja"
                        elif getattr(first_ch, "text_zh", ""):
                            lang = "zh"
                agg["baselines"] = compute_dynamic_baselines(lang)

                if semantic:
                    agg["archetype_match"] = {
                        "closest_trope": semantic["genre"],
                        "territory": semantic["territory"],
                        "confidence": semantic["genre_confidence"],
                        "top_genres": [{"genre": x["genre"], "confidence": x["score"]} for x in semantic["genre_scores"][:3]],
                        "top_territories": [{"territory": x["territory"], "confidence": x["score"]} for x in semantic["territory_scores"][:2]]
                    }
                    if "taxonomy" in semantic:
                        agg["taxonomy"] = semantic["taxonomy"]
                        agg["archetype_match"]["taxonomy"] = semantic["taxonomy"]
                else:
                    agg["archetype_match"] = {
                        "closest_trope": novel.genre or "Unknown",
                        "territory": novel.territory or "Unknown",
                        "confidence": 0.75,
                        "top_genres": [{"genre": novel.genre, "confidence": 0.75}] if novel.genre else [],
                        "top_territories": [{"territory": novel.territory, "confidence": 0.75}] if novel.territory else []
                    }

                # Compute top 3 nearest neighbor matching novels from database
                from kisholens.ml.similarity import extract_feature_vector
                import numpy as np
                feat_vec = extract_feature_vector(agg)
                agg["id"] = novel_id
                agg["title"] = novel.title
                agg["author"] = novel.author or "Unknown Author"
                agg["genre"] = novel.genre
                agg["territory"] = novel.territory or (semantic.get("territory") if semantic else "Unknown")
                if semantic:
                    agg["archetype"] = semantic["genre"]
                    agg["archetype_percentages"] = semantic["genre_scores"]

                _save_novel_to_vector_cache(
                    novel_id=novel_id,
                    title=novel.title,
                    author=novel.author or "",
                    genre=novel.genre or "",
                    territory=novel.territory or "Unknown",
                    feature_vec=feat_vec,
                )
                agg["top_matches"] = find_top_matches(agg, exclude_novel_id=novel_id, top_k=5)

            _cached_novel_stats[novel_id] = agg
            _save_disk_cache()
            return agg
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Database connection error: {str(e)}"
        )


@app.get("/api/novels/{novel_id}/arc")
def get_novel_arc(novel_id: int):
    """
    Computes the 4-act Kishōtenketsu sentiment arc for a novel with in-memory caching.
    """
    global _cached_novel_arcs
    if novel_id in _cached_novel_arcs:
        cached = _cached_novel_arcs[novel_id]
        # Always inject fresh curated baselines (cached entries may have stale flat averages)
        lang = cached.get("_lang", "en")
        cached["baselines"] = compute_dynamic_baselines(lang)["arc"]
        return cached

    try:
        with Session(engine) as session:
            novel = session.get(Novel, novel_id)
            if not novel:
                raise HTTPException(status_code=404, detail="Novel not found")

            chapters = session.exec(
                select(Chapter)
                .where(Chapter.novel_id == novel_id)
                .order_by(Chapter.chapter_number)
            ).all()

        lang = "en"
        if chapters:
            first_ch = chapters[0]
            if not first_ch.text_en:
                if first_ch.text_ja:
                    lang = "ja"
                elif getattr(first_ch, "text_zh", ""):
                    lang = "zh"

        # Sample up to 16 evenly-spaced chapters across the novel for fast arc computation
        if len(chapters) > 16:
            step = (len(chapters) - 1) / 15.0
            sampled_chapters = [chapters[int(round(i * step))] for i in range(16)]
        else:
            sampled_chapters = chapters

        all_sentences: list[str] = []
        for ch in sampled_chapters:
            if lang == "en":
                text = ch.text_en or ""
                sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            elif lang == "ja":
                text = ch.text_ja or ""
                sents = [s.strip() for s in re.split(r'[。！？]+', text) if s.strip()]
            else:
                text = getattr(ch, "text_zh", "") or ""
                sents = [s.strip() for s in re.split(r'[。！？]+', text) if s.strip()]
            all_sentences.extend(sents)

        if not all_sentences:
            raise HTTPException(
                status_code=422,
                detail="No raw text found for this novel"
            )

        # Fast representative sampling: sample up to 160 evenly-spaced sentences across the novel for instant arc calculation
        if len(all_sentences) > 160:
            step = (len(all_sentences) - 1) / 159.0
            sampled_sents = [all_sentences[int(round(i * step))] for i in range(160)]
        else:
            sampled_sents = all_sentences

        arc_res = compute_kishotenketsu_quantile_arc(sampled_sents, lang)

        arc_data = {
            "novel_id": novel_id,
            "title": novel.title,
            "acts": arc_res["acts"],
            "quantiles": arc_res["quantiles"],
            "baselines": compute_dynamic_baselines(lang)["arc"],
            "_lang": lang
        }
        _cached_novel_arcs[novel_id] = arc_data
        _save_arc_disk_cache()
        return arc_data

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Arc computation error: {str(e)}"
        )


ingestion_jobs: Dict[str, Any] = {}


def _execute_ingest_job(job_id: str, dataset_name: str, num_records: int):
    try:
        run_etl(dataset_name, num_records=num_records)
        ingestion_jobs[job_id]["status"] = "completed"
    except Exception as e:
        ingestion_jobs[job_id]["status"] = "failed"
        ingestion_jobs[job_id]["error"] = str(e)


@app.post("/api/pipeline/ingest")
def post_pipeline_ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    try:
        job_id = str(uuid.uuid4())[:8]
        ingestion_jobs[job_id] = {
            "job_id": job_id,
            "dataset_name": request.dataset_name,
            "status": "running",
            "error": None,
        }
        background_tasks.add_task(_execute_ingest_job, job_id, request.dataset_name, request.num_records)
        return {
            "status": "ingest_scheduled",
            "job_id": job_id,
            "dataset_name": request.dataset_name,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to schedule ingestion: {str(e)}"
        )


@app.get("/api/pipeline/ingest/status/{job_id}")
def get_ingest_status(job_id: str):
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return ingestion_jobs[job_id]


class AnalysisRequest(BaseModel):
    text: str
    lang: str = "auto"
    title: str = "Untitled"


_cached_dynamic_visual_baselines = {}

def compute_dynamic_baselines(lang: str):
    global _cached_dynamic_visual_baselines
    if lang in _cached_dynamic_visual_baselines:
        return _cached_dynamic_visual_baselines[lang]

    fallbacks = {
        "en": {
            "radar": {
                "web_novel":   [0.35, 0.55, 0.60, 0.40, 0.40, 0.65, 0.45, 0.50],
                "classic_lit": [0.30, 0.20, 0.40, 0.80, 0.70, 0.30, 0.80, 0.40],
            },
            "pacing": {
                "web_novel": [15, 8, 22, 5, 12, 30, 9, 14, 6, 18, 11, 25, 7, 16, 13, 8, 20, 10, 15, 6, 24, 9, 12, 17, 8, 14, 5, 21, 7, 13, 11, 28, 6, 15, 9, 19, 12, 10, 16, 8, 23, 7, 11, 14, 5, 17, 12, 9, 21, 6],
                "classic_lit": [85, 112, 64, 140, 95, 130, 78, 105, 88, 120, 90, 115, 130, 80, 125, 95, 110, 140, 75, 100, 105, 120, 85, 135, 95, 110, 130, 90, 115, 100, 112, 88, 98, 125, 70, 118, 132, 92, 104, 115],
            },
            "arc": {
                "web_novel":   [0.1620, 0.2450, 0.3800, 0.2200],
                "classic_lit": [0.2800, 0.3650, -0.1500, 0.3100],
            }
        },
        "ja": {
            "radar": {
                "web_novel":   [0.35, 0.50, 0.65, 0.40, 0.40, 0.65, 0.40, 0.60],
                "classic_lit": [0.50, 0.40, 0.70, 0.70, 0.50, 0.45, 0.45, 0.45],
            },
            "pacing": {
                "web_novel": [45, 28, 62, 18, 35, 80, 24, 40, 15, 55, 30, 72, 22, 45, 38, 26, 60, 32, 44, 20, 70, 28, 35, 50, 25, 40, 16, 65, 22, 38, 32, 84, 18, 44, 28, 55, 36, 30, 48, 25, 68, 20, 32, 42, 15, 50, 35, 26, 62, 18],
                "classic_lit": [180, 240, 150, 310, 200, 280, 160, 220, 190, 260, 175, 250, 290, 165, 270, 210, 230, 320, 155, 225, 210, 255, 185, 300, 205, 240, 275, 195, 265, 220, 245, 190, 215, 280, 140, 260, 295, 205, 235, 250],
            },
            "arc": {
                "web_novel":   [0.2100, 0.3100, 0.4200, 0.2900],
                "classic_lit": [0.1800, 0.2500, -0.2800, -0.1200],
            }
        },
        "zh": {
            "radar": {
                "web_novel":   [0.30, 0.50, 0.65, 0.40, 0.40, 0.65, 0.45, 0.60],
                "classic_lit": [0.50, 0.40, 0.80, 0.70, 0.50, 0.45, 0.45, 0.45],
            },
            "pacing": {
                "web_novel": [60, 35, 80, 22, 45, 100, 30, 50, 20, 70, 40, 90, 28, 55, 48, 32, 75, 40, 55, 25, 85, 35, 45, 60, 30, 50, 20, 80, 28, 48, 40, 105, 22, 55, 35, 70, 45, 38, 60, 30, 85, 25, 40, 52, 18, 65, 45, 32, 78, 22],
                "classic_lit": [150, 200, 130, 260, 170, 230, 140, 180, 160, 220, 150, 210, 240, 140, 230, 180, 190, 270, 130, 190, 180, 215, 155, 250, 175, 200, 230, 165, 220, 185, 205, 160, 180, 240, 120, 220, 250, 170, 200, 210],
            },
            "arc": {
                "web_novel":   [0.1800, 0.3800, 0.4900, 0.3400],
                "classic_lit": [0.2200, 0.3200, -0.2400, 0.1900],
            }
        }
    }

    import copy
    result = copy.deepcopy(fallbacks.get(lang, fallbacks["en"]))
    
    try:
        import random
        with Session(engine) as session:
            # Gutenberg Chapters for this language
            gutenberg_stmt = (
                select(Chapter)
                .join(Novel, Chapter.novel_id == Novel.id)
                .where(Novel.source == "gutenberg")
            )
            gutenberg_chapters = session.exec(gutenberg_stmt).all()
            
            # Webnovel Chapters for this language
            webnovel_stmt = (
                select(Chapter)
                .join(Novel, Chapter.novel_id == Novel.id)
                .where(Novel.source != "gutenberg")
            )
            webnovel_chapters = session.exec(webnovel_stmt).all()

            def filter_by_lang(chapters_list):
                res = []
                for ch in chapters_list:
                    if lang == "en" and ch.text_en:
                        res.append(ch)
                    elif lang == "ja" and ch.text_ja:
                        res.append(ch)
                    elif lang == "zh" and getattr(ch, "text_zh", ""):
                        res.append(ch)
                return res

            g_chs = filter_by_lang(gutenberg_chapters)
            w_chs = filter_by_lang(webnovel_chapters)

            g_chs = sorted(g_chs, key=lambda c: c.id)
            w_chs = sorted(w_chs, key=lambda c: c.id)

            rng = random.Random(42)
            if len(g_chs) > 30:
                g_chs = rng.sample(g_chs, 30)
            if len(w_chs) > 30:
                w_chs = rng.sample(w_chs, 30)

            def compute_full_novel_arcs_for_source(source_type: str) -> list[float]:
                if source_type == "gutenberg":
                    novels = session.exec(select(Novel).where(Novel.source == "gutenberg")).all()
                else:
                    novels = session.exec(select(Novel).where(Novel.source != "gutenberg")).all()

                arc_grid = [[] for _ in range(4)]
                for n in novels[:25]:
                    chs = session.exec(
                        select(Chapter)
                        .where(Chapter.novel_id == n.id)
                        .order_by(Chapter.chapter_number)
                    ).all()
                    
                    lang_texts = []
                    for ch in chs:
                        if lang == "en" and ch.text_en:
                            lang_texts.append(ch.text_en)
                        elif lang == "ja" and ch.text_ja:
                            lang_texts.append(ch.text_ja)
                        elif lang == "zh" and getattr(ch, "text_zh", ""):
                            lang_texts.append(getattr(ch, "text_zh", ""))
                    
                    if lang_texts:
                        full_text = "\n".join(lang_texts)
                        if lang == "en":
                            sents = [s.strip() for s in re.split(r'[.!?]+', full_text) if s.strip()]
                        else:
                            sents = [s.strip() for s in re.split(r'[。！？]+', full_text) if s.strip()]
                        if len(sents) >= 4:
                            arc_res = compute_kishotenketsu_quantile_arc(sents, lang)
                            for i, val in enumerate(arc_res["quantiles"]):
                                arc_grid[i].append(val)

                avg_arcs = [
                    round(sum(vals) / len(vals), 4) if vals else 0.0
                    for vals in arc_grid
                ]
                return avg_arcs if any(x != 0.0 for x in avg_arcs) else []

            def extract_baselines_from_sampled_chapters(sampled_chs, source_type):
                feats = []
                pacing_grid = [[] for _ in range(40)]
                
                # 1. Fast regex pacing across sampled chapters
                for ch in sampled_chs:
                    text = ch.text_en if lang == "en" else (ch.text_ja if lang == "ja" else getattr(ch, "text_zh", ""))
                    if text:
                        paras = split_paragraphs(text)
                        for i, p in enumerate(paras[:40]):
                            wc = len(re.findall(r'\b\w+\b', p)) if lang == "en" else len([c for c in p if not c.isspace()])
                            if wc > 0:
                                pacing_grid[i].append(wc)

                avg_pacings = [
                    int(round(sum(vals) / len(vals))) if vals else 20
                    for vals in pacing_grid
                ]

                avg_arcs = compute_full_novel_arcs_for_source(source_type)

                # 2. Heavy spaCy/NLP feature extraction on a small 5-chapter subset
                radar_chs = sampled_chs[:5]
                for ch in radar_chs:
                    row = {}
                    if lang == "en" and ch.text_en:
                        row = extract_english_features(ch.text_en)
                    elif lang == "ja" and ch.text_ja:
                        row = extract_japanese_features(ch.text_ja)
                    elif lang == "zh" and getattr(ch, "text_zh", ""):
                        row = extract_chinese_features(getattr(ch, "text_zh", ""))
                    if row:
                        feats.append(row)

                radar_vals = []
                keys_list = [
                    "theme_explication_ratio",
                    "linearity_subversion_score",
                    "sensory_body_density",
                    "outside_world_engagement",
                    "narrative_feature_diversity",
                    "dialogue_ratio",
                    "ttr",
                    "compound_sentiment"
                ]
                
                if feats:
                    for k in keys_list:
                        raw_vals = [f.get(k, 0) for f in feats if f.get(k) is not None]
                        avg_raw = sum(raw_vals) / len(raw_vals) if raw_vals else 0.0
                        norm_val = normalize_feature_percentile(k, avg_raw)
                        radar_vals.append(round(norm_val, 4))
                
                return radar_vals, (avg_pacings if avg_pacings else []), (avg_arcs if avg_arcs else [])

            if g_chs:
                g_radar, g_pacing, g_arc = extract_baselines_from_sampled_chapters(g_chs, "gutenberg")
                if g_radar:
                    result["radar"]["classic_lit"] = g_radar
                if g_pacing:
                    result["pacing"]["classic_lit"] = g_pacing
                # NOTE: Arc baselines are intentionally NOT overwritten here.
                # Averaging diverse novels' sentiment arcs produces flat ~0.20 curves
                # that destroy the distinctive archetypal shapes (e.g., classic lit's
                # dramatic negative Ten act). The curated fallback values above preserve
                # the characteristic narrative trajectories for each language.

            if w_chs:
                w_radar, w_pacing, w_arc = extract_baselines_from_sampled_chapters(w_chs, "web")
                if w_radar:
                    result["radar"]["web_novel"] = w_radar
                if w_pacing:
                    result["pacing"]["web_novel"] = w_pacing
                # Same rationale: keep curated arc baselines.
                
    except Exception as e:
        print(f"Error computing dynamic visual baselines: {e}")
        
    _cached_dynamic_visual_baselines[lang] = result
    return result


_cached_baselines = {}

FALLBACK_BASELINES = {
    "en": {
        "gutenberg": {
            "ttr": 0.339,
            "dialogue_ratio": 0.349,
            "avg_sentence_len": 11.4
        },
        "webnovel": {
            "ttr": 0.361,
            "dialogue_ratio": 0.223,
            "avg_sentence_len": 10.8
        }
    },
    "ja": {
        "gutenberg": {
            "ttr": 0.220,
            "dialogue_ratio": 0.150,
            "avg_sentence_len": 25.0
        },
        "webnovel": {
            "ttr": 0.288,
            "dialogue_ratio": 0.402,
            "avg_sentence_len": 35.7
        }
    },
    "zh": {
        "gutenberg": {
            "ttr": 0.007,
            "dialogue_ratio": 0.015,
            "avg_sentence_len": 13.5
        },
        "webnovel": {
            "ttr": 0.031,
            "dialogue_ratio": 0.294,
            "avg_sentence_len": 22.1
        }
    }
}


def get_baseline_stats(lang: str = "en"):
    global _cached_baselines
    if lang in _cached_baselines:
        return _cached_baselines[lang]

    fallbacks = FALLBACK_BASELINES.get(lang, FALLBACK_BASELINES["en"])

    try:
        with Session(engine) as session:
            count = session.exec(select(func.count(Chapter.id))).one()
            if count > 30:
                _cached_baselines[lang] = fallbacks
                return _cached_baselines[lang]

            chapters = session.exec(select(Chapter)).all()

            novels = session.exec(select(Novel)).all()
            novels_map = {n.id: n for n in novels}

            gutenberg_feats = []
            webnovel_feats = []

            for ch in chapters:
                novel = novels_map.get(ch.novel_id)
                if not novel:
                    continue

                if lang == "en" and ch.text_en:
                    f = extract_english_features(ch.text_en)
                elif lang == "ja" and ch.text_ja:
                    f = extract_japanese_features(ch.text_ja)
                elif lang == "zh" and ch.text_zh:
                    f = extract_chinese_features(ch.text_zh)
                else:
                    continue

                if novel.source == 'gutenberg':
                    gutenberg_feats.append(f)
                else:
                    webnovel_feats.append(f)

            def avg_dict(lst):
                if not lst:
                    return None
                keys = lst[0].keys()
                return {k: sum(d.get(k, 0) for d in lst) / len(lst) for k in keys}

            gutenberg_stats = avg_dict(gutenberg_feats)
            if not gutenberg_stats or "ttr" not in gutenberg_stats:
                gutenberg_stats = fallbacks["gutenberg"]

            webnovel_stats = avg_dict(webnovel_feats)
            if not webnovel_stats or "ttr" not in webnovel_stats:
                webnovel_stats = fallbacks["webnovel"]

            _cached_baselines[lang] = {
                "gutenberg": gutenberg_stats,
                "webnovel": webnovel_stats
            }
    except Exception as e:
        print(f"Error computing baselines from DB: {e}")

    if lang not in _cached_baselines:
        _cached_baselines[lang] = fallbacks

    return _cached_baselines[lang]


@app.post("/api/analyze")
def post_analyze(request: AnalysisRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty")

    lang = request.lang
    if lang == "auto":
        lang = detect_language(request.text)

    if lang == "en":
        features = extract_english_features(request.text)
    elif lang == "ja":
        features = extract_japanese_features(request.text)
    elif lang == "zh":
        features = extract_chinese_features(request.text)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")

    baselines = get_baseline_stats(lang)

    # 1. Compute paragraph pacing lengths for input text
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', request.text) if p.strip()]
    paragraph_lengths = []
    for p in paragraphs:
        if lang == "en":
            word_count = len(re.findall(r'\b\w+\b', p))
        else:
            word_count = len([c for c in p if not c.isspace()])
        if word_count > 0:
            paragraph_lengths.append(word_count)
    pacing = paragraph_lengths[:100]

    # 2. Format features for matcher and add pacing array
    agg = {f"{lang}_{k}": v for k, v in features.items()}
    agg["pacing"] = pacing

    normalized_radar = {}
    for k, v in features.items():
        if v is not None:
            normalized_radar[f"{lang}_{k}"] = normalize_feature_percentile(k, v)
    agg["normalized_radar"] = normalized_radar

    dyn_baselines = compute_dynamic_baselines(lang)
    agg["baselines"] = {
        "radar": dyn_baselines.get("radar", {}),
        "pacing": dyn_baselines.get("pacing", {}),
    }

    archetype = match_archetype(agg)
    # Semantic genre & territory matching is currently optimized for English text (all-MiniLM-L6-v2)
    semantic = match_semantic(request.text, features=features) if lang == "en" else None
    
    if semantic:
        agg["archetype_match"] = {
            "closest_trope": semantic["genre"],
            "territory": semantic["territory"],
            "confidence": semantic["genre_confidence"],
            "top_genres": [{"genre": x["genre"], "confidence": x["score"]} for x in semantic["genre_scores"]],
            "genre_scores": semantic["genre_scores"],
            "top_territories": [{"territory": x["territory"], "confidence": x["score"]} for x in semantic["territory_scores"]],
            "territory_confidence": semantic.get("territory_confidence"),
            "taxonomy": semantic.get("taxonomy"),
        }
        if "taxonomy" in semantic:
            agg["taxonomy"] = semantic["taxonomy"]
    else:
        agg["archetype_match"] = archetype

    # 3. Compute Kishōtenketsu 4-quantile sentiment arc
    if lang == "en":
        sents = [s.strip() for s in re.split(r'[.!?]+', request.text) if s.strip()]
    else:
        sents = [s.strip() for s in re.split(r'[。！？]+', request.text) if s.strip()]

    # For short samples (< 4 sentences), split by clauses so 4 distinct acts render gracefully
    if len(sents) < 4:
        clause_pat = r'[,;:\.!\?\-\n]+' if lang == "en" else r'[，、；：。！？\n]+'
        arc_units = [c.strip() for c in re.split(clause_pat, request.text) if len(c.strip()) > 3]
        if not arc_units:
            arc_units = sents
    else:
        arc_units = sents

    arc_res = compute_kishotenketsu_quantile_arc(arc_units, lang)

    # 4. Rhythmic Pacing (Paragraph / Sentence Barcode)
    paragraphs = split_paragraphs(request.text)
    pacing_bars = []
    for p in paragraphs:
        w_cnt = len(re.findall(r'\b\w+\b', p)) if lang == "en" else len([c for c in p if not c.isspace()])
        if w_cnt > 0:
            pacing_bars.append(w_cnt)

    if len(pacing_bars) < 8:
        # Short sample: generate sentence/clause-level word count bars so barcode populates 15-25 bars
        units = [u.strip() for u in re.split(r'[,;\.!\?\n]+', request.text) if u.strip()]
        unit_lengths = [len(re.findall(r'\b\w+\b', u)) if lang == "en" else len([c for c in u if not c.isspace()]) for u in units]
        unit_lengths = [l for l in unit_lengths if l > 0]
        if len(unit_lengths) >= len(pacing_bars):
            pacing_bars = unit_lengths

    agg["pacing"] = pacing_bars[:100]

    arc = {
        "title": request.title or "Untitled",
        "acts": arc_res["acts"],
        "quantiles": arc_res["quantiles"],
        "baselines": dyn_baselines.get("arc", {})
    }

    response = {
        "status": "success",
        "detected_lang": lang,
        "features": features,
        "archetype": {
            "archetype": semantic["genre"] if semantic else archetype["closest_trope"],
            "confidence": semantic["genre_confidence"] if semantic else archetype["confidence"],
            "description": f"Classification: {semantic['territory'] if semantic else archetype['territory']}. Semantically matched genre and territory.",
            "top_genres": [{"genre": x["genre"], "confidence": x["score"]} for x in semantic["genre_scores"][:3]] if semantic else [],
            "top_territories": [{"territory": x["territory"], "confidence": x["score"]} for x in semantic["territory_scores"][:3]] if semantic else []
        },
        "baselines": {
            "gutenberg": {
                "ttr": baselines["gutenberg"]["ttr"],
                "dialogue_ratio": baselines["gutenberg"]["dialogue_ratio"],
                "avg_sentence_len": baselines["gutenberg"]["avg_sentence_len"]
            },
            "webnovel": {
                "ttr": baselines["webnovel"]["ttr"],
                "dialogue_ratio": baselines["webnovel"]["dialogue_ratio"],
                "avg_sentence_len": baselines["webnovel"]["avg_sentence_len"]
            }
        },
        "stats": agg,
        "arc": arc,
        "top_matches": find_top_matches(agg, query_text=request.text, top_k=5)
    }
    if semantic is not None:
        response["semantic"] = semantic
    return response

