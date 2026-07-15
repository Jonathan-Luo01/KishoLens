import re
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
)

app = FastAPI(title="KishoLens API")

# Add CORS middleware so the Astro frontend can fetch data
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321"],
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
        with Session(engine) as session:
            stmt = (
                select(Novel, func.count(Chapter.id).label("chapter_count"))
                .outerjoin(Chapter, Novel.id == Chapter.novel_id)
                .group_by(Novel.id)
            )
            results = session.exec(stmt).all()

            return [
                {
                    "id": novel.id,
                    "title": novel.title,
                    "author": novel.author,
                    "source": novel.source,
                    "chapter_count": chapter_count,
                }
                for novel, chapter_count in results
            ]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database connection error: {str(e)}"
        )


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
                    agg[k] = sum(vals) / len(vals) if vals else None
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


@app.get("/api/novels/{novel_id}/stats")
def get_novel_stats(novel_id: int):
    try:
        with Session(engine) as session:
            novel = session.get(Novel, novel_id)
            if not novel:
                raise HTTPException(status_code=404, detail="Novel not found")

            chapters = session.exec(
                select(Chapter)
                .where(Chapter.novel_id == novel_id)
            ).all()

            features_list = []
            for ch in chapters:
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

            if agg:
                agg["archetype_match"] = match_archetype(agg)

            return agg
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Database connection error: {str(e)}"
        )


@app.post("/api/pipeline/ingest")
def post_pipeline_ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_etl, request.dataset_name, request.num_records)
        return {"status": "ingest_scheduled", "dataset_name": request.dataset_name}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to schedule ingestion: {str(e)}"
        )


class AnalysisRequest(BaseModel):
    text: str
    lang: str = "auto"
    title: str = "Untitled"


def detect_language(text: str) -> str:
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


_cached_baselines = None


def get_baseline_stats():
    global _cached_baselines
    if _cached_baselines is not None:
        return _cached_baselines

    try:
        with Session(engine) as session:
            chapters = session.exec(select(Chapter)).all()
            novels = session.exec(select(Novel)).all()
            novels_map = {n.id: n for n in novels}

            gutenberg_feats = []
            webnovel_feats = []

            for ch in chapters:
                novel = novels_map.get(ch.novel_id)
                if not novel:
                    continue

                if ch.text_en:
                    f = extract_english_features(ch.text_en)
                elif ch.text_ja:
                    f = extract_japanese_features(ch.text_ja)
                elif ch.text_zh:
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

            _cached_baselines = {
                "gutenberg": avg_dict(gutenberg_feats),
                "webnovel": avg_dict(webnovel_feats)
            }
    except Exception as e:
        print(f"Error computing baselines from DB: {e}")

    # Fallback defaults in case DB query is empty/fails
    if not _cached_baselines or not _cached_baselines["gutenberg"] or not _cached_baselines["webnovel"]:
        _cached_baselines = {
            "gutenberg": {
                "ttr": 0.173,
                "dialogue_ratio": 0.182,
                "avg_sentence_len": 12.464
            },
            "webnovel": {
                "ttr": 0.306,
                "dialogue_ratio": 0.235,
                "avg_sentence_len": 12.699
            }
        }
    return _cached_baselines


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

    baselines = get_baseline_stats()

    # format features for matcher
    agg = {f"{lang}_{k}": v for k, v in features.items()}
    archetype = match_archetype(agg)

    return {
        "status": "success",
        "detected_lang": lang,
        "features": features,
        "archetype": {
            "archetype": archetype["closest_trope"],
            "confidence": archetype["confidence"],
            "description": f"Classification: {archetype['territory']}. Closest matched writing archetype based on stylistic features."
        },
        "baselines": {
            "gutenberg": {
                "ttr": baselines["gutenberg"].get("ttr", 0.173) if baselines["gutenberg"] else 0.173,
                "dialogue_ratio": baselines["gutenberg"].get("dialogue_ratio", 0.182) if baselines["gutenberg"] else 0.182,
                "avg_sentence_len": baselines["gutenberg"].get("avg_sentence_len", 12.464) if baselines["gutenberg"] else 12.464
            },
            "webnovel": {
                "ttr": baselines["webnovel"].get("ttr", 0.306) if baselines["webnovel"] else 0.306,
                "dialogue_ratio": baselines["webnovel"].get("dialogue_ratio", 0.235) if baselines["webnovel"] else 0.235,
                "avg_sentence_len": baselines["webnovel"].get("avg_sentence_len", 12.699) if baselines["webnovel"] else 12.699
            }
        }
    }

