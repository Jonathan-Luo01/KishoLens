from typing import Optional
from datasets import load_dataset
from sqlmodel import SQLModel, Session, select, func

from kisholens.models import Novel, Chapter, get_engine
from kisholens.pipeline.normalization import clean_html, clean_japanese, clean_english
from kisholens.pipeline.scraping import DATASET_REGISTRY, download_gutenberg, parse_gutenberg
from kisholens.ml.features import (
    extract_english_features,
    extract_japanese_features,
    extract_chinese_features
)
from kisholens.ml.build_centroids import consolidate_genre

def extract_features(text: str, lang: str = "en"):
    """Computes baseline preview features by delegating to the unified extractors in ml.features."""
    if lang == "en":
        f = extract_english_features(text)
        return {
            "token_count": f.get("word_count", 0),
            "sentence_count": f.get("sentence_count", 0),
            "punctuation_density": f.get("punc_density", 0.0),
            "dialogue_ratio": f.get("dialogue_ratio", 0.0)
        }
    elif lang == "ja":
        f = extract_japanese_features(text)
        return {
            "token_count": f.get("char_count", 0),
            "sentence_count": f.get("sentence_count", 0),
            "punctuation_density": f.get("punc_density", 0.0),
            "dialogue_ratio": f.get("dialogue_ratio", 0.0)
        }
    else:  # zh
        f = extract_chinese_features(text)
        return {
            "token_count": f.get("char_count", 0),
            "sentence_count": f.get("sentence_count", 0),
            "punctuation_density": f.get("punc_density", 0.0),
            "dialogue_ratio": f.get("dialogue_ratio", 0.0)
        }


from kisholens.pipeline.disambiguation import disambiguate_and_rank_genres

def _get_or_create_novel(session: Session, title: str, author: str, source: str, cache: dict, genre: Optional[str] = None, territory: Optional[str] = None) -> int:
    clean_title = (title or "").strip()
    clean_author = (author or "").strip()
    novel_key = (clean_title, clean_author)
    if novel_key not in cache:
        statement = select(Novel).where(func.lower(func.trim(Novel.title)) == clean_title.lower(), func.lower(func.trim(Novel.author)) == clean_author.lower())
        existing_novel = session.exec(statement).first()
        
        effective_genre = genre
        if not effective_genre:
            dis_res = disambiguate_and_rank_genres(tags_str=clean_title, source=source, territory=territory or "")
            effective_genre = dis_res["parent_genre_str"]

        if existing_novel:
            updated = False
            if effective_genre and not existing_novel.genre:
                existing_novel.genre = effective_genre
                updated = True
            if territory and not existing_novel.territory:
                existing_novel.territory = territory
                updated = True
            if updated:
                session.add(existing_novel)
                session.commit()
            cache[novel_key] = existing_novel.id
        else:
            default_territory = territory or ("Classic Literature Territory" if source.lower() == "gutenberg" else "Web Novel Territory")
            novel = Novel(title=clean_title, author=clean_author, source=source, genre=effective_genre, territory=default_territory)
            session.add(novel)
            session.commit()
            session.refresh(novel)
            cache[novel_key] = novel.id
            print(f"Added Novel: '{clean_title}' by {clean_author} (ID: {novel.id}, Genre: {effective_genre}, Territory: {default_territory})")
    return cache[novel_key]

def run_etl(dataset_name: str, num_records: int = 20, max_chapters: int = 12, only_existing: bool = False, genre: Optional[str] = None, territory: Optional[str] = None):
    """Orchestrates ingestion of a dataset/book into SQLite with chapter limits and optional existing-novel filters."""
    engine = get_engine()
    
    try:
        # Ensure schema tables are created
        SQLModel.metadata.create_all(engine)
        novels_cache = {}
        
        # Handle Project Gutenberg path
        if dataset_name.startswith("gutenberg/"):
            book_id = dataset_name.split("/")[1]
            print(f"\nDownloading Gutenberg book ID: {book_id}...")
            raw_text = download_gutenberg(book_id)
            series_title, author, raw_tags, chapters = parse_gutenberg(raw_text)
            source = "gutenberg"
            detected_genre = consolidate_genre(raw_tags)
            final_genre = genre or detected_genre or "Drama"
            final_territory = territory or "Classic Literature Territory"
            
            with Session(engine) as session:
                if only_existing:
                    statement = select(Novel).where(Novel.title == series_title, Novel.author == author)
                    existing_novel = session.exec(statement).first()
                    if not existing_novel:
                        print(f"Skipping new Gutenberg novel: '{series_title}'")
                        return

                novel_id = _get_or_create_novel(
                    session, series_title, author, source, novels_cache,
                    genre=final_genre, territory=final_territory
                )
                
                ch_count = session.exec(select(func.count(Chapter.id)).where(Chapter.novel_id == novel_id)).one()
                chapters_to_ingest = []
                for ch_item in chapters:
                    if ch_count >= max_chapters:
                        break
                    
                    chapter_number = ch_item["chapter_number"]
                    statement_ch = select(Chapter).where(Chapter.novel_id == novel_id, Chapter.chapter_number == chapter_number)
                    existing_chapter = session.exec(statement_ch).first()
                    if not existing_chapter:
                        chapters_to_ingest.append(ch_item)
                        ch_count += 1
                
                # Limit to num_records if applicable
                chapters_to_ingest = chapters_to_ingest[:num_records]
                
                for ch_item in chapters_to_ingest:
                    chapter_number = ch_item["chapter_number"]
                    chapter_title = ch_item["chapter_title"]
                    ch_lang = ch_item.get("lang", "en")
                    
                    raw_body = clean_html(ch_item["text"])
                    if ch_lang == "ja":
                        cleaned_text = clean_japanese(raw_body)
                    elif ch_lang == "en":
                        cleaned_text = clean_english(raw_body)
                    else:
                        cleaned_text = raw_body.strip()
                    
                    chapter = Chapter(
                        novel_id=novel_id,
                        chapter_number=chapter_number,
                        title=chapter_title,
                        text_ja=cleaned_text if ch_lang == "ja" else "",
                        text_en=cleaned_text if ch_lang == "en" else "",
                        text_zh=cleaned_text if ch_lang == "zh" else ""
                    )
                    session.add(chapter)
                    session.commit()
                    print(f"  Ingested Chapter {chapter_number}: {chapter_title}")
                    
                    if cleaned_text:
                        feat = extract_features(cleaned_text, lang=ch_lang)
                        print(f"    Features ({ch_lang.upper()}): Tokens={feat['token_count']}, Sentences={feat['sentence_count']}, PuncDensity={feat['punctuation_density']:.3f}, DialogueRatio={feat['dialogue_ratio']:.3f}")
        
        # Handle Hugging Face datasets path
        else:
            print(f"\nLoading {dataset_name} in streaming mode...")
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            iterator = iter(dataset)
            extractor = DATASET_REGISTRY[dataset_name]["extractor"]
            
            with Session(engine) as session:
                ingested_count = 0
                # Process enough records to find chapters for our novels
                for idx in range(num_records):
                    try:
                        item = next(iterator)
                    except StopIteration:
                        print("Stream ran dry early.")
                        break
                    
                    parsed = extractor(item, idx)
                    series_title = parsed["series_title"]
                    author = parsed["author"]
                    source = parsed["source"]
                    chapter_number = parsed["chapter_number"]
                    chapter_title = parsed["chapter_title"]
                    cleaned_ja = parsed.get("text_ja", "")
                    cleaned_en = parsed.get("text_en", "")
                    cleaned_zh = parsed.get("text_zh", "")
                    
                    # Verify if it belongs to an existing novel
                    statement = select(Novel).where(Novel.title == series_title, Novel.author == author)
                    existing_novel = session.exec(statement).first()
                    
                    if only_existing and not existing_novel:
                        continue
                        
                    novel_id = _get_or_create_novel(
                        session, series_title, author, source, novels_cache,
                        genre=genre, territory=territory
                    )
                    
                    # Check chapter count
                    ch_count = session.exec(select(func.count(Chapter.id)).where(Chapter.novel_id == novel_id)).one()
                    if ch_count >= max_chapters:
                        continue
                        
                    if chapter_number is None:
                        max_ch = session.exec(select(func.max(Chapter.chapter_number)).where(Chapter.novel_id == novel_id)).one()
                        chapter_number = (max_ch or 0) + 1
                    
                    if chapter_title in ("Chapter", "Chapter "):
                        chapter_title = f"Chapter {chapter_number}"
                    
                    statement_ch = select(Chapter).where(Chapter.novel_id == novel_id, Chapter.chapter_number == chapter_number)
                    existing_chapter = session.exec(statement_ch).first()
                    if not existing_chapter:
                        chapter = Chapter(
                            novel_id=novel_id,
                            chapter_number=chapter_number,
                            title=chapter_title,
                            text_ja=cleaned_ja,
                            text_en=cleaned_en,
                            text_zh=cleaned_zh
                        )
                        session.add(chapter)
                        session.commit()
                        print(f"  Ingested Chapter {chapter_number}: {chapter_title} (Novel ID: {novel_id})")
                        ingested_count += 1
                        
                        if cleaned_ja:
                            feat_ja = extract_features(cleaned_ja, lang="ja")
                            print(f"    Features (JA): Tokens={feat_ja['token_count']}, Sentences={feat_ja['sentence_count']}, PuncDensity={feat_ja['punctuation_density']:.3f}, DialogueRatio={feat_ja['dialogue_ratio']:.3f}")
                        if cleaned_en:
                            feat_en = extract_features(cleaned_en, lang="en")
                            print(f"    Features (EN): Tokens={feat_en['token_count']}, Sentences={feat_en['sentence_count']}, PuncDensity={feat_en['punctuation_density']:.3f}, DialogueRatio={feat_en['dialogue_ratio']:.3f}")
                        if cleaned_zh:
                            feat_zh = extract_features(cleaned_zh, lang="zh")
                            print(f"    Features (ZH): Tokens={feat_zh['token_count']}, Sentences={feat_zh['sentence_count']}, PuncDensity={feat_zh['punctuation_density']:.3f}, DialogueRatio={feat_zh['dialogue_ratio']:.3f}")
                    else:
                        print(f"  Chapter {chapter_number} already ingested for Novel ID: {novel_id}. Skipping.")
    finally:
        engine.dispose()
        print(f"ETL run for {dataset_name} completed.")

def fetch_gutenberg_book_ids_by_topic(topic: str, limit: int = 20) -> list[str]:
    """Fetch up to limit Project Gutenberg book IDs from gutendex for a topic."""
    import urllib.request
    import urllib.parse
    import json
    import time
    import ssl

    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    
    book_ids = []
    base_url = "https://gutendex.com/books"
    params = urllib.parse.urlencode({
        "topic": topic,
        "languages": "en",
    })
    url = f"{base_url}?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    )
    
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                for book in results:
                    if len(book_ids) >= limit:
                        break
                    book_ids.append(str(book["id"]))
                break
        except Exception as e:
            print(f"[WARN] Gutenberg ID fetch attempt {attempt+1} failed for topic {topic}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
            
    return book_ids

def verify_pipeline():
    """Runs a database verification check and prints active records and sample metrics."""
    engine = get_engine()
    with Session(engine) as session:
        novels = session.exec(select(Novel)).all()
        print(f"\nTotal novels in database: {len(novels)}")
        
        # Source Distribution
        print("\nSource Distribution:")
        res_source = session.execute(select(Novel.source, func.count(Novel.id)).group_by(Novel.source)).all()
        for source, count in res_source:
            print(f"  - {source}: {count} novels")
            
        # Genre Distribution
        print("\nGenre Distribution:")
        res_genre = session.execute(select(Novel.genre, func.count(Novel.id)).group_by(Novel.genre)).all()
        for genre, count in sorted(res_genre, key=lambda x: x[1], reverse=True):
            print(f"  - {genre or 'Unclassified'}: {count} novels")

DATASET_ALIASES = {
    "syosetu": "NilanE/ParallelFiction-Ja_En-100k",
    "parallelfiction": "NilanE/ParallelFiction-Ja_En-100k",
    "parallel-fiction": "NilanE/ParallelFiction-Ja_En-100k",
    "scribblehub": "botp/RyokoAI_ScribbleHub17K",
    "royalroad": "OmniAICreator/RoyalRoad-1.61M",
    "cnnovel": "botp/RyokoAI_CNNovel125K",
}

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    import argparse
    parser = argparse.ArgumentParser(description="Generic KishoLens Dataset Ingestion Pipeline")
    parser.add_argument("--dataset", type=str, help="Dataset name or alias (e.g. syosetu, scribblehub, royalroad, cnnovel, gutenberg/1234)")
    parser.add_argument("--count", type=int, default=20, help="Number of records/novels to ingest")
    parser.add_argument("--genre", type=str, help="Optional genre filter/assignment")
    parser.add_argument("--rebuild-centroids", action="store_true", help="Rebuild ML centroids after ingestion")
    args = parser.parse_args()

    if args.dataset:
        ds_name = DATASET_ALIASES.get(args.dataset.lower(), args.dataset)
        print(f"\nIngesting dataset '{ds_name}' (Target count: {args.count})...")
        run_etl(ds_name, num_records=args.count, max_chapters=5, genre=args.genre)
        if args.rebuild_centroids:
            import os
            os.system("uv run python scratch/build_centroids_db.py")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    engine = get_engine()
    
    # 1. Ingest Classic Gutenberg Books for Classic Literature Territory
    classic_genres_topics = {
        "Action / Adventure": "adventure",
        "Comedy": "humor",
        "Drama": "plays",
        "Fantasy": "fairy tales",
        "Horror": "horror",
        "Historical": "history",
        "Sci-Fi": "science fiction",
        "Philosophy": "philosophy",
        "Mystery": "detective",
        "Tragedy": "tragedy",
        "Supernatural": "gothic fiction",
        "Poetry": "poetry",
        "Romance": "romance",
    }
    
    for genre, topic in classic_genres_topics.items():
        # Check current count of Gutenberg books for this genre
        with Session(engine) as s:
            current_count = s.exec(select(func.count(Novel.id)).where(Novel.source == "gutenberg", Novel.genre == genre)).one()
        
        needed = 20 - current_count
        if needed <= 0:
            print(f"Already have {current_count} books in database for classic genre: '{genre}'")
            continue
            
        print(f"Fetching {needed} book IDs from Project Gutenberg for topic: '{topic}'...")
        book_ids = fetch_gutenberg_book_ids_by_topic(topic, limit=needed)
        print(f"Found IDs: {book_ids}")
        for book_id in book_ids:
            run_etl(f"gutenberg/{book_id}", num_records=5, max_chapters=5, only_existing=False, genre=genre, territory="Classic Literature Territory")

    # 2. Ingest Web Novels for Web/Traditional Territories
    print("\n--- Ingesting Web Novels for Web/Traditional Territories (Stratified Sampling) ---")
    web_and_trad_genres = {
        "Action / Adventure", "Comedy", "Romance", "Drama", "Fantasy",
        "Horror", "Historical", "Sci-Fi", "Slice of Life", "Cultivation",
        "Tragedy", "Isekai", "Supernatural"
    }
    
    # Count current books per genre in database
    genre_counts = {g: 0 for g in web_and_trad_genres}
    with Session(engine) as s:
        res = s.execute(select(Novel.genre, func.count(Novel.id)).group_by(Novel.genre)).all()
        for g, count in res:
            if g in genre_counts:
                genre_counts[g] = count
                
    print(f"Current web/traditional database counts: {genre_counts}")
    
    # Check if we need more
    needed_genres = {g for g in web_and_trad_genres if genre_counts[g] < 20}
    if not needed_genres:
        print("All web/traditional genres already have at least 20 novels in the database.")
    else:
        print(f"Genres needing more novels (under 20): {needed_genres}")
        
        # Stream ScribbleHub and RoyalRoad to fill in under-represented genres
        datasets_to_stream = [
            ("botp/RyokoAI_ScribbleHub17K", "scribblehub"),
            ("OmniAICreator/RoyalRoad-1.61M", "royalroad")
        ]
        
        for dataset_name, source_type in datasets_to_stream:
            if not needed_genres:
                break
                
            print(f"\nStreaming {dataset_name} to satisfy stratified sampling...")
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            extractor = DATASET_REGISTRY[dataset_name]["extractor"]
            
            with Session(engine) as session:
                scan_limit = 100000
                for idx, item in enumerate(dataset):
                    if idx >= scan_limit:
                        print("Scan limit reached for dataset stream.")
                        break
                    if not needed_genres:
                        break
                        
                    parsed = extractor(item, idx)
                    series_title = parsed["series_title"]
                    author = parsed["author"]
                    
                    # Extract tags to consolidate genre
                    raw_tags = item.get("tags", []) or []
                    if not raw_tags and "meta" in item and isinstance(item["meta"], dict):
                        raw_tags = item["meta"].get("tags", []) or []
                    if isinstance(raw_tags, str):
                        raw_tags = [t.strip() for t in raw_tags.split(",")]
                        
                    genre = consolidate_genre(raw_tags)
                    if genre in needed_genres:
                        # Double check we don't already have this novel
                        statement = select(Novel).where(Novel.title == series_title, Novel.author == author)
                        existing_novel = session.exec(statement).first()
                        if not existing_novel:
                            # Ingest this novel!
                            novel = Novel(
                                title=series_title,
                                author=author,
                                source=source_type,
                                genre=genre,
                                territory="Classic Literature Territory" if source_type.lower() == "gutenberg" else "Web Novel Territory"
                            )
                            session.add(novel)
                            session.commit()
                            session.refresh(novel)
                            print(f"Added Novel: '{series_title}' by {author} (Genre: {genre})")
                            
                            # Ingest first 5 chapters
                            cleaned_en = parsed.get("text_en", "")
                            if cleaned_en:
                                chapter = Chapter(
                                    novel_id=novel.id,
                                    chapter_number=1,
                                    title=parsed.get("chapter_title", "Chapter 1"),
                                    text_en=cleaned_en,
                                    text_ja="",
                                    text_zh=""
                                )
                                session.add(chapter)
                                session.commit()
                                print(f"  Ingested Chapter 1 for Novel ID: {novel.id}")
                                
                            # Increment count and update needed
                            genre_counts[genre] += 1
                            if genre_counts[genre] >= 20:
                                needed_genres.remove(genre)
                                print(f"Genre '{genre}' has successfully reached 20 novels.")
                                
    # Run database verification
    verify_pipeline()

    # Explicitly exit to prevent hanging from non-daemon background threads in datasets/huggingface
    import os
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
