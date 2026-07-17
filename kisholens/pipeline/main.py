import re
import sys
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


def _get_or_create_novel(session: Session, title: str, author: str, source: str, cache: dict) -> int:
    novel_key = (title, author)
    if novel_key not in cache:
        statement = select(Novel).where(Novel.title == title, Novel.author == author)
        existing_novel = session.exec(statement).first()
        if existing_novel:
            cache[novel_key] = existing_novel.id
        else:
            novel = Novel(title=title, author=author, source=source)
            session.add(novel)
            session.commit()
            session.refresh(novel)
            cache[novel_key] = novel.id
            print(f"Added Novel: '{title}' by {author} (ID: {novel.id})")
    return cache[novel_key]

def run_etl(dataset_name: str, num_records: int = 20, max_chapters: int = 12, only_existing: bool = False):
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
            series_title, author, chapters = parse_gutenberg(raw_text)
            source = "gutenberg"
            
            with Session(engine) as session:
                if only_existing:
                    statement = select(Novel).where(Novel.title == series_title, Novel.author == author)
                    existing_novel = session.exec(statement).first()
                    if not existing_novel:
                        print(f"Skipping new Gutenberg novel: '{series_title}'")
                        return

                novel_id = _get_or_create_novel(session, series_title, author, source, novels_cache)
                
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
                        
                    novel_id = _get_or_create_novel(session, series_title, author, source, novels_cache)
                    
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

def verify_pipeline():
    """Runs a database verification check and prints active records and sample metrics."""
    engine = get_engine()
    with Session(engine) as session:
        novels = session.exec(select(Novel)).all()
        print(f"\nTotal novels in database: {len(novels)}")
        for n in novels:
            ch_count = session.exec(select(func.count(Chapter.id)).where(Chapter.novel_id == n.id)).one()
            print(f"  - [{n.id}] {n.title} (by {n.author}) [Source: {n.source}] - Chapters: {ch_count}")
            
        expected_sources = ["syosetu", "scribblehub", "royalroad", "gutenberg", "cnnovel"]
        actual_sources = set(n.source for n in novels)
        print("\nSource Verification Check:")
        for source in expected_sources:
            if source in actual_sources:
                print(f"  [OK] Successfully retrieved novels and chapters from source: '{source}'")
            else:
                print(f"  [FAIL] No data found in database for source: '{source}'")

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
        
    print("Ingesting more chapters for current novels (minimum 20 chapters per novel)...")
    
    # 1. Web Novels: scan a large window (1500 records) to find and ingest up to 20 chapters for all existing novels
    print("\n--- Ingesting up to 20 chapters for existing Web Novels ---")
    run_etl("NilanE/ParallelFiction-Ja_En-100k", 1500, max_chapters=20, only_existing=True)
    run_etl("botp/RyokoAI_ScribbleHub17K", 1500, max_chapters=20, only_existing=True)
    run_etl("OmniAICreator/RoyalRoad-1.61M", 1500, max_chapters=20, only_existing=True)
    run_etl("botp/RyokoAI_CNNovel125K", 1500, max_chapters=20, only_existing=True)
    
    # 2. Gutenberg Novels: run each Gutenberg novel with max_chapters=20 (to bring those under 20 up to 20 or their max)
    print("\n--- Ingesting up to 20 chapters for existing Gutenberg Novels ---")
    gutenberg_books = [
        "1342",  # Pride and Prejudice
        "23950", # 三國志演義
        "11",    # Alice's Adventures in Wonderland
        "84",    # Frankenstein
        "345",   # Dracula
        "1661",  # The Adventures of Sherlock Holmes
        "98",    # A Tale of Two Cities
        "174",   # The Picture of Dorian Gray
        "2701",  # Moby Dick
        "526",   # Heart of Darkness
        "35",    # The Time Machine
        "120",   # Treasure Island
        "1400",  # Great Expectations
        "121",   # Jane Eyre
        "768",   # Wuthering Heights
        "20",    # Twenty Thousand Leagues Under the Sea
        "113",   # The Secret Garden
        "158",   # Emma
        "153",   # The Tales of Mother Goose
        "209"    # The Turn of the Screw
    ]
    
    for book_id in gutenberg_books:
        run_etl(f"gutenberg/{book_id}", 20, max_chapters=20, only_existing=False)
        
    # Run database verification
    verify_pipeline()

if __name__ == "__main__":
    main()
