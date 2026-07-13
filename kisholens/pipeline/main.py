import re
import sys
from typing import Optional
from datasets import load_dataset
from sqlmodel import SQLModel, Session, select, func

from kisholens.models import Novel, Chapter, get_engine
from kisholens.pipeline.normalization import clean_html, clean_japanese, clean_english
from kisholens.pipeline.scraping import DATASET_REGISTRY, download_gutenberg, parse_gutenberg

def extract_features(text: str, lang: str = "en"):
    """Computes baseline preview features: token counts, sentence counts, punctuation density, and dialogue ratios."""
    if not text:
        return {"token_count": 0, "sentence_count": 0, "punctuation_density": 0.0, "dialogue_ratio": 0.0}
        
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    char_count = len(text)
    
    if lang == "en":
        tokens = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)
        punctuations = re.findall(r'[.,\/#!$%\^&\*;:{}=\-_`~()?\"\']', text)
        dialogue_start = ('"', "'", '“', '”')
    else:  # ja or zh
        tokens = [c for c in text if not c.isspace()]
        sentences = re.split(r'[。！？\n]+', text)
        punc_pat = r'[，、。！？；：\"\"‘’（）《》【】『』「」——……]' if lang == "zh" else r'[、。！？「」『』（）―…ー・]'
        punctuations = re.findall(punc_pat, text)
        dialogue_start = ('“', '「', '『') if lang == "zh" else ('「', '『')
        
    sentence_count = len([s for s in sentences if s.strip()])
    dialogue_lines = [l for l in lines if l.startswith(dialogue_start)]
    
    return {
        "token_count": len(tokens),
        "sentence_count": sentence_count,
        "punctuation_density": len(punctuations) / char_count if char_count > 0 else 0.0,
        "dialogue_ratio": len(dialogue_lines) / len(lines) if lines else 0.0
    }

def run_etl(dataset_name: str, num_records: int = 20):
    """Orchestrates ingestion of a dataset/book into SQLite."""
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
            chapters_to_ingest = chapters[:num_records]
            
            with Session(engine) as session:
                novel_key = (series_title, author)
                if novel_key not in novels_cache:
                    statement = select(Novel).where(Novel.title == series_title, Novel.author == author)
                    existing_novel = session.exec(statement).first()
                    if existing_novel:
                        novels_cache[novel_key] = existing_novel.id
                    else:
                        novel = Novel(title=series_title, author=author, source=source)
                        session.add(novel)
                        session.commit()
                        session.refresh(novel)
                        novels_cache[novel_key] = novel.id
                        print(f"Added Novel: '{series_title}' by {author} (ID: {novel.id})")
                
                novel_id = novels_cache[novel_key]
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
                    
                    statement_ch = select(Chapter).where(Chapter.novel_id == novel_id, Chapter.chapter_number == chapter_number)
                    existing_chapter = session.exec(statement_ch).first()
                    if not existing_chapter:
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
                    else:
                        print(f"  Chapter {chapter_number} already ingested. Skipping DB insertion.")
                        
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
                    
                    novel_key = (series_title, author)
                    if novel_key not in novels_cache:
                        statement = select(Novel).where(Novel.title == series_title, Novel.author == author)
                        existing_novel = session.exec(statement).first()
                        if existing_novel:
                            novels_cache[novel_key] = existing_novel.id
                        else:
                            novel = Novel(title=series_title, author=author, source=source)
                            session.add(novel)
                            session.commit()
                            session.refresh(novel)
                            novels_cache[novel_key] = novel.id
                            print(f"Added Novel: '{series_title}' by {author} (ID: {novel.id})")
                        
                    novel_id = novels_cache[novel_key]
                    
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
                        print(f"  Ingested Chapter {chapter_number}: {chapter_title}")
                    else:
                        print(f"  Chapter {chapter_number} already ingested. Skipping DB insertion.")
                    
                    if cleaned_ja:
                        feat_ja = extract_features(cleaned_ja, lang="ja")
                        print(f"    Features (JA): Tokens={feat_ja['token_count']}, Sentences={feat_ja['sentence_count']}, PuncDensity={feat_ja['punctuation_density']:.3f}, DialogueRatio={feat_ja['dialogue_ratio']:.3f}")
                    if cleaned_en:
                        feat_en = extract_features(cleaned_en, lang="en")
                        print(f"    Features (EN): Tokens={feat_en['token_count']}, Sentences={feat_en['sentence_count']}, PuncDensity={feat_en['punctuation_density']:.3f}, DialogueRatio={feat_en['dialogue_ratio']:.3f}")
                    if cleaned_zh:
                        feat_zh = extract_features(cleaned_zh, lang="zh")
                        print(f"    Features (ZH): Tokens={feat_zh['token_count']}, Sentences={feat_zh['sentence_count']}, PuncDensity={feat_zh['punctuation_density']:.3f}, DialogueRatio={feat_zh['dialogue_ratio']:.3f}")
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
            print(f"  - [{n.id}] {n.title} (by {n.author}) [Source: {n.source}]")
            
        expected_sources = ["syosetu", "scribblehub", "royalroad", "gutenberg", "cnnovel"]
        actual_sources = set(n.source for n in novels)
        print("\nSource Verification Check:")
        for source in expected_sources:
            if source in actual_sources:
                print(f"  [OK] Successfully retrieved novels and chapters from source: '{source}'")
            else:
                print(f"  [FAIL] No data found in database for source: '{source}'")
                
        print("\nVerifying chapter content samples and metrics from different sources:")
        for source in expected_sources:
            print(f"\nChapters from source: {source}")
            novel_ids = [n.id for n in novels if n.source == source]
            if novel_ids:
                chapters = session.exec(select(Chapter).where(Chapter.novel_id.in_(novel_ids)).limit(2)).all()
                for c in chapters:
                    print(f"  - [{c.id}] Chapter {c.chapter_number}: {c.title} (Novel ID: {c.novel_id})")
                    if c.text_ja:
                        print(f"    JA text (first 100 chars): {c.text_ja[:100]}...")
                        feat_ja = extract_features(c.text_ja, lang="ja")
                        print(f"    JA Metrics: Tokens={feat_ja['token_count']}, Sentences={feat_ja['sentence_count']}, PuncDensity={feat_ja['punctuation_density']:.3f}, DialogueRatio={feat_ja['dialogue_ratio']:.3f}")
                    if c.text_en:
                        print(f"    EN text (first 100 chars): {c.text_en[:100]}...")
                        feat_en = extract_features(c.text_en, lang="en")
                        print(f"    EN Metrics: Tokens={feat_en['token_count']}, Sentences={feat_en['sentence_count']}, PuncDensity={feat_en['punctuation_density']:.3f}, DialogueRatio={feat_en['dialogue_ratio']:.3f}")
                    if c.text_zh:
                        print(f"    ZH text (first 100 chars): {c.text_zh[:100]}...")
                        feat_zh = extract_features(c.text_zh, lang="zh")
                        print(f"    ZH Metrics: Tokens={feat_zh['token_count']}, Sentences={feat_zh['sentence_count']}, PuncDensity={feat_zh['punctuation_density']:.3f}, DialogueRatio={feat_zh['dialogue_ratio']:.3f}")
            else:
                print("  No novels found for this source.")
    engine.dispose()

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
        
    print("Initializing KishoLens Ingestion Pipeline...")
    
    # Run the ETL tasks for our prototype sources
    run_etl("NilanE/ParallelFiction-Ja_En-100k", 5)
    run_etl("botp/RyokoAI_ScribbleHub17K", 5)
    run_etl("OmniAICreator/RoyalRoad-1.61M", 5)
    run_etl("gutenberg/1342", 5)
    run_etl("gutenberg/23950", 3)
    run_etl("botp/RyokoAI_CNNovel125K", 5)
    
    # Run the end-to-end database verification query
    verify_pipeline()

if __name__ == "__main__":
    main()
