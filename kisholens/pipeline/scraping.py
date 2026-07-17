import re
import asyncio
import aiohttp
from typing import Dict, Any, List, Tuple
from kisholens.pipeline.normalization import (
    clean_html,
    clean_japanese,
    clean_english,
    parse_chapter_number,
    extract_chapter_info
)

# Hugging Face Registry Parsers

def parse_parallel_fiction(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Extracts, cleans, and packages fields for ParallelFiction dataset."""
    meta = item.get('meta', {})
    series_title_eng = meta.get('general', {}).get('series_title_eng', 'Unknown')
    writer = meta.get('syosetu', {}).get('writer', 'Unknown')
    source = "syosetu"
    
    chapter_number, title_ja, title_en, body_ja, body_en = extract_chapter_info(item['src'], item['trg'])
    cleaned_ja = clean_japanese(clean_html(body_ja))
    cleaned_en = clean_english(clean_html(body_en))
    
    return {
        "series_title": series_title_eng,
        "author": writer,
        "source": source,
        "chapter_number": chapter_number,
        "chapter_title": title_en,
        "text_ja": cleaned_ja,
        "text_en": cleaned_en,
        "text_zh": ""
    }

def parse_scribblehub(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Extracts, cleans, and packages fields for ScribbleHub dataset (monolingual English)."""
    meta = item.get('meta', {})
    title_str = meta.get('title', 'Unknown')
    
    parts = title_str.split(" - ", 1)
    series_title, chapter_title = (parts[0].strip(), parts[1].strip()) if len(parts) > 1 else (title_str, title_str)
        
    author = meta.get('author', 'Unknown')
    source = "scribblehub"
    chapter_number = parse_chapter_number(chapter_title)
    cleaned_en = clean_english(clean_html(item.get('text', '')))
    
    return {
        "series_title": series_title,
        "author": author,
        "source": source,
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "text_ja": "",
        "text_en": cleaned_en,
        "text_zh": ""
    }

def parse_royalroad(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Extracts, cleans, and packages fields for RoyalRoad dataset (monolingual English)."""
    series_title = item.get('title', 'Unknown')
    author = item.get('author', 'Unknown')
    source = "royalroad"
    chapter_title = item.get('chapter_title', 'Unknown')
    chapter_number = parse_chapter_number(chapter_title)
    cleaned_en = clean_english(clean_html(item.get('text', '')))
    
    return {
        "series_title": series_title,
        "author": author,
        "source": source,
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "text_ja": "",
        "text_en": cleaned_en,
        "text_zh": ""
    }

def parse_cnnovel(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Extracts, cleans, and packages fields for RyokoAI_CNNovel125K dataset (monolingual Chinese)."""
    meta = item.get('meta', {})
    title_str = meta.get('title', 'Unknown')
    
    parts = title_str.split(" - ", 1)
    series_title, chapter_title = (parts[0].strip(), parts[1].strip()) if len(parts) > 1 else (title_str, title_str)
        
    author = meta.get('author', 'Unknown')
    source = "cnnovel"
    chapter_number = parse_chapter_number(chapter_title)
    cleaned_zh = clean_html(item.get('text', ''))
    
    return {
        "series_title": series_title,
        "author": author,
        "source": source,
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "text_ja": "",
        "text_en": "",
        "text_zh": cleaned_zh
    }

DATASET_REGISTRY = {
    "NilanE/ParallelFiction-Ja_En-100k": {
        "extractor": parse_parallel_fiction,
    },
    "botp/RyokoAI_ScribbleHub17K": {
        "extractor": parse_scribblehub,
    },
    "OmniAICreator/RoyalRoad-1.61M": {
        "extractor": parse_royalroad,
    },
    "botp/RyokoAI_CNNovel125K": {
        "extractor": parse_cnnovel,
    }
}

# Project Gutenberg Downloader and Parser

async def download_gutenberg_async(book_id: str) -> str:
    """Downloads raw text from Project Gutenberg asynchronously using aiohttp (max 30 simultaneous connections)."""
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"
    ]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    connector = aiohttp.TCPConnector(limit=30, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        for url in urls:
            try:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.read()
                        return data.decode('utf-8', errors='ignore')
            except Exception:
                continue
    raise ValueError(f"Could not download Gutenberg book with ID: {book_id}")

def download_gutenberg(book_id: str) -> str:
    """Wrapper to run the async Gutenberg downloader synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, download_gutenberg_async(book_id))
            return future.result()
    else:
        return loop.run_until_complete(download_gutenberg_async(book_id))

def parse_gutenberg(text: str) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Extracts title, author, language and splits the body into chapters."""
    title = "Unknown Gutenberg Book"
    author = "Unknown Author"
    lang = "en"

    title_match = re.search(r"Title:\s*(.*)", text)
    if title_match:
        title = title_match.group(1).strip()

    author_match = re.search(r"Author:\s*(.*)", text)
    if author_match:
        author = author_match.group(1).strip()

    lang_match = re.search(r"Language:\s*(.*)", text)
    if lang_match:
        lang_str = lang_match.group(1).strip().lower()
        if "chinese" in lang_str or "zh" in lang_str:
            lang = "zh"
        elif "japanese" in lang_str or "ja" in lang_str:
            lang = "ja"

    start_match = re.search(r"\*\*\*\s*START OF TH[IS|E] PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)
    end_match = re.search(r"\*\*\*\s*END OF TH[IS|E] PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)

    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(text)

    body = text[start_idx:end_idx].strip()

    # Split by chapter headers (supports standard English, Roman stories, and Chinese 回/章)
    chapter_pattern = r"(?mi)^(?:\s*(?:CHAPTER|Chapter|Ch\.|CHAP|Chap|Story|Adventure|ADVENTURE|STORY)\s+(?:[0-9]+|[IVXLCDM]+)\.?.*|(?:\s*[IVXLCDM]+\.\s+[A-Z].*)|(?:\s*[IVXLCDM]+\.?\s*\n)|(?:\s*第[一二三四五六七八九十百千零0-9]+[回章].*))"
    all_matches = list(re.finditer(chapter_pattern, body))

    # Filter out Table of Contents matches (where chapter body would be too short)
    matches = []
    for idx, match in enumerate(all_matches):
        start = match.start()
        end = all_matches[idx + 1].start() if idx + 1 < len(all_matches) else len(body)
        if (end - start) >= 400:
            matches.append(match)

    chapters = []
    if not matches:
        chapters.append({
            "chapter_number": 1,
            "chapter_title": "Full Book",
            "text": body,
            "lang": lang
        })
    else:
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)

            ch_header = match.group(0).strip()
            ch_body = body[start + len(match.group(0)):end].strip()

            if not ch_body:
                continue

            ch_num = idx + 1
            num_match = re.search(r"(?:CHAPTER|Chapter|Ch\.)\s+([0-9]+|[IVXLCDM]+)", ch_header, re.IGNORECASE)
            if num_match:
                val = num_match.group(1)
                if val.isdigit():
                    ch_num = int(val)

            ch_lines = ch_body.split('\n')
            first_line = ch_lines[0].strip()
            if len(first_line) > 0 and len(first_line) < 100 and not re.search(r'[.!?]', first_line):
                ch_title = f"{ch_header}: {first_line}"
                ch_text = "\n".join(ch_lines[1:]).strip()
            else:
                ch_title = ch_header
                ch_text = ch_body

            chapters.append({
                "chapter_number": ch_num,
                "chapter_title": ch_title,
                "text": ch_text,
                "lang": lang
            })
    return title, author, chapters
