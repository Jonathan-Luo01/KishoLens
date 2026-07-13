import re
import unicodedata
from typing import Optional, Tuple
from bs4 import BeautifulSoup

def clean_html(text: str) -> str:
    """Removes HTML tags using BeautifulSoup with lxml parser."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text()

def clean_japanese(text: str) -> str:
    """Removes Japanese ruby tags (｜ and 《 》) and normalizes unicode (NFKC)."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'《.*?》', '', text)
    text = text.replace('｜', '')
    return text

def clean_english(text: str) -> str:
    """Strips translator/editor notes matching [TL note: ...] or (T/N: ...)."""
    if not text:
        return ""
    # Match bracketed or parenthesized translator notes spanning multiple lines, preventing cross-bracket overmatching
    pattern = r"(?s)(?:\[(?:TL\s*note|T/N|Editor's\s*note|EN|TN):.*?\]|\((?:TL\s*note|T/N|Editor's\s*note|EN|TN):.*?\))"
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text.strip()

def parse_chapter_number(title: str) -> Optional[int]:
    """Helper to extract a chapter number from a chapter title string."""
    match_en = re.search(r'(?:[Cc]hapter|[Cc]h)\s*(\d+)', title, re.IGNORECASE)
    if match_en:
        return int(match_en.group(1))
    match_num = re.match(r'^(\d+)', title)
    if match_num:
        return int(match_num.group(1))
    return None

def extract_chapter_info(src: str, trg: str) -> Tuple[Optional[int], str, str, str, str]:
    """
    Extracts chapter number and titles from the first line of raw text.
    If the first line represents a title, it strips it from the returned body text.
    """
    src_first_line = src.strip().split('\n')[0]
    trg_first_line = trg.strip().split('\n')[0]
    
    chapter_number = None
    # Match Japanese headers like "77.素人の気づき"
    match_ja = re.match(r'^(\d+)[.．\s]', src_first_line)
    if match_ja:
        chapter_number = int(match_ja.group(1))
    else:
        chapter_number = parse_chapter_number(trg_first_line)
    
    is_header = chapter_number is not None
        
    title_ja = src_first_line
    if match_ja:
        title_ja = src_first_line[match_ja.end():].strip()
    
    title_en = trg_first_line
    match_en_title = re.match(r'^(?:[Cc]?hapter|[Cc]h)\s*\d+[\s:.\-]*', trg_first_line, re.IGNORECASE)
    if match_en_title:
        title_en = trg_first_line[match_en_title.end():].strip()
    else:
        match_en_start_num = re.match(r'^\d+[\s:.\-]*', trg_first_line)
        if match_en_start_num:
            title_en = trg_first_line[match_en_start_num.end():].strip()
            
    if not is_header:
        title_ja = "Chapter "
        title_en = "Chapter "
        
    body_ja = src
    body_en = trg
    if is_header:
        parts_ja = src.strip().split('\n', 1)
        body_ja = parts_ja[1].strip() if len(parts_ja) > 1 else ""
        parts_en = trg.strip().split('\n', 1)
        body_en = parts_en[1].strip() if len(parts_en) > 1 else ""
        
    return chapter_number, title_ja, title_en, body_ja, body_en
