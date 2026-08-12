import argparse
import os
import json
import sqlite3
import sys
import re
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from kisholens.ml.build_centroids import COMMON_GENRES
from kisholens.pipeline.scraping import parse_parallel_fiction, parse_scribblehub, parse_royalroad

from kisholens.pipeline.disambiguation import disambiguate_and_rank_genres

def map_tags_to_parent_genres_priority(
    tags_str: str,
    text_sample: str = "",
    source: str = "",
    territory: str = "",
    initial_genre: str = ""
) -> str:
    res = disambiguate_and_rank_genres(
        tags_str=tags_str,
        text_sample=text_sample,
        source=source,
        territory=territory,
        initial_genre=initial_genre
    )
    return res["parent_genre_str"]

def matches_target_genre(target_genre: str, tags_str: Optional[str]) -> bool:
    if not tags_str:
        return False
    target = target_genre.lower().strip()
    tags_list = [t.strip().lower() for t in tags_str.split(",")]
    
    # Check direct tag match with word boundaries
    if any(re.search(r'\b' + re.escape(target) + r'\b', t) for t in tags_list):
        return True
    
    # Check parent taxonomy tag match with word boundaries
    keywords = COMMON_GENRES.get(target_genre, [target])
    for kw in keywords:
        if any(re.search(r'\b' + re.escape(kw) + r'\b', t) for t in tags_list):
            return True
            
    return False

def fetch_gutenberg_by_genre(genre: str, count: int) -> List[Dict[str, Any]]:
    print(f"Searching Project Gutenberg API for '{genre}' novels...")
    base_url = "https://gutendex.com/books"
    params = urllib.parse.urlencode({"topic": genre.lower(), "languages": "en"})
    url = f"{base_url}?{params}"
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    results = []
    
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            books = data.get("results", [])
            for b in books:
                if len(results) >= count:
                    break
                title = b.get("title", "Unknown Title")
                authors = b.get("authors", [])
                author_name = authors[0]["name"] if authors else "Unknown Author"
                
                # Fetch text plain format
                formats = b.get("formats", {})
                txt_url = formats.get("text/plain; charset=utf-8") or formats.get("text/plain")
                if not txt_url:
                    continue
                    
                try:
                    txt_req = urllib.request.Request(txt_url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(txt_req, timeout=10) as txt_resp:
                        raw_text = txt_resp.read().decode("utf-8", errors="ignore")
                        # Basic text slice for chapter
                        sample_text = " ".join(raw_text.split()[:3000])
                        results.append({
                            "series_title": title,
                            "author": author_name,
                            "source": "gutenberg",
                            "territory": "Classic Literature Territory",
                            "tags": ", ".join(b.get("subjects", [])),
                            "genre": genre,
                            "chapters": [{"chapter_number": 1, "chapter_title": "Full Text", "text_en": sample_text, "text_ja": ""}]
                        })
                except Exception:
                    continue
    except Exception as e:
        print(f"Gutenberg API fetch skipped/failed: {e}")
        
    return results

def ingest_genre(target_genre: str, count: int = 10):
    print(f"Targeting Genre Ingestion for: '{target_genre}' (Goal: {count} novels)")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    novels_added = 0
    novel_chapters = {}

    # 1. Stream HuggingFace ParallelFiction
    try:
        print("Scanning Hugging Face NilanE/ParallelFiction-Ja_En-100k...")
        ds_pf = load_dataset("NilanE/ParallelFiction-Ja_En-100k", split="train", streaming=True)
        for idx, item in enumerate(ds_pf):
            parsed = parse_parallel_fiction(item, idx)
            title = parsed["series_title"]
            tags = parsed.get("tags")
            
            if title and title != "Unknown" and matches_target_genre(target_genre, tags):
                if title not in novel_chapters:
                    cursor.execute("SELECT id FROM novel WHERE title = ?", (title,))
                    if cursor.fetchone():
                        continue
                    novel_chapters[title] = {
                        "author": parsed["author"] or "Unknown Author",
                        "source": "syosetu",
                        "territory": "Web Novel Territory",
                        "tags": tags,
                        "chapters": []
                    }
                    novels_added += 1
                    print(f"  [Found #{novels_added}] {title} (Source: ParallelFiction)")
                
                if len(novel_chapters[title]["chapters"]) < 5:
                    novel_chapters[title]["chapters"].append(parsed)

                if novels_added >= count:
                    break
    except Exception as e:
        print(f"  HF ParallelFiction streaming error: {e}")

    # 2. Check Gutenberg if count not reached
    if novels_added < count:
        needed = count - novels_added
        gutenberg_novels = fetch_gutenberg_by_genre(target_genre, count=needed)
        for g_novel in gutenberg_novels:
            title = g_novel["series_title"]
            cursor.execute("SELECT id FROM novel WHERE title = ?", (title,))
            if cursor.fetchone():
                continue
            novel_chapters[title] = g_novel
            novels_added += 1
            print(f"  [Found #{novels_added}] {title} (Source: Gutenberg)")
            if novels_added >= count:
                break

    # Save to SQLite
    print(f"\nSaving {len(novel_chapters)} matching '{target_genre}' novels to SQLite...")
    for title, info in novel_chapters.items():
        tags = info["tags"]
        genre_str = map_tags_to_parent_genres_priority(tags)
        if target_genre.lower() not in genre_str.lower():
            genre_str = f"{target_genre}, {genre_str}"

        author = info["author"] or "Unknown Author"
        cursor.execute(
            "INSERT INTO novel (title, author, source, territory, tags, genre) VALUES (?, ?, ?, ?, ?, ?)",
            (title, author, info["source"], info["territory"], tags, genre_str)
        )
        novel_id = cursor.lastrowid

        for ch_idx, ch in enumerate(info.get("chapters", [])):
            c_num = ch.get("chapter_number") or (ch_idx + 1)
            c_title = ch.get("chapter_title") or f"Chapter {c_num}"
            cursor.execute(
                "INSERT INTO chapter (novel_id, chapter_number, title, text_ja, text_en, text_zh) VALUES (?, ?, ?, ?, ?, ?)",
                (novel_id, c_num, c_title, ch.get("text_ja", ""), ch.get("text_en", ""), "")
            )

    conn.commit()
    conn.close()

    # Automatically rebuild centroids
    print("\nRebuilding centroids after genre ingestion...")
    os.system("uv run python scratch/build_centroids_db.py")
    print(f"\nSuccessfully completed targeted genre ingestion for '{target_genre}'!")

def main():
    parser = argparse.ArgumentParser(description="Targeted Genre Ingester for KishoLens")
    parser.add_argument("--genre", type=str, required=True, help="Specific genre to ingest (e.g. Comedy, Sci-Fi, Horror, Slice of Life)")
    parser.add_argument("--count", type=int, default=10, help="Number of novels to target")
    args = parser.parse_args()

    ingest_genre(args.genre, args.count)

if __name__ == "__main__":
    main()
