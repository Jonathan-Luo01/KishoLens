"""
build_centroids.py — Offline centroid generation for semantic genre matching.

Usage:
    uv run python -m kisholens.ml.build_centroids [--samples N] [--data-dir PATH]

Streams genre-labelled texts from HuggingFace datasets and the Gutenberg API,
embeds the first 1000 words of each text using all-MiniLM-L6-v2, and saves:
    data/genre_centroids.npy       — (G, 384) float32 centroid matrix
    data/genre_centroids_meta.json — genre names, territories, sample counts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Genre taxonomy
# ---------------------------------------------------------------------------

COMMON_GENRES = {
    "Action / Adventure": [
        "action", "adventure", "martial arts", "sports", "assassin", "mercenary",
        "revenge", "survival", "dungeons", "hunting", "death game", "unlimited flow",
        "sea stories", "wilderness survival", "voyages and travels", "expeditions",
        "pirates", "shipwrecks", "adventure stories", "quest", "journey",
    ],
    "Comedy": [
        "comedy", "humor", "humorous stories", "satire", "parody", "wit and humor",
        "comedies", "domestic comedy", "slapstick", "funny protagonist", "misunderstandings",
        "cheerful protagonist", "gag", "funny", "rom-com", "romantic comedy",
    ],
    "Romance": [
        "romance", "romantic", "love", "romantic fiction", "courtship", "love stories",
        "romance fiction", "marriage", "man-woman relationships", "shoujo", "josei",
        "harem", "smut", "modern romance", "urban romance", "ceo", "office romance",
        "entertainment industry", "first love", "contract marriage", "reverse harem",
        "yuri", "girls love", "girls-love", "yaoi", "boys love", "boys-love",
        "shounen-ai", "shoujo-ai", "danmei", "gl", "bl",
    ],
    "Drama": [
        "drama", "plays", "theatre", "domestic drama", "social drama", "domestic fiction",
        "family life", "interpersonal relations", "family conflict", "betrayal",
        "emotional", "social hierarchy", "politics", "psychological",
    ],
    "Fantasy": [
        "fantasy", "mythology", "folklore", "fairy tales", "legends", "fables",
        "magic", "allegories", "high fantasy", "high-fantasy", "epic fantasy",
        "epic-fantasy", "sword and sorcery", "sword-and-sorcery", "tolkienesque",
        "medieval fantasy", "medieval-fantasy", "litrpg", "system", "vrmmo",
        "leveling", "gamelit", "game elements", "stat", "stats", "kingdom building",
        "kingdom-building", "dragons", "elves", "sword and magic", "academy",
        "cozy fantasy", "cozy-fantasy", "monster protagonist", "beast mc",
        "dungeon core", "dungeon mc", "dungeon building",
    ],
    "Horror": [
        "horror", "gothic fiction", "ghost stories", "horror tales", "vampires",
        "monsters", "occult fiction", "macabre", "survival horror", "ghosts",
        "demons", "gore", "psychological horror", "mystery horror",
    ],
    "Historical": [
        "historical fiction", "historical", "regency fiction", "victorian",
        "victorian novel", "19th century", "history", "biographical fiction",
        "war stories", "middle ages", "ancient china", "palace court", "royalty",
        "historical romance", "historical politics",
    ],
    "Sci-Fi": [
        "science fiction", "sci-fi", "sci fi", "mecha", "time travel", "space voyages",
        "space flight", "dystopias", "dystopian", "precursors of science fiction", "imaginary voyages",
        "cyberpunk", "post-apocalyptic", "interstellar", "vrmmo", "space opera",
        "galactic empire", "futuristic", "hard sci-fi", "hard-sci-fi", "future", "future life",
        "technology", "invention", "robot", "cyborg", "space", "outer space", "aliens",
    ],
    "Philosophy": [
        "philosophical fiction", "philosophical", "philosophy", "utilitarianism",
        "ethics", "conduct of life", "existentialism", "existential",
        "epistemology", "metaphysics", "stoicism", "nihilism", "rationalism",
    ],
    "Mystery": [
        "detective fiction", "crime stories", "mystery", "detective and mystery stories",
        "crime", "murder", "police procedural", "whodunit", "criminology", "investigation",
        "thriller", "suspense", "noir", "modern thriller", "detective", "sleuth", "clue", "puzzle",
    ],
    "Tragedy": [
        "tragedies", "tragedy", "tragic ending", "tragic novel", "tragic story",
        "dark literature", "fatalism", "moral downfall", "melancholy", "suffering",
        "sad ending", "terminal illness", "heartbreak", "character death", "angst",
        "grief", "sorrow", "mourning", "doom",
    ],
    "Supernatural": [
        "supernatural", "occult", "paranormal", "spiritualism", "apparitions",
        "demonology", "metaphysical fiction", "urban fantasy", "modern magic",
        "hunters", "necromancy", "vampires", "vampire", "werewolves", "exorcism", "gates",
        "ghost", "ghosts", "witch", "witches", "witchcraft", "demon", "demons", "monster", "monsters", "spirit", "spirits",
    ],
    "Poetry": [
        "poetry", "ballads", "epic poetry", "sonnets", "lyric poetry", "verse",
    ],
    "Slice of Life": [
        "slice of life", "slice-of-life", "slow life", "slow-life", "farming", "cooking",
        "pet raising", "school life", "school-life", "esports", "medical", "cute children",
        "ordinary life", "daily life", "contemporary", "cozy", "relaxing", "everyday life",
        "village life", "gardening", "ranching", "peaceful life", "baking", "housework",
    ],
    "Cultivation": [
        "cultivation", "cultivator", "cultivate", "xianxia", "xuanhuan", "wuxia", "qi", "dao", "dantian",
        "alchemy", "immortal", "immortality", "sects", "sect", "realm breakthrough", "eastern fantasy",
        "daoist", "golden core", "foundation building", "internal energy",
    ],
    "Isekai": [
        "isekai", "portal fantasy", "portal-fantasy", "reincarnation", "reincarnated", "reincarnate",
        "reborn", "transmigration", "transmigrated", "transmigrate", "otome game", "otome-game",
        "villainess", "regressor", "second chance", "world travel", "transported", "another world",
        "other world", "different world", "tensei", "summoned hero", "summoned",
    ],
    "Progression Fantasy": [
        "progression fantasy", "progression-fantasy", "progression", "litrpg", "system",
        "weak to strong", "level system", "cheats", "accelerated growth", "system administrator",
        "vrmmorpg", "vrmmo", "dungeon", "adventurers", "status window", "stat points",
        "level up", "dungeon core", "monster protagonist", "gamelit", "stat screen",
        "leveling", "dungeon building", "system notification", "class rank",
    ],
}

GENRE_TAG_MAP: dict[str, list[str]] = COMMON_GENRES

GENRE_TERRITORIES: dict[str, str] = {
    # Classic Literature Territory
    "Action / Adventure": "Classic Literature Territory",
    "Comedy":             "Classic Literature Territory",
    "Drama":              "Classic Literature Territory",
    "Fantasy":            "Classic Literature Territory",
    "Horror":             "Classic Literature Territory",
    "Historical":         "Classic Literature Territory",
    "Sci-Fi":             "Classic Literature Territory",
    "Philosophy":         "Classic Literature Territory",
    "Mystery":            "Classic Literature Territory",
    "Tragedy":            "Classic Literature Territory",
    "Supernatural":       "Classic Literature Territory",
    "Poetry":             "Classic Literature Territory",
    "Romance":            "Classic Literature Territory",

    # Web Novel Territory
    "Slice of Life":        "Web Novel Territory",
    "Cultivation":          "Web Novel Territory",
    "Isekai":               "Web Novel Territory",
    "Progression Fantasy": "Web Novel Territory",
}

# ---------------------------------------------------------------------------
# Tag consolidation
# ---------------------------------------------------------------------------

def consolidate_genre(tags: list[str]) -> Optional[str]:
    """
    Map a list of raw source tags to one of the 17 canonical parent genres.
    """
    import re
    t_lows = [t.lower().strip() for t in tags]
    
    # Priority order for matching canonical parent genres
    priority_order = [
        "Isekai", "Cultivation", "Progression Fantasy", "Slice of Life", "Poetry", "Tragedy",
        "Philosophy", "Supernatural", "Mystery", "Horror", "Historical",
        "Sci-Fi", "Fantasy", "Drama", "Comedy", "Romance", "Action / Adventure"
    ]
    
    for canonical in priority_order:
        tag_list = GENRE_TAG_MAP.get(canonical, [])
        for tag in tag_list:
            if any(re.search(r'\b' + re.escape(tag) + r'\b', t) for t in t_lows):
                return canonical
            
    return None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

import threading

_model_cache: dict[str, object] = {}
_model_lock = threading.Lock()


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load and cache the SentenceTransformer model on CPU safely."""
    if model_name not in _model_cache:
        with _model_lock:
            if model_name not in _model_cache:
                from sentence_transformers import SentenceTransformer
                _model_cache[model_name] = SentenceTransformer(model_name, device="cpu")
    return _model_cache[model_name]


def get_representative_sample(text: str, target_words: int = 1000) -> str:
    """
    Extracts a representative sample of text by combining chunks from the
    beginning, middle, and end of the book/draft, bypassing local topic bias.
    """
    import re
    # If the text is short (e.g., short story, web novel chapter < 15000 characters),
    # just return the first target_words.
    if not text or len(text) < 15000:
        words = text.split()
        return " ".join(words[:target_words])

    # We want to sample from three regions:
    # 1. Beginning (at 20% mark, to bypass TOC/preface)
    # 2. Middle (at 50% mark)
    # 3. End (at 80% mark)
    words_per_segment = target_words // 3
    
    def get_chunk(start_pct: float) -> str:
        start_idx = int(len(text) * start_pct)
        # Align to next paragraph break (double newline) to keep clean boundaries
        match = re.search(r'\n\s*\n', text[start_idx:])
        if match:
            start_idx = start_idx + match.end()
        
        # Grab a character buffer large enough for the target words
        buffer_len = words_per_segment * 12
        raw_chunk = text[start_idx : start_idx + buffer_len]
        chunk_words = raw_chunk.split()
        return " ".join(chunk_words[:words_per_segment])

    chunk_beg = get_chunk(0.20)
    chunk_mid = get_chunk(0.50)
    chunk_end = get_chunk(0.80)
    
    return f"{chunk_beg}\n\n{chunk_mid}\n\n{chunk_end}"


def extract_3window_slices(text: str, words_per_slice: int = 300) -> tuple[str, str, str]:
    """
    Sample 3 distinct 300-word slices across the text:
    - Window A (0% mark / Beginning): Captures setup, exposition, and core tropes
    - Window B (25% mark): Captures general pacing and narrative tone
    - Window C (60% mark): Captures mid-story execution and secondary elements
    """
    import re
    if not text:
        return ("", "", "")

    def get_slice(start_pct: float) -> str:
        start_idx = int(len(text) * start_pct)
        match = re.search(r'\n\s*\n', text[start_idx:])
        if match and match.end() < 500:
            start_idx = start_idx + match.end()
        buffer_len = words_per_slice * 12
        raw_chunk = text[start_idx : start_idx + buffer_len]
        return " ".join(raw_chunk.split()[:words_per_slice])

    win_a = get_slice(0.0)
    win_b = get_slice(0.25)
    win_c = get_slice(0.60)
    return (win_a, win_b, win_c)


def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    max_words: int = 1000,
) -> np.ndarray:
    """
    Embed a list of texts using 3-window weighted pooling:
    V_prose = 0.50 * V_A + 0.25 * V_B + 0.25 * V_C
    Returns a (N, 384) float32 numpy array.
    """
    model = _get_model(model_name)
    all_embeddings = []

    for t in texts:
        win_a, win_b, win_c = extract_3window_slices(t, words_per_slice=300)
        slices = [s if s.strip() else "sample prose text" for s in [win_a, win_b, win_c]]
        v_s = model.encode(slices, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        v_a, v_b, v_c = v_s[0], v_s[1], v_s[2]

        v_prose = 0.50 * v_a + 0.25 * v_b + 0.25 * v_c
        norm = np.linalg.norm(v_prose)
        if norm == 0:
            norm = 1.0
        v_prose = v_prose / norm
        all_embeddings.append(v_prose)

    return np.vstack(all_embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Centroid math
# ---------------------------------------------------------------------------

def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the average (centroid) of a (N, D) embedding matrix.
    Returns a (D,) float32 vector.
    """
    return embeddings.mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_centroids(
    centroids: np.ndarray,
    meta: dict,
    filename_prefix: str = "genre",
    data_dir: str = "data",
) -> None:
    """
    Save centroid matrix and metadata to disk.

    Files written:
        {data_dir}/{filename_prefix}_centroids.npy
        {data_dir}/{filename_prefix}_centroids_meta.json
    """
    os.makedirs(data_dir, exist_ok=True)
    npy_path = os.path.join(data_dir, f"{filename_prefix}_centroids.npy")
    meta_path = os.path.join(data_dir, f"{filename_prefix}_centroids_meta.json")
    np.save(npy_path, centroids)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved {filename_prefix} centroids to {npy_path}")
    print(f"Saved {filename_prefix} metadata to {meta_path}")


def load_centroids_from_disk(
    filename_prefix: str = "genre",
    data_dir: str = "data",
) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Load pre-built centroids from disk.
    Returns (None, None) if either file is missing.
    """
    npy_path = os.path.join(data_dir, f"{filename_prefix}_centroids.npy")
    meta_path = os.path.join(data_dir, f"{filename_prefix}_centroids_meta.json")
    if not os.path.exists(npy_path) or not os.path.exists(meta_path):
        return None, None
    centroids = np.load(npy_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return centroids, meta


# ---------------------------------------------------------------------------
# HuggingFace streaming helpers
# ---------------------------------------------------------------------------

def _stream_hf_genre_texts(
    dataset_name: str,
    text_field: str,
    tags_field: str,
    samples_per_genre: int,
) -> dict[str, list[str]]:
    """
    Stream a HuggingFace dataset and collect up to `samples_per_genre` texts
    per canonical genre. Returns {genre: [text, ...]}.
    """
    from datasets import load_dataset

    genre_texts: dict[str, list[str]] = {g: [] for g in GENRE_TAG_MAP}
    hf_genres = {g for g in GENRE_TAG_MAP if GENRE_TERRITORIES[g] != "Classic Literature Territory"}

    try:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"[WARN] Could not load {dataset_name}: {e}", file=sys.stderr)
        return genre_texts

    max_scan = max(5000, samples_per_genre * 50)
    ds_iter = iter(ds)
    try:
        for idx, row in enumerate(ds_iter):
            if idx >= max_scan:
                break
            if all(len(genre_texts[g]) >= samples_per_genre for g in hf_genres):
                break
            raw_tags = row.get(tags_field, []) or []
            if not raw_tags and "meta" in row and isinstance(row["meta"], dict):
                raw_tags = row["meta"].get(tags_field, []) or []
            if isinstance(raw_tags, str):
                raw_tags = raw_tags.strip()
                if raw_tags.startswith("[") and raw_tags.endswith("]"):
                    try:
                        import json
                        raw_tags = json.loads(raw_tags)
                    except Exception:
                        raw_tags = [t.strip() for t in raw_tags.split(",")]
                else:
                    raw_tags = [t.strip() for t in raw_tags.split(",")]
            genre = consolidate_genre(raw_tags)
            if genre is None or len(genre_texts[genre]) >= samples_per_genre:
                continue
            text = row.get(text_field, "") or ""
            if len(text.strip()) < 100:
                continue
            genre_texts[genre].append(text)
    finally:
        if 'ds_iter' in locals():
            if hasattr(ds_iter, 'close'):
                try:
                    ds_iter.close()
                except Exception:
                    pass
            del ds_iter
        if 'ds' in locals():
            del ds
        import gc
        gc.collect()

    return genre_texts


# ---------------------------------------------------------------------------
# Gutenberg API helper
# ---------------------------------------------------------------------------

def _fetch_gutenberg_texts_by_topic(
    topic: str,
    genre: str,
    samples: int,
) -> list[str]:
    """
    Fetch plain-text books from the Gutenberg API for a given topic.
    Uses gutendex.com (free, no auth).
    Returns a list of text snippets (first 1000 words each).
    """
    import urllib.request
    import urllib.parse

    texts: list[str] = []
    base_url = "https://gutendex.com/books"
    page = 1

    while len(texts) < samples:
        params = urllib.parse.urlencode({
            "topic": topic,
            "languages": "en",
            "page": page,
        })
        url = f"{base_url}?{params}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[WARN] Gutenberg API error for topic={topic}: {e}", file=sys.stderr)
            break

        books = data.get("results", [])
        if not books:
            break

        for book in books:
            if len(texts) >= samples:
                break
            # Prefer plain text format
            formats = book.get("formats", {})
            txt_url = (
                formats.get("text/plain; charset=utf-8")
                or formats.get("text/plain; charset=us-ascii")
                or formats.get("text/plain")
            )
            if not txt_url:
                continue
            try:
                txt_req = urllib.request.Request(
                    txt_url,
                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
                )
                with urllib.request.urlopen(txt_req, timeout=20) as r:
                    raw = r.read().decode("utf-8", errors="ignore")
                # Strip Gutenberg header/footer
                start = raw.find("*** START OF")
                end = raw.find("*** END OF")
                if start != -1:
                    raw = raw[start + 50:]
                if end != -1:
                    raw = raw[:end]
                if len(raw.split()) < 200:
                    continue
                texts.append(raw)
                print(f"  [{genre}] fetched: {book.get('title', 'Unknown')[:60]}")
            except Exception as e:
                print(f"[WARN] Could not fetch text for {book.get('title', '?')}: {e}", file=sys.stderr)

        if data.get("next") is None:
            break
        page += 1

    return texts


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_genre_centroids(
    samples_per_genre: int = 200,
) -> tuple[np.ndarray, dict, np.ndarray, dict]:
    """
    Build genre and territory centroids from HuggingFace (stable live streams)
    and local database classic novels (to avoid flaky live Gutenberg API).
    """
    import ssl
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass

    # Helper to load classic texts from local DB first
    def get_classic_texts_from_db(genre_name: str, limit: int) -> list[str]:
        db_path = "data/kisholens.db"
        if os.path.exists(db_path):
            try:
                from sqlmodel import Session, select
                from kisholens.models import Novel, Chapter, get_engine
                engine = get_engine()
                with Session(engine) as s:
                    # Find Gutenberg novels with this genre
                    novels = s.exec(select(Novel).where(Novel.source == "gutenberg", Novel.genre == genre_name)).all()
                    texts = []
                    for n in novels:
                        if len(texts) >= limit:
                            break
                        # Fetch chapters
                        chapters = s.exec(select(Chapter).where(Chapter.novel_id == n.id).order_by(Chapter.chapter_number)).all()
                        if not chapters:
                            continue
                        # Use the representative multi-chapter structure
                        if len(chapters) == 1:
                            t = chapters[0].text_en or chapters[0].text_ja or getattr(chapters[0], 'text_zh', '') or ''
                        elif len(chapters) == 2:
                            t = (chapters[0].text_en or '') + '\n\n' + (chapters[1].text_en or '')
                        else:
                            t = (chapters[0].text_en or '') + '\n\n' + (chapters[len(chapters)//2].text_en or '') + '\n\n' + (chapters[-1].text_en or '')
                        if t.strip():
                            texts.append(t)
                    if len(texts) > 0:
                        print(f"Loaded {len(texts)} classic '{genre_name}' texts from database offline.")
                        return texts
            except Exception as e:
                print(f"[WARN] Error loading classic '{genre_name}' from DB: {e}")
        return []

    # 1. HuggingFace: ScribbleHub17K
    print("Streaming ScribbleHub17K...")
    sh_texts = _stream_hf_genre_texts(
        dataset_name="botp/RyokoAI_ScribbleHub17K",
        text_field="text",
        tags_field="tags",
        samples_per_genre=samples_per_genre,
    )

    # 2. HuggingFace: RoyalRoad-1.61M
    print("Streaming RoyalRoad-1.61M...")
    rr_texts = _stream_hf_genre_texts(
        dataset_name="OmniAICreator/RoyalRoad-1.61M",
        text_field="text",
        tags_field="tags",
        samples_per_genre=samples_per_genre,
    )

    # Merge HF results
    hf_pool: dict[str, list[str]] = {}
    for genre in GENRE_TAG_MAP:
        hf_pool[genre] = sh_texts.get(genre, []) + rr_texts.get(genre, [])

    # Gutenberg classic topics mapping
    gutenberg_topics = {
        "Philosophy": ["philosophy", "ethics", "philosophical fiction"],
        "Poetry": ["poetry", "sonnets", "verse"],
        "Tragedy": ["tragedy", "tragedies", "fatalism"],
        "Supernatural": ["supernatural", "apparitions", "spiritualism"],
        "Mystery": ["detective", "mystery", "crime"],
        "Horror": ["horror", "gothic"],
        "Romance": ["romance", "love"],
        "Fantasy": ["fantasy", "fairy tales", "mythology"],
        "Sci-Fi": ["science fiction", "sci-fi"],
        "Action / Adventure": ["adventure", "action", "sea stories"],
        "Comedy": ["humor", "comedy", "satire"],
        "Historical": ["historical fiction", "history"],
        "Drama": ["drama", "plays"],
    }

    # Lists for territory centroids
    web_texts = []
    trad_texts = []
    classic_texts = []

    # Combined dictionary of texts per canonical genre
    combined: dict[str, list[str]] = {}

    web_genres = {"Slice of Life", "Cultivation", "Isekai"}
    classic_specific_genres = {"Philosophy", "Poetry", "Tragedy", "Supernatural"}
    common_genres = {"Action / Adventure", "Comedy", "Drama", "Fantasy", "Horror", "Historical", "Sci-Fi", "Mystery", "Romance"}

    for genre in GENRE_TAG_MAP:
        if genre in web_genres:
            texts = hf_pool.get(genre, [])[:samples_per_genre]
            combined[genre] = texts
            web_texts.extend(texts)
        elif genre in classic_specific_genres:
            # Try DB first
            texts = get_classic_texts_from_db(genre, samples_per_genre)
            if not texts:
                needed = samples_per_genre
                print(f"Fetching Gutenberg texts for {genre}...")
                fetched = []
                for topic in gutenberg_topics.get(genre, [genre.lower()]):
                    if needed <= 0:
                        break
                    f = _fetch_gutenberg_texts_by_topic(topic, genre, needed)
                    fetched.extend(f)
                    needed -= len(f)
                texts = fetched[:samples_per_genre]
            combined[genre] = texts
            classic_texts.extend(texts)
        elif genre in common_genres:
            # Half from HF (traditional), half from Gutenberg (classic)
            half = max(1, samples_per_genre // 2)

            # HF portion (traditional)
            hf_texts = hf_pool.get(genre, [])[:half]
            trad_texts.extend(hf_texts)

            # Gutenberg portion (classic) - Try DB first
            g_texts = get_classic_texts_from_db(genre, half)
            if not g_texts:
                needed = half
                print(f"Fetching Gutenberg texts for Classic {genre}...")
                fetched = []
                for topic in gutenberg_topics.get(genre, [genre.lower()]):
                    if needed <= 0:
                        break
                    f = _fetch_gutenberg_texts_by_topic(topic, genre, needed)
                    fetched.extend(f)
                    needed -= len(f)
                g_texts = fetched[:half]
            classic_texts.extend(g_texts)

            combined[genre] = hf_texts + g_texts

    # 4. Embed + compute genre centroids
    all_genres = list(GENRE_TAG_MAP.keys())
    genre_centroid_list = []
    genre_samples_used = {}

    for genre in all_genres:
        texts = combined.get(genre, [])
        print(f"Embedding {len(texts)} texts for '{genre}'...")
        if not texts:
            print(f"[WARN] No texts for genre '{genre}' — using zero vector.", file=sys.stderr)
            centroid = np.zeros(384, dtype=np.float32)
        else:
            embeddings = embed_texts(texts)
            centroid = compute_centroid(embeddings)
        genre_centroid_list.append(centroid)
        genre_samples_used[genre] = len(texts)

    genre_centroids = np.stack(genre_centroid_list, axis=0)
    genre_meta = {
        "genres": all_genres,
        "samples_used": genre_samples_used,
    }

    # 5. Embed + compute territory centroids
    territory_centroids_list = []
    territory_samples_used = {}
    territory_names = ["Classic Literature Territory", "Web Novel Territory"]

    for name, texts in [("Classic Literature Territory", classic_texts),
                        ("Web Novel Territory", web_texts)]:
        print(f"Embedding {len(texts)} texts for territory '{name}'...")
        if not texts:
            print(f"[WARN] No texts for territory '{name}' — using zero vector.", file=sys.stderr)
            centroid = np.zeros(384, dtype=np.float32)
        else:
            embeddings = embed_texts(texts)
            centroid = compute_centroid(embeddings)
        territory_centroids_list.append(centroid)
        territory_samples_used[name] = len(texts)

    territory_centroids = np.stack(territory_centroids_list, axis=0)
    territory_meta = {
        "territories": territory_names,
        "samples_used": territory_samples_used,
    }

    return genre_centroids, genre_meta, territory_centroids, territory_meta


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build semantic genre and territory centroids for KishoLens."
    )
    parser.add_argument(
        "--samples", type=int, default=200,
        help="Max texts per genre (default: 200)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Output directory for centroids (default: data/)"
    )
    args = parser.parse_args()

    print(f"Building centroids (up to {args.samples} samples/genre)...")
    g_centroids, g_meta, t_centroids, t_meta = build_genre_centroids(
        samples_per_genre=args.samples,
    )

    save_centroids(g_centroids, g_meta, filename_prefix="genre", data_dir=args.data_dir)
    save_centroids(t_centroids, t_meta, filename_prefix="territory", data_dir=args.data_dir)

    print("\nCentroid build complete.")
    print(f"  Genres: {g_meta['genres']}")
    print(f"  Territories: {t_meta['territories']}")

    # Explicitly exit to prevent hanging from non-daemon background threads in datasets/huggingface
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
