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

GENRE_TAG_MAP: dict[str, list[str]] = {
    # --- Specific Genres (checked first) ---
    "LitRPG": [
        "litrpg", "system", "vrmmo", "leveling", "game", "game-elements",
        "game elements", "stat", "stats", "gamelit",
    ],
    "Isekai": [
        "isekai", "reincarnation", "portal-fantasy", "portal fantasy",
        "transported", "another-world", "another world", "transmigration",
    ],
    "Xianxia / Wuxia": [
        "xianxia", "wuxia", "cultivation", "eastern-fantasy", "eastern fantasy",
        "xuanhuan", "martial-arts", "martial arts", "daoist", "dao",
    ],
    "Urban Romance": [
        "contemporary-romance", "contemporary romance", "slice-of-life", "slice of life",
        "school-life", "school life", "modern-day", "modern day",
    ],
    "High Fantasy": [
        "high-fantasy", "high fantasy", "epic-fantasy", "epic fantasy",
        "sword-and-sorcery", "sword and sorcery", "tolkienesque", "medieval-fantasy",
        "medieval fantasy",
    ],
    "Hard Sci-Fi": [
        "hard-sci-fi", "hard sci-fi", "space-opera", "space opera", "cyberpunk",
    ],
    "Modern Thriller": [
        "thriller", "suspense", "psychological", "noir",
    ],
    "Victorian Novel": [
        "victorian", "gothic", "19th-century", "19th century",
    ],
    "Philosophical Fiction": [
        "philosophy", "philosophical", "existential",
    ],
    # --- General / Common Genres (checked last) ---
    "Mystery": [
        "mystery", "detective", "crime", "cozy mystery", "investigation", "mystery-thriller",
    ],
    "Horror": [
        "horror", "ghosts", "paranormal", "dark-fantasy", "dark fantasy",
    ],
    "Romance": [
        "romance", "romantic", "love", "romantic comedy", "rom-com",
    ],
    "Fantasy": [
        "fantasy", "magic", "mythology", "myth", "supernatural",
    ],
    "Sci-Fi": [
        "science-fiction", "science fiction", "sci-fi", "sci fi", "aliens", "mecha",
    ],
    "Action / Adventure": [
        "action", "adventure", "quest", "journey",
    ],
    "Comedy": [
        "comedy", "humor", "satire", "parody", "funny",
    ],
}

GENRE_TERRITORIES: dict[str, str] = {
    "LitRPG":                "Web Novel Territory",
    "Isekai":                "Web Novel Territory",
    "Xianxia / Wuxia":      "Web Novel Territory",
    "Urban Romance":         "Web Novel Territory",
    "High Fantasy":          "Traditional Fiction Territory",
    "Hard Sci-Fi":           "Traditional Fiction Territory",
    "Modern Thriller":       "Traditional Fiction Territory",
    "Victorian Novel":       "Classic Literature Territory",
    "Philosophical Fiction": "Classic Literature Territory",
    # General / Common Genres
    "Mystery":               "Traditional Fiction Territory",
    "Horror":                "Traditional Fiction Territory",
    "Romance":               "Traditional Fiction Territory",
    "Fantasy":               "Traditional Fiction Territory",
    "Sci-Fi":                "Traditional Fiction Territory",
    "Action / Adventure":    "Traditional Fiction Territory",
    "Comedy":                "Traditional Fiction Territory",
}

# ---------------------------------------------------------------------------
# Tag consolidation
# ---------------------------------------------------------------------------

def consolidate_genre(tags: list[str]) -> Optional[str]:
    """
    Map a list of raw source tags to a canonical genre name.
    Returns the first canonical genre whose tag set intersects with `tags`,
    or None if no match.

    Comparison is case-insensitive.
    """
    normalised = {t.lower() for t in tags}
    for genre, genre_tags in GENRE_TAG_MAP.items():
        if normalised & {gt.lower() for gt in genre_tags}:
            return genre
    return None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_model_cache: dict[str, object] = {}


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load and cache the SentenceTransformer model."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _truncate_to_words(text: str, max_words: int = 1000) -> str:
    """Return the first `max_words` whitespace-separated tokens of `text`."""
    words = text.split()
    return " ".join(words[:max_words])


def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    max_words: int = 1000,
) -> np.ndarray:
    """
    Embed a list of texts (each truncated to `max_words` words).
    Returns a (N, 384) float32 numpy array.
    """
    truncated = [_truncate_to_words(t, max_words) for t in texts]
    model = _get_model(model_name)
    embeddings = model.encode(truncated, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


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
    data_dir: str = "data",
) -> None:
    """
    Save centroid matrix and metadata to disk.

    Files written:
        {data_dir}/genre_centroids.npy
        {data_dir}/genre_centroids_meta.json
    """
    os.makedirs(data_dir, exist_ok=True)
    npy_path = os.path.join(data_dir, "genre_centroids.npy")
    meta_path = os.path.join(data_dir, "genre_centroids_meta.json")
    np.save(npy_path, centroids)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved centroids to {npy_path}")
    print(f"Saved metadata to {meta_path}")


def load_centroids_from_disk(
    data_dir: str = "data",
) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Load pre-built centroids from disk.
    Returns (None, None) if either file is missing.
    """
    npy_path = os.path.join(data_dir, "genre_centroids.npy")
    meta_path = os.path.join(data_dir, "genre_centroids_meta.json")
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
    hf_genres = {g for g in GENRE_TAG_MAP if g not in {"Victorian Novel", "Philosophical Fiction"}}

    try:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"[WARN] Could not load {dataset_name}: {e}", file=sys.stderr)
        return genre_texts

    max_scan = max(5000, samples_per_genre * 50)
    for idx, row in enumerate(ds):
        if idx >= max_scan:
            break
        if all(len(genre_texts[g]) >= samples_per_genre for g in hf_genres):
            break
        raw_tags = row.get(tags_field, []) or []
        if not raw_tags and "meta" in row and isinstance(row["meta"], dict):
            raw_tags = row["meta"].get(tags_field, []) or []
        if isinstance(raw_tags, str):
            raw_tags = [t.strip() for t in raw_tags.split(",")]
        genre = consolidate_genre(raw_tags)
        if genre is None or len(genre_texts[genre]) >= samples_per_genre:
            continue
        text = row.get(text_field, "") or ""
        if len(text.strip()) < 100:
            continue
        genre_texts[genre].append(text)

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
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
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
                with urllib.request.urlopen(txt_url, timeout=20) as r:
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
) -> tuple[np.ndarray, dict]:
    """
    Build genre centroids from HuggingFace + Gutenberg data.

    Steps:
    1. Stream ScribbleHub17K for web/traditional fiction genres
    2. Cross-check with RoyalRoad-1.61M for High Fantasy, Hard Sci-Fi, Modern Thriller
    3. Fetch Gutenberg texts for Victorian Novel and Philosophical Fiction
    4. Embed first 1000 words of each text per genre
    5. Average embeddings per genre → centroid vector

    Returns:
        centroids: (G, 384) float32 array
        meta: {"genres": [...], "territories": [...], "samples_used": {...}}
    """
    import ssl
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass

    # 1. HuggingFace: ScribbleHub17K
    print("Streaming ScribbleHub17K...")
    sh_texts = _stream_hf_genre_texts(
        dataset_name="botp/RyokoAI_ScribbleHub17K",
        text_field="text",
        tags_field="tags",
        samples_per_genre=samples_per_genre,
    )

    # 2. HuggingFace: RoyalRoad-1.61M (cross-check for traditional fiction)
    print("Streaming RoyalRoad-1.61M...")
    rr_texts = _stream_hf_genre_texts(
        dataset_name="OmniAICreator/RoyalRoad-1.61M",
        text_field="text",
        tags_field="tags",
        samples_per_genre=samples_per_genre,
    )

    # Merge HF results (union, capped at samples_per_genre)
    combined: dict[str, list[str]] = {}
    for genre in GENRE_TAG_MAP:
        pool = sh_texts.get(genre, []) + rr_texts.get(genre, [])
        combined[genre] = pool[:samples_per_genre]

    # 3. Gutenberg for Classic Literature genres
    gutenberg_topics = {
        "Victorian Novel": ["gothic fiction", "fiction"],
        "Philosophical Fiction": ["philosophy", "fiction"],
    }
    for genre, topics in gutenberg_topics.items():
        needed = samples_per_genre - len(combined.get(genre, []))
        if needed <= 0:
            continue
        print(f"Fetching Gutenberg texts for {genre}...")
        for topic in topics:
            if needed <= 0:
                break
            fetched = _fetch_gutenberg_texts_by_topic(topic, genre, needed)
            combined.setdefault(genre, []).extend(fetched)
            needed -= len(fetched)
        combined[genre] = combined.get(genre, [])[:samples_per_genre]

    # 4. Embed + compute centroids
    all_genres = list(GENRE_TAG_MAP.keys())
    centroid_list = []
    territories = []
    samples_used = {}

    for genre in all_genres:
        texts = combined.get(genre, [])
        print(f"Embedding {len(texts)} texts for '{genre}'...")
        if not texts:
            print(f"[WARN] No texts for genre '{genre}' — using zero vector.", file=sys.stderr)
            centroid = np.zeros(384, dtype=np.float32)
        else:
            embeddings = embed_texts(texts)
            centroid = compute_centroid(embeddings)

        centroid_list.append(centroid)
        territories.append(GENRE_TERRITORIES[genre])
        samples_used[genre] = len(texts)

    centroids = np.stack(centroid_list, axis=0)  # (G, 384)

    meta = {
        "genres": all_genres,
        "territories": territories,
        "samples_used": samples_used,
    }
    return centroids, meta


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build semantic genre centroids for KishoLens."
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

    print(f"Building genre centroids (up to {args.samples} samples/genre)...")
    centroids, meta = build_genre_centroids(
        samples_per_genre=args.samples,
    )
    save_centroids(centroids, meta, data_dir=args.data_dir)

    print("\nCentroid build complete.")
    print(f"  Genres: {meta['genres']}")
    print(f"  Samples used: {meta['samples_used']}")


if __name__ == "__main__":
    main()
