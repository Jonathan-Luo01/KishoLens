# Semantic Trope & Territory Matching — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add semantic genre/territory matching using `all-MiniLM-L6-v2` sentence embeddings alongside the existing stylometric `match_archetype()` classifier, without breaking any existing API endpoints.

**Architecture:** Two new modules — `build_centroids` (offline CLI that streams HF datasets + Gutenberg, embeds texts, and saves centroid vectors) and `semantic_match` (live inference that loads those centroids, embeds input text, and returns cosine-ranked genre matches). The API's `post_analyze` gains one additive `semantic` key in its response.

**Tech Stack:** `sentence-transformers>=2.6.0` (all-MiniLM-L6-v2, CPU), `scikit-learn>=1.3.0` (cosine_similarity), HuggingFace `datasets` (already in deps, streaming), `aiohttp` (already in deps, for Gutenberg API).

## Global Constraints

- Python >= 3.10 (project requirement)
- No GPU required — CPU inference only
- No paid APIs — HuggingFace free tier + Gutenberg API (https://gutendex.com)
- match_archetype() return dict shape MUST NOT change — HIGH blast radius (3 live endpoints + CLI)
- `semantic` key in POST /api/analyze response is optional (omit if centroids absent, never raise 500)
- All new files under `kisholens/ml/` follow the existing lazy-import pattern from `features.py`
- Centroids saved to `data/genre_centroids.npy` + `data/genre_centroids_meta.json` (gitignored)
- Run via `uv run python -m kisholens.ml.build_centroids` and `uv run python -m kisholens.ml.semantic_match`
- Tests run via `uv run pytest tests/`

---

### Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `sentence-transformers` and `scikit-learn` available in the venv

- [ ] **Step 1: Add deps to pyproject.toml**

```toml
# In [project] dependencies list, add after "pandas>=2.0.0,":
"sentence-transformers>=2.6.0",
"scikit-learn>=1.3.0",
```

- [ ] **Step 2: Sync the venv**

```bash
uv sync
```

Expected: resolves without error; `sentence_transformers` and `sklearn` importable.

- [ ] **Step 3: Verify imports work**

```bash
uv run python -c "from sentence_transformers import SentenceTransformer; print('OK')"
uv run python -c "from sklearn.metrics.pairwise import cosine_similarity; print('OK')"
```

Expected: both print `OK`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add sentence-transformers and scikit-learn dependencies"
```

---

### Task 2: `build_centroids.py` — Tag Consolidation + Centroid Math

**Files:**
- Create: `kisholens/ml/build_centroids.py`
- Create: `tests/ml/__init__.py` (empty)
- Create: `tests/ml/test_build_centroids.py`

**Interfaces:**
- Consumes: nothing from prior tasks (standalone module)
- Produces:
  - `GENRE_TAG_MAP: dict[str, list[str]]` — maps canonical genre name → list of lowercase source tags
  - `GENRE_TERRITORIES: dict[str, str]` — maps canonical genre name → territory label
  - `consolidate_genre(tags: list[str]) -> str | None` — returns first canonical match or None
  - `embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray` — shape (N, 384)
  - `compute_centroid(embeddings: np.ndarray) -> np.ndarray` — shape (384,)
  - `save_centroids(centroids: np.ndarray, meta: dict, data_dir: str = "data") -> None`
  - `load_centroids_from_disk(data_dir: str = "data") -> tuple[np.ndarray, dict] | tuple[None, None]`

- [ ] **Step 1: Write failing tests**

Create `tests/ml/__init__.py` (empty file).

Create `tests/ml/test_build_centroids.py`:

```python
import numpy as np
import os
import json
import tempfile
import pytest

from kisholens.ml.build_centroids import (
    consolidate_genre,
    embed_texts,
    compute_centroid,
    save_centroids,
    load_centroids_from_disk,
    GENRE_TAG_MAP,
    GENRE_TERRITORIES,
)


def test_consolidate_genre_known_litrpg():
    assert consolidate_genre(["litrpg", "adventure"]) == "LitRPG"


def test_consolidate_genre_known_system():
    assert consolidate_genre(["system", "action"]) == "LitRPG"


def test_consolidate_genre_known_isekai():
    assert consolidate_genre(["isekai", "fantasy"]) == "Isekai"


def test_consolidate_genre_known_cultivation():
    assert consolidate_genre(["cultivation", "action"]) == "Xianxia / Wuxia"


def test_consolidate_genre_no_match():
    assert consolidate_genre(["unknown-tag-xyz"]) is None


def test_consolidate_genre_empty():
    assert consolidate_genre([]) is None


def test_consolidate_genre_case_insensitive():
    assert consolidate_genre(["Isekai"]) == "Isekai"
    assert consolidate_genre(["LITRPG"]) == "LitRPG"


def test_genre_tag_map_has_all_genres():
    expected = {"LitRPG", "Isekai", "Xianxia / Wuxia", "Urban Romance",
                "High Fantasy", "Hard Sci-Fi", "Modern Thriller",
                "Victorian Novel", "Philosophical Fiction"}
    assert expected == set(GENRE_TAG_MAP.keys())


def test_genre_territories_covers_all_genres():
    for genre in GENRE_TAG_MAP:
        assert genre in GENRE_TERRITORIES, f"Missing territory for {genre}"


def test_embed_texts_shape():
    texts = ["The hero raised his sword.", "She cultivated qi in silence.", "A mystery unfolded."]
    embeddings = embed_texts(texts)
    assert embeddings.shape == (3, 384)
    assert embeddings.dtype == np.float32


def test_embed_texts_single():
    texts = ["Just one sentence."]
    embeddings = embed_texts(texts)
    assert embeddings.shape == (1, 384)


def test_embed_texts_truncation():
    # 2000-word text should not raise
    long_text = "word " * 2000
    embeddings = embed_texts([long_text])
    assert embeddings.shape == (1, 384)


def test_compute_centroid_of_identical_vectors():
    v = np.array([1.0, 0.0, 0.0] * 128, dtype=np.float32).reshape(1, 384)
    repeated = np.tile(v, (5, 1))
    centroid = compute_centroid(repeated)
    np.testing.assert_allclose(centroid, v[0], atol=1e-6)


def test_compute_centroid_shape():
    embeddings = np.random.rand(10, 384).astype(np.float32)
    centroid = compute_centroid(embeddings)
    assert centroid.shape == (384,)


def test_save_load_roundtrip():
    centroids = np.random.rand(3, 384).astype(np.float32)
    meta = {
        "genres": ["LitRPG", "Isekai", "High Fantasy"],
        "territories": ["Web Novel Territory", "Web Novel Territory", "Traditional Fiction Territory"],
        "samples_used": {"LitRPG": 10, "Isekai": 10, "High Fantasy": 10}
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        save_centroids(centroids, meta, data_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "genre_centroids.npy"))
        assert os.path.exists(os.path.join(tmpdir, "genre_centroids_meta.json"))
        loaded_centroids, loaded_meta = load_centroids_from_disk(data_dir=tmpdir)
        np.testing.assert_array_equal(centroids, loaded_centroids)
        assert loaded_meta["genres"] == meta["genres"]


def test_load_centroids_missing_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_centroids_from_disk(data_dir=tmpdir)
        assert result == (None, None)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/ml/test_build_centroids.py -v 2>&1 | head -30
```

Expected: `ImportError` or `ModuleNotFoundError` for `kisholens.ml.build_centroids`.

- [ ] **Step 3: Implement `build_centroids.py`**

Create `kisholens/ml/build_centroids.py`:

```python
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
        "romance", "contemporary-romance", "contemporary romance",
        "urban", "slice-of-life", "slice of life", "school-life", "school life",
        "modern-day", "modern day",
    ],
    "High Fantasy": [
        "high-fantasy", "high fantasy", "epic-fantasy", "epic fantasy",
        "sword-and-sorcery", "sword and sorcery", "tolkienesque", "medieval-fantasy",
        "medieval fantasy",
    ],
    "Hard Sci-Fi": [
        "science-fiction", "science fiction", "hard-sci-fi", "hard sci-fi",
        "sci-fi", "sci fi", "space-opera", "space opera", "cyberpunk",
        "post-apocalyptic", "post apocalyptic",
    ],
    "Modern Thriller": [
        "thriller", "mystery", "crime", "detective", "suspense",
        "horror", "psychological", "noir",
    ],
    "Victorian Novel": [
        "victorian", "gothic", "19th-century", "19th century",
        "historical", "classical",
    ],
    "Philosophical Fiction": [
        "philosophy", "philosophical", "literary-fiction", "literary fiction",
        "existential", "literary", "classic",
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
    needed = set(GENRE_TAG_MAP.keys())

    try:
        ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Could not load {dataset_name}: {e}", file=sys.stderr)
        return genre_texts

    for row in ds:
        if not needed:
            break
        raw_tags = row.get(tags_field, []) or []
        if isinstance(raw_tags, str):
            raw_tags = [t.strip() for t in raw_tags.split(",")]
        genre = consolidate_genre(raw_tags)
        if genre is None or len(genre_texts[genre]) >= samples_per_genre:
            continue
        text = row.get(text_field, "") or ""
        if len(text.strip()) < 100:
            continue
        genre_texts[genre].append(text)
        if all(len(genre_texts[g]) >= samples_per_genre for g in needed if g not in {"Victorian Novel", "Philosophical Fiction"}):
            needed -= {g for g in needed if len(genre_texts[g]) >= samples_per_genre}

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
    data_dir: str = "data",
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
    # 1. HuggingFace: ScribbleHub17K
    print("Streaming ScribbleHub17K...")
    sh_texts = _stream_hf_genre_texts(
        dataset_name="ScribbleHub17K",
        text_field="text",
        tags_field="tags",
        samples_per_genre=samples_per_genre,
    )

    # 2. HuggingFace: RoyalRoad-1.61M (cross-check for traditional fiction)
    print("Streaming RoyalRoad-1.61M...")
    rr_texts = _stream_hf_genre_texts(
        dataset_name="RoyalRoad-1.61M",
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
        data_dir=args.data_dir,
    )
    save_centroids(centroids, meta, data_dir=args.data_dir)

    print("\nCentroid build complete.")
    print(f"  Genres: {meta['genres']}")
    print(f"  Samples used: {meta['samples_used']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — they should pass now**

```bash
uv run pytest tests/ml/test_build_centroids.py -v
```

Expected: all tests pass. The `embed_texts` tests will download `all-MiniLM-L6-v2` (~80 MB) on first run.

- [ ] **Step 5: Commit**

```bash
git add kisholens/ml/build_centroids.py tests/ml/__init__.py tests/ml/test_build_centroids.py
git commit -m "feat: add build_centroids module with tag consolidation and centroid math"
```

---

### Task 3: `semantic_match.py` — Live Inference Module

**Files:**
- Create: `kisholens/ml/semantic_match.py`
- Create: `tests/ml/test_semantic_match.py`
- Modify: `kisholens/ml/__init__.py`

**Interfaces:**
- Consumes:
  - `embed_texts(texts, model_name) -> np.ndarray` from `build_centroids.py`
  - `load_centroids_from_disk(data_dir) -> (np.ndarray | None, dict | None)` from `build_centroids.py`
  - `GENRE_TERRITORIES: dict[str, str]` from `build_centroids.py`
- Produces:
  - `match_semantic(text: str, model_name: str = "all-MiniLM-L6-v2", data_dir: str = "data") -> dict | None`
    Returns `None` if centroids not built.
    Returns `{"genre": str, "territory": str, "confidence": float, "scores": [{"genre": str, "territory": str, "score": float}, ...]}`

- [ ] **Step 1: Write failing tests**

Create `tests/ml/test_semantic_match.py`:

```python
import numpy as np
import json
import os
import tempfile
import pytest

from kisholens.ml.semantic_match import match_semantic


def _write_dummy_centroids(tmpdir: str):
    """Write minimal genre centroids for testing (normalized random vectors)."""
    from kisholens.ml.build_centroids import GENRE_TAG_MAP, GENRE_TERRITORIES

    genres = list(GENRE_TAG_MAP.keys())
    # Use deterministic vectors so tests are reproducible
    rng = np.random.default_rng(42)
    centroids = rng.random((len(genres), 384)).astype(np.float32)
    # Normalize rows so cosine similarity is well-defined
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / norms

    meta = {
        "genres": genres,
        "territories": [GENRE_TERRITORIES[g] for g in genres],
        "samples_used": {g: 10 for g in genres},
    }
    np.save(os.path.join(tmpdir, "genre_centroids.npy"), centroids)
    with open(os.path.join(tmpdir, "genre_centroids_meta.json"), "w") as f:
        json.dump(meta, f)
    return genres, centroids, meta


def test_match_semantic_returns_none_without_centroids():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = match_semantic("Some text here.", data_dir=tmpdir)
    assert result is None


def test_match_semantic_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        genres, _, _ = _write_dummy_centroids(tmpdir)
        result = match_semantic(
            "The young hero reincarnated into another world and gained a system.",
            data_dir=tmpdir,
        )
    assert result is not None
    assert "genre" in result
    assert "territory" in result
    assert "confidence" in result
    assert "scores" in result
    assert isinstance(result["scores"], list)
    assert len(result["scores"]) == len(genres)


def test_match_semantic_scores_sorted_descending():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("Magic cultivation and martial arts in ancient China.", data_dir=tmpdir)
    scores = [s["score"] for s in result["scores"]]
    assert scores == sorted(scores, reverse=True)


def test_match_semantic_scores_in_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("A detective investigates a murder in London.", data_dir=tmpdir)
    for s in result["scores"]:
        assert -1.0 <= s["score"] <= 1.0


def test_match_semantic_top_genre_matches_confidence():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("Leveling up with game stats and a system prompt.", data_dir=tmpdir)
    assert result["genre"] == result["scores"][0]["genre"]
    assert abs(result["confidence"] - result["scores"][0]["score"]) < 1e-6


def test_match_semantic_scores_have_all_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("Philosophy and existence.", data_dir=tmpdir)
    for s in result["scores"]:
        assert "genre" in s
        assert "territory" in s
        assert "score" in s


def test_match_semantic_territory_correct():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("Victorian gothic mystery in London fog.", data_dir=tmpdir)
    # Each score entry's territory must match the known territory for that genre
    from kisholens.ml.build_centroids import GENRE_TERRITORIES
    for s in result["scores"]:
        assert s["territory"] == GENRE_TERRITORIES[s["genre"]]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/ml/test_semantic_match.py -v 2>&1 | head -20
```

Expected: `ImportError` for `kisholens.ml.semantic_match`.

- [ ] **Step 3: Implement `semantic_match.py`**

Create `kisholens/ml/semantic_match.py`:

```python
"""
semantic_match.py — Live inference for semantic genre/territory matching.

Loads pre-built genre centroids from disk and matches a user's text
to the closest genre using cosine similarity over sentence embeddings.

Gracefully returns None if centroids have not been built yet.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from kisholens.ml.build_centroids import (
    embed_texts,
    load_centroids_from_disk,
    GENRE_TERRITORIES,
)

# Module-level centroid cache: (data_dir -> (centroids, meta))
_centroid_cache: dict[str, tuple[np.ndarray, dict]] = {}

# Default centroid location (relative to project root)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)


def _load_with_cache(data_dir: str) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Load centroids from disk, caching per data_dir so repeated calls within
    the same process do not re-read files.
    """
    if data_dir not in _centroid_cache:
        centroids, meta = load_centroids_from_disk(data_dir)
        if centroids is not None:
            _centroid_cache[data_dir] = (centroids, meta)
        else:
            return None, None
    return _centroid_cache[data_dir]


def match_semantic(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
    data_dir: str = DEFAULT_DATA_DIR,
) -> Optional[dict]:
    """
    Embed `text` and compute cosine similarity against all genre centroids.

    Returns None if centroids have not been built (data files absent).

    Returns a dict:
    {
        "genre":      str,    # canonical genre with highest similarity
        "territory":  str,    # territory for that genre
        "confidence": float,  # highest cosine similarity score [0, 1]
        "scores": [           # ALL genres, sorted descending by score
            {"genre": str, "territory": str, "score": float},
            ...
        ]
    }
    """
    centroids, meta = _load_with_cache(data_dir)
    if centroids is None or meta is None:
        return None

    genres: list[str] = meta["genres"]
    territories: list[str] = meta["territories"]

    # Embed the input text
    embedding = embed_texts([text], model_name=model_name)  # (1, 384)

    # Cosine similarity: (1, 384) @ (384, G) → (1, G)
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(embedding, centroids)[0]  # (G,)

    # Build sorted scores list
    scores = [
        {
            "genre": genres[i],
            "territory": territories[i],
            "score": float(sims[i]),
        }
        for i in range(len(genres))
    ]
    scores.sort(key=lambda x: x["score"], reverse=True)

    best = scores[0]
    return {
        "genre": best["genre"],
        "territory": best["territory"],
        "confidence": best["score"],
        "scores": scores,
    }
```

- [ ] **Step 4: Update `kisholens/ml/__init__.py`**

```python
# Current content of kisholens/ml/__init__.py:
# (check with: cat kisholens/ml/__init__.py)
# Add the following export:
```

Read the current `__init__.py` first, then append:

```bash
cat kisholens/ml/__init__.py
```

Add to `kisholens/ml/__init__.py` (append line):

```python
from kisholens.ml.semantic_match import match_semantic  # noqa: F401
```

- [ ] **Step 5: Run tests — they should pass**

```bash
uv run pytest tests/ml/test_semantic_match.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 6: Commit**

```bash
git add kisholens/ml/semantic_match.py tests/ml/test_semantic_match.py kisholens/ml/__init__.py
git commit -m "feat: add semantic_match module for live genre/territory inference"
```

---

### Task 4: API Integration — Add `semantic` Key to `POST /api/analyze`

**Files:**
- Modify: `kisholens/api/main.py` (lines 549–553 and 624–647)

**Interfaces:**
- Consumes: `match_semantic(text: str, data_dir: str) -> dict | None` from `kisholens.ml.semantic_match`
- Produces: `POST /api/analyze` response gains optional `"semantic"` key (additive, non-breaking)

> **IMPORTANT:** `match_archetype()` return shape MUST NOT change. Only add after it.
> The key `archetype_match` and its sub-keys `closest_trope`, `confidence`, `territory` are
> consumed by lines 628–631 and must remain identical.

- [ ] **Step 1: Add import at top of `kisholens/api/main.py`**

Find the existing import block (around line 11–20). Add ONE new import line:

```python
from kisholens.ml.semantic_match import match_semantic
```

(Add it after the existing `from kisholens.ml.features import (...)` block.)

- [ ] **Step 2: Call `match_semantic` inside `post_analyze`**

In `post_analyze()` (around line 552), find this block:

```python
    agg = {f"{lang}_{k}": v for k, v in features.items()}
    agg["pacing"] = pacing
    archetype = match_archetype(agg)
    agg["archetype_match"] = archetype
```

Add ONE line after it (before the Kishōtenketsu arc section):

```python
    semantic = match_semantic(request.text)
```

- [ ] **Step 3: Add `semantic` key to the return dict**

Find the return statement (around line 624). It currently looks like:

```python
    return {
        "status": "success",
        "detected_lang": lang,
        "features": features,
        "archetype": {
            "archetype": archetype["closest_trope"],
            "confidence": archetype["confidence"],
            "description": f"Classification: {archetype['territory']}. Closest matched writing archetype based on stylistic features."
        },
        "baselines": { ... },
        "stats": agg,
        "arc": arc
    }
```

Add `"semantic": semantic` as a new key (after `"archetype"`):

```python
    response = {
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
        "arc": arc
    }
    if semantic is not None:
        response["semantic"] = semantic
    return response
```

- [ ] **Step 4: Verify the server starts without error**

```bash
uv run uvicorn kisholens.api.main:app --reload &
sleep 3
curl -s http://localhost:8000/health
```

Expected: `{"status":"ok","service":"kisholens"}`

- [ ] **Step 5: Test the analyze endpoint manually (no centroids — graceful degradation)**

```bash
curl -s -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The hero raised his sword against the demon king.", "lang": "en"}' \
  | python3 -m json.tool | grep -E '"archetype"|"semantic"|"status"'
```

Expected: `"status": "success"`, `"archetype"` key present, NO `"semantic"` key (centroids not built).

- [ ] **Step 6: Kill dev server and commit**

```bash
kill %1 2>/dev/null || true
git add kisholens/api/main.py
git commit -m "feat: add semantic match to POST /api/analyze response (additive, graceful degradation)"
```

---

### Task 5: npm Script + Build Centroids Smoke Test

**Files:**
- Modify: `package.json`

**Interfaces:**
- Produces: `npm run dev:build-centroids` runs the centroid build CLI

- [ ] **Step 1: Add npm script**

In `package.json`, find the `"scripts"` object and add:

```json
"dev:build-centroids": "node run-venv.js python -m kisholens.ml.build_centroids"
```

- [ ] **Step 2: Verify the script is callable**

```bash
npm run dev:build-centroids -- --help
```

Expected: prints the argparse help message with `--samples` and `--data-dir` options.

- [ ] **Step 3: Run a minimal smoke test (5 samples to keep it fast)**

```bash
npm run dev:build-centroids -- --samples 5 --data-dir data
```

Expected: downloads model (first run), fetches texts, saves `data/genre_centroids.npy` and `data/genre_centroids_meta.json`, prints "Centroid build complete."

> **Note:** Some genres may show 0 samples if HuggingFace datasets require auth or Gutenberg has no results for the topic — this is acceptable. A zero vector is used as fallback.

- [ ] **Step 4: Verify centroids file exists and has correct shape**

```bash
uv run python -c "
import numpy as np, json
c = np.load('data/genre_centroids.npy')
m = json.load(open('data/genre_centroids_meta.json'))
print('Shape:', c.shape)
print('Genres:', m['genres'])
print('Samples:', m['samples_used'])
"
```

Expected: Shape `(9, 384)`, 9 genres listed.

- [ ] **Step 5: Test `POST /api/analyze` with centroids present**

```bash
uv run uvicorn kisholens.api.main:app --reload &
sleep 3
curl -s -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The young man reincarnated with a system and began leveling up his stats.", "lang": "en"}' \
  | python3 -m json.tool | python3 -c "
import sys, json
data = json.load(sys.stdin)
sem = data.get('semantic')
if sem:
    print('Genre:', sem['genre'])
    print('Territory:', sem['territory'])
    print('Confidence:', sem['confidence'])
    print('Top 3 scores:', sem['scores'][:3])
else:
    print('No semantic key — centroids not loaded')
"
kill %1 2>/dev/null || true
```

Expected: genre close to `LitRPG` or `Isekai` with confidence > 0.5.

- [ ] **Step 6: Commit**

```bash
git add package.json
git commit -m "feat: add dev:build-centroids npm script"
```

---

### Task 6: Full Test Suite + Final Verification

**Files:**
- No new files (runs existing tests)

- [ ] **Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Run detect_changes (GitNexus requirement)**

```bash
node .gitnexus/run.cjs analyze
```

Then verify changes are limited to expected files:
- `kisholens/ml/build_centroids.py` (new)
- `kisholens/ml/semantic_match.py` (new)
- `kisholens/ml/__init__.py` (modified)
- `kisholens/api/main.py` (modified)
- `pyproject.toml` (modified)
- `package.json` (modified)

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final integration — semantic trope/territory matching complete"
```

---

## Self-Review

1. **Spec coverage:**
   - ✅ Function 1 (build centroids): Task 2 implements `build_centroids.py` with all required pieces
   - ✅ Function 2 (live inference): Task 3 implements `semantic_match.py` with cosine similarity + sorted scores
   - ✅ Tag consolidation: `consolidate_genre()` with full `GENRE_TAG_MAP`
   - ✅ HF datasets: `_stream_hf_genre_texts()` covers ScribbleHub17K + RoyalRoad-1.61M
   - ✅ Gutenberg API: `_fetch_gutenberg_texts_by_topic()` uses gutendex.com
   - ✅ API integration: Task 4 adds `semantic` key additively
   - ✅ Graceful degradation: `match_semantic` returns `None` when centroids absent
   - ✅ All tests: Tasks 2, 3 have full test suites

2. **Placeholder scan:** No TBDs. All code is complete.

3. **Type consistency:**
   - `embed_texts` used identically in both `build_centroids.py` and `semantic_match.py` (same signature)
   - `load_centroids_from_disk` returns `(np.ndarray | None, dict | None)` — consistent with `_load_with_cache`
   - `match_semantic` return dict keys exactly match the spec and test assertions
