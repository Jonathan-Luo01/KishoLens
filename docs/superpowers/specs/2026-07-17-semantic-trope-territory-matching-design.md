# Semantic Trope & Territory Matching — Design Spec

**Date:** 2026-07-17  
**Feature:** Semantic genre/territory classification via sentence embeddings  
**Status:** Approved

---

## Problem

The existing `match_archetype()` in `kisholens/ml/features.py` classifies novels into
tropes and territories using cosine similarity over **13 hand-crafted stylometric
feature vectors** (TTR, dialogue ratio, dep-tree depth, etc.). While effective at
capturing *how* prose is written, it cannot capture *what* it is about — thematic
content, setting vocabulary, or narrative subject matter.

Adding a semantic embedding layer allows genre matching on **meaning**, not just style.

---

## Goal

Add two new components to `kisholens/ml/`:

1. **`build_centroids` module** — offline CLI script that:
   - Streams genre-labelled texts from HuggingFace datasets and the Gutenberg API
   - Consolidates messy genre tags into canonical genres (e.g. `litrpg`, `system`, `game` → `LitRPG`)
   - Embeds the first 1 000 tokens of each text with `all-MiniLM-L6-v2`
   - Averages all embeddings per canonical genre → one centroid vector per genre
   - Saves: `data/genre_centroids.npy` (float32 matrix) + `data/genre_centroids_meta.json`

2. **`semantic_match` module** — live inference helper that:
   - Loads the pre-built centroids (lazy, cached)
   - Embeds a user's input text (first 1 000 tokens)
   - Computes cosine similarity against every centroid
   - Returns: closest genre, its territory, and all similarity scores sorted descending

3. **API integration** — extends the `POST /api/analyze` response to include a
   `semantic` key alongside the existing `archetype` key (additive — no breaking
   changes to the existing response shape).

---

## Constraints

- **No GPU required.** `all-MiniLM-L6-v2` runs comfortably on CPU (~50 ms/sample).
- **No paid APIs.** HuggingFace free tier (streaming) + Gutenberg API (free REST).
- **Non-breaking.** `match_archetype()` return shape stays identical; semantic result
  is added as a new key `semantic` in the `POST /api/analyze` response.
- **Graceful degradation.** If `genre_centroids.npy` does not exist, the API falls back
  silently and omits the `semantic` key rather than erroring.
- **Python >= 3.10**, runs under `uv sync`.

---

## Canonical Genre Taxonomy

| Canonical Genre | Territory | Source tags consolidated |
|---|---|---|
| LitRPG | Web Novel Territory | litrpg, system, vrmmo, leveling, game, game-elements, stat |
| Isekai | Web Novel Territory | isekai, reincarnation, portal-fantasy, transported, another-world |
| Xianxia / Wuxia | Web Novel Territory | xianxia, wuxia, cultivation, eastern-fantasy, xuanhuan, martial-arts |
| Urban Romance | Web Novel Territory | romance, contemporary-romance, urban, slice-of-life, school-life |
| High Fantasy | Traditional Fiction Territory | high-fantasy, epic-fantasy, sword-and-sorcery, tolkienesque |
| Hard Sci-Fi | Traditional Fiction Territory | science-fiction, hard-sci-fi, space-opera, cyberpunk, post-apocalyptic |
| Modern Thriller | Traditional Fiction Territory | thriller, mystery, crime, detective, suspense, horror |
| Victorian Novel | Classic Literature Territory | victorian, gothic, 19th-century (Gutenberg texts <= 1900 UK/US) |
| Philosophical Fiction | Classic Literature Territory | philosophy, literary-fiction, existential (Gutenberg; French/Russian lit) |

---

## Data Sources for Centroid Generation

| Genre(s) | Dataset | Filter / Method |
|---|---|---|
| LitRPG, Isekai, Xianxia, Urban Romance, High Fantasy, Hard Sci-Fi, Modern Thriller | HuggingFace: ScribbleHub17K | stream; filter rows where tags overlap with canonical tag sets |
| High Fantasy, Hard Sci-Fi, Modern Thriller (cross-check) | HuggingFace: RoyalRoad-1.61M | stream; same tag filtering |
| Victorian Novel, Philosophical Fiction | Gutenberg API GET /books?topic=<topic> | topics: fiction, philosophy, gothic fiction; filter download_count > 100, languages=en |

**Sample limit per genre:** 200 texts. Configurable via --samples-per-genre.

---

## File Map

| Action | File | Purpose |
|---|---|---|
| Create | kisholens/ml/build_centroids.py | CLI: streams texts, embeds, saves centroids |
| Create | kisholens/ml/semantic_match.py | Inference: load centroids + cosine match |
| Modify | pyproject.toml | Add sentence-transformers, scikit-learn to core deps |
| Modify | kisholens/api/main.py | Call semantic_match in post_analyze; add semantic key |
| Modify | kisholens/ml/__init__.py | Export semantic_match |
| Modify | package.json | Add dev:build-centroids npm script |
| Create | tests/ml/test_semantic_match.py | Unit tests for semantic_match module |
| Create | tests/ml/test_build_centroids.py | Unit tests for tag consolidation + centroid math |

---

## Module Interfaces

### kisholens/ml/build_centroids.py

```python
GENRE_TAG_MAP: dict[str, list[str]]  # canonical_genre -> list of source tags

def consolidate_genre(tags: list[str]) -> str | None: ...
def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray: ...
def build_genre_centroids(samples_per_genre: int = 200) -> tuple[np.ndarray, dict]: ...
def save_centroids(centroids: np.ndarray, meta: dict, data_dir: str = "data") -> None: ...
```

### kisholens/ml/semantic_match.py

```python
def load_centroids(npy_path: str, meta_path: str) -> tuple[np.ndarray, dict] | tuple[None, None]: ...
def match_semantic(text: str, model_name: str = "all-MiniLM-L6-v2") -> dict | None: ...
```

Returns:
```json
{
  "genre": "Isekai",
  "territory": "Web Novel Territory",
  "confidence": 0.91,
  "scores": [
    {"genre": "Isekai", "territory": "Web Novel Territory", "score": 0.91},
    {"genre": "LitRPG", "territory": "Web Novel Territory", "score": 0.73}
  ]
}
```

---

## API Response Change (POST /api/analyze)

Existing keys are UNCHANGED. New optional key `semantic` added:

```json
{
  "status": "success",
  "archetype": { "archetype": "Isekai", "confidence": 0.87, "description": "..." },
  "semantic": {
    "genre": "Isekai",
    "territory": "Web Novel Territory",
    "confidence": 0.91,
    "scores": [...]
  }
}
```

If centroids are not built, `semantic` key is omitted (no error).

---

## Dependencies to Add

```toml
"sentence-transformers>=2.6.0"
"scikit-learn>=1.3.0"
```

First call downloads model (~80 MB) to ~/.cache/huggingface/. Subsequent calls are local.

---

## Impact Analysis

- match_archetype() is called by 3 API endpoints + 1 CLI runner (HIGH risk to touch)
- This design adds semantic matching as a NEW parallel path — match_archetype is NOT modified
- Only post_analyze() gains one new additive line after existing archetype matching
