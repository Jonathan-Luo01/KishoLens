# Accurate Literary Similarity & Style Matching Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hydrate `_novel_vector_cache` from `data/stats_cache.json` and implement primary-genre-affinity multi-factor similarity matching so that custom prose analysis in the KishoLens Analyzer returns accurate, stylistically and semantically aligned literary doppelgängers.

**Architecture:** Load all 10,320 precomputed 8D stylistic radar vectors, taxonomy classifications, and metadata into in-memory cache on startup. Compute vector cosine + L1 distance alongside primary-genre affinity and territory matching to rank candidates in $< 20\text{ms}$.

**Tech Stack:** Python 3.11+, NumPy, SQLModel, PyTorch / Sentence-Transformers, Pytest, FastAPI.

## Global Constraints

- Target Files: `kisholens/ml/similarity.py`, `tests/ml/test_similarity.py`
- Precomputed Cache Path: `data/stats_cache.json`
- Feature Keys (8D): `theme_explication_ratio`, `linearity_subversion_score`, `sensory_body_density`, `outside_world_engagement`, `narrative_feature_diversity`, `dialogue_ratio`, `ttr`, `temporal_shift_score`
- All pytest tests must pass cleanly with `uv run pytest tests/`.

---

### Task 1: Vector Cache Disk Hydration in similarity.py

**Files:**
- Modify: `kisholens/ml/similarity.py:35-100`
- Test: `tests/ml/test_similarity.py`

**Interfaces:**
- Produces: `_init_cache_from_disk() -> None` populating `_novel_vector_cache: Dict[int, dict]` with 8D radar vectors, primary genres, and metadata.

- [ ] **Step 1: Write the failing test for cache hydration**

```python
# In tests/ml/test_similarity.py
from kisholens.ml.similarity import _init_cache_from_disk, _novel_vector_cache

def test_cache_hydration_from_disk():
    _init_cache_from_disk()
    assert len(_novel_vector_cache) >= 10000
    # Novel 235 is The Adventures of Sherlock Holmes
    assert 235 in _novel_vector_cache
    entry = _novel_vector_cache[235]
    assert entry["title"] == "The Adventures of Sherlock Holmes"
    assert entry["vector"].shape == (8,)
    assert entry["primary_genre"] == "Mystery"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/ml/test_similarity.py::test_cache_hydration_from_disk -v`
Expected: FAIL with missing `_init_cache_from_disk` or unhydrated cache.

- [ ] **Step 3: Implement `_init_cache_from_disk` in `kisholens/ml/similarity.py`**

```python
import json
import os
from pathlib import Path

DATA_CACHE_PATH = Path("data/stats_cache.json")

def _init_cache_from_disk() -> None:
    """
    Hydrates _novel_vector_cache from data/stats_cache.json.
    Extracts 8D normalized radar vectors, primary taxonomy genres, top genres,
    territories, and metadata for fast in-memory similarity matching.
    """
    global _novel_vector_cache
    if _novel_vector_cache:
        return

    if not DATA_CACHE_PATH.exists():
        return

    try:
        with open(DATA_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for k, v in data.items():
            if k.startswith("_"):
                continue
            try:
                nid = int(k)
            except ValueError:
                continue

            # Extract 8D normalized radar vector
            norm_radar = v.get("normalized_radar", {})
            vec = []
            prefix = ""
            for p in ["en_", "ja_", "zh_"]:
                if any(r_k.startswith(p) for r_k in norm_radar.keys()):
                    prefix = p
                    break

            for feat_key in RADAR_FEATURE_KEYS:
                full_key = f"{prefix}{feat_key}" if prefix else feat_key
                val = norm_radar.get(full_key, norm_radar.get(feat_key, 0.5))
                vec.append(float(val))

            np_vec = np.array(vec, dtype=float)

            # Extract taxonomy & genres
            am = v.get("archetype_match", {})
            top_genres = [g.get("genre", "") for g in am.get("top_genres", []) if g.get("genre")]
            primary_genre = top_genres[0] if top_genres else (v.get("genre") or "Fiction")
            territories = [t.get("territory", "") for t in am.get("top_territories", []) if t.get("territory")]
            territory = territories[0] if territories else (v.get("territory") or "Unknown")

            _novel_vector_cache[nid] = {
                "id": nid,
                "title": v.get("title", f"Novel #{nid}"),
                "author": v.get("author") or "Unknown Author",
                "genre": ", ".join(top_genres) if top_genres else primary_genre,
                "primary_genre": primary_genre,
                "top_genres": top_genres,
                "territory": territory,
                "vector": np_vec,
                "semantic": am,
            }
    except Exception as e:
        print(f"[similarity] Warning: Failed to hydrate vector cache from disk: {e}")
```

Call `_init_cache_from_disk()` at module load or at top of `find_top_matches`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_similarity.py::test_cache_hydration_from_disk -v`
Expected: PASS

- [ ] **Step 5: Commit Task 1**

```bash
git add kisholens/ml/similarity.py tests/ml/test_similarity.py
git commit -m "feat(similarity): hydrate novel vector cache from disk with 8D radar stylistics"
```

---

### Task 2: Primary Genre Affinity & Enhanced Multi-Factor Scoring

**Files:**
- Modify: `kisholens/ml/similarity.py:200-350`
- Test: `tests/ml/test_similarity.py`

**Interfaces:**
- Consumes: `_novel_vector_cache` from Task 1
- Produces: `find_top_matches(query_features, query_text, exclude_novel_id, top_k) -> List[Dict]` with high accuracy for Victorian mystery, romance, and isekai excerpts.

- [ ] **Step 1: Write failing test for Victorian mystery sample matching**

```python
# In tests/ml/test_similarity.py
from kisholens.ml.features import extract_english_features
from kisholens.ml.similarity import find_top_matches

def test_sherlock_holmes_mystery_matching():
    text = """The rain beat against the fog-stained windowpanes of 221B Baker Street as Inspector Lestrade threw open the heavy oak door. His coat was drenched, and his eyes burned with anxiety. "Holmes, you must come at once," he gasped, resting his hands upon the polished mahogany table. "Lord Harrington lies motionless in his study, the doors locked from within and a shattered crystal decanter resting beside his chair."

Sherlock Holmes did not rise immediately. He slowly lowered his pipe, allowing a dense ring of blue smoke to curl toward the ceiling before adjusting his magnifying lens. "A locked room, Lestrade? How delightfully elementary. And tell me, did you observe the faint scent of bitter almonds clinging to the victim's lips?" Lestrade blinked in astonishment. "Why, yes, Holmes—how could you possibly know?" Holmes turned to me with a faint smile. "A classic case of cyanide poisoning, Watson. Pack your bag; the hunt is afoot."""

    feat = extract_english_features(text)
    matches = find_top_matches(feat, query_text=text, top_k=5)
    
    assert len(matches) == 5
    # The top matches must contain Arthur Conan Doyle or Sherlock Holmes works
    matched_titles = [m["title"] for m in matches]
    has_sherlock = any("Sherlock" in t for t in matched_titles)
    assert has_sherlock, f"Expected Sherlock Holmes in top matches, got: {matched_titles}"
    
    top_match = matches[0]
    assert top_match["similarity_score"] >= 0.70
    assert top_match["breakdown"]["genre"] >= 0.80
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/ml/test_similarity.py::test_sherlock_holmes_mystery_matching -v`
Expected: FAIL (returns isekai/web novels).

- [ ] **Step 3: Implement primary genre affinity and enhanced scoring in `find_top_matches`**

In `kisholens/ml/similarity.py`:
1. Call `_init_cache_from_disk()`.
2. Extract query primary genre and top genres from `query_semantic` or `query_features`.
3. For each candidate in `_novel_vector_cache`:
   - Calculate 8D style similarity using real cached `n_vec`.
   - Calculate genre similarity:
     - If query primary genre is strong ($S_{\text{primary}} \ge 0.65$) and candidate matches primary genre: assign $0.85 + 0.15 \times \text{Jaccard}(G_q, G_n)$.
     - Else if primary genre is in top 3: assign $0.60 + 0.25 \times \text{Jaccard}(G_q, G_n)$.
     - Else: assign $0.20 \times \text{Jaccard}(G_q, G_n)$.
   - Calculate concept embedding cosine similarity and territory similarity.
   - Composite score = $0.30 \times \text{style} + 0.35 \times \text{genre} + 0.20 \times \text{semantic} + 0.15 \times \text{territory}$.
4. Sort by `(similarity_score, breakdown["genre"])` descending.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_similarity.py::test_sherlock_holmes_mystery_matching -v`
Expected: PASS

- [ ] **Step 5: Commit Task 2**

```bash
git add kisholens/ml/similarity.py tests/ml/test_similarity.py
git commit -m "feat(similarity): implement primary genre affinity and multi-factor similarity ranking"
```

---

### Task 3: End-to-End API Verification & Integration Suite

**Files:**
- Test: `tests/ml/test_similarity.py`
- Test: Full Pytest Suite

- [ ] **Step 1: Add multi-genre test cases (Romance, Isekai, Mystery)**

```python
# In tests/ml/test_similarity.py
def test_isekai_fantasy_matching():
    text = """I woke up in an unfamiliar stone chamber with a glowing blue interface hovering before my eyes. 
[System Initialized. Welcome, User. Status: Level 1 Reincarnated Adventurer.]
I grabbed my iron dagger and stepped out into the monster-infested dungeon."""
    feat = extract_english_features(text)
    matches = find_top_matches(feat, query_text=text, top_k=5)
    assert len(matches) == 5
    # Should match Isekai / Fantasy web novels
    top_match = matches[0]
    assert top_match["similarity_score"] >= 0.60
```

- [ ] **Step 2: Run the full PyTest test suite**

Run: `uv run pytest tests/`
Expected: 100% tests pass (64+ passed).

- [ ] **Step 3: Commit Task 3**

```bash
git add tests/ml/test_similarity.py
git commit -m "test(similarity): add end-to-end multi-genre doppelganger verification tests"
```
