# Library Similarity Sync & Match Reasons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synchronize the Library page similarity function with the Analyze Prose similarity engine and display descriptive match reasons (e.g. similar prose, matching genre archetype, similar plot premise) without emojis or icons across both pages.

**Architecture:** Extend `find_top_matches()` in `kisholens/ml/similarity.py` to generate a `reasons: List[str]` array based on 5-factor breakdown scores. Ensure `get_novel_stats` in `kisholens/api/main.py` serves fresh `find_top_matches` payloads. Update `renderSimilarNovels` in `library.astro` and `analyze.astro` to display reason pills and factor details.

**Tech Stack:** Python 3.11+, FastAPI, NumPy, Astro, Vanilla CSS/JS, PyTest.

## Global Constraints

- Target Files: `kisholens/ml/similarity.py`, `kisholens/api/main.py`, `frontend/src/pages/library.astro`, `frontend/src/pages/analyze.astro`, `tests/ml/test_similarity.py`
- No emojis or icons in match reasons strings or badge markup.
- All pytest tests must pass cleanly (`uv run pytest tests/`).
- Frontend static build must pass cleanly (`cd frontend && npm run build`).

---

### Task 1: Backend Match Reasons Generator & API Sync

**Files:**
- Modify: `kisholens/ml/similarity.py:510-550`
- Modify: `kisholens/api/main.py:375-385`
- Test: `tests/ml/test_similarity.py`

**Interfaces:**
- Produces: `find_top_matches(...) -> List[Dict]` with `"reasons": List[str]` in each match dictionary.
- Updates: `get_novel_stats(novel_id)` to return updated `top_matches` containing `reasons` and `breakdown`.

- [ ] **Step 1: Write failing test for match reasons in `tests/ml/test_similarity.py`**

```python
# In tests/ml/test_similarity.py
def test_match_reasons_generation():
    text = """The rain beat against the fog-stained windowpanes of 221B Baker Street as Inspector Lestrade threw open the heavy oak door. "Holmes, you must come at once," he gasped."""
    feat = extract_english_features(text)
    matches = find_top_matches(feat, query_text=text, top_k=5)
    
    assert len(matches) > 0
    for m in matches:
        assert "reasons" in m
        assert isinstance(m["reasons"], list)
        assert len(m["reasons"]) >= 1
        # Confirm no emojis/icons are present
        for reason in m["reasons"]:
            assert isinstance(reason, str)
            assert all(ord(c) < 128 or '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in reason), f"Found emoji or non-text char in reason: {reason}"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/ml/test_similarity.py::test_match_reasons_generation -v`
Expected: FAIL with `KeyError: 'reasons'`.

- [ ] **Step 3: Implement reason synthesis in `kisholens/ml/similarity.py` and API sync in `kisholens/api/main.py`**

In `kisholens/ml/similarity.py`:
```python
        # Generate clean human-readable match reasons (no emojis)
        reasons = []
        if style_sim >= 0.88:
            reasons.append("Similar prose style & sentence structure")
        elif style_sim >= 0.75:
            reasons.append("Comparable sentence cadence")

        if q_primary_genre_lower and cand_primary and q_primary_genre_lower == cand_primary:
            reasons.append(f"Matching primary archetype: {n_meta.get('primary_genre') or n_meta.get('genre')}")
        elif q_primary_genre_lower and q_primary_genre_lower in cand_genres:
            reasons.append(f"Shared genre: {q_primary_genre}")
        elif genre_sim >= 0.60:
            reasons.append("Strong genre overlap")

        if semantic_sim >= 0.80:
            reasons.append("Closely aligned plot premise & themes")
        elif semantic_sim >= 0.65:
            reasons.append("Thematic narrative overlap")

        if territory_sim >= 0.85 and n_territory and n_territory != "Unknown":
            if "classic" in n_territory.lower():
                reasons.append("Shared Classic Literature tradition")
            elif "web" in n_territory.lower():
                reasons.append("Shared Web Novel territory")

        if tag_sim >= 0.60:
            reasons.append("Overlapping narrative tropes")

        if not reasons:
            reasons.append("Overall stylistic and structural affinity")
```

In `kisholens/api/main.py`:
```python
    if novel_id in _cached_novel_stats:
        stats = _cached_novel_stats[novel_id]
        if not stats.get("top_matches") or not any(m.get("reasons") for m in stats.get("top_matches", [])):
            from kisholens.ml.similarity import find_top_matches
            stats["top_matches"] = find_top_matches(stats, exclude_novel_id=novel_id, top_k=5)
        return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_similarity.py::test_match_reasons_generation -v`
Expected: PASS

- [ ] **Step 5: Commit Task 1**

```bash
git add kisholens/ml/similarity.py kisholens/api/main.py tests/ml/test_similarity.py
git commit -m "feat(similarity): generate descriptive match reasons and sync library stats endpoint"
```

---

### Task 2: Frontend Reason Pills & Factor Detail UI

**Files:**
- Modify: `frontend/src/pages/library.astro`
- Modify: `frontend/src/pages/analyze.astro`

**Interfaces:**
- Updates `renderSimilarNovels(matches)` in both files to render `.match-reason-pill` badges and factor breakdown metrics.

- [ ] **Step 1: Update CSS in `library.astro` and `analyze.astro`**

Add CSS styles for `.doppelganger-reasons`, `.reason-pill`, and interactive factor metrics without emojis.

- [ ] **Step 2: Update `renderSimilarNovels` in `library.astro` and `analyze.astro`**

Render reason pills under the title and author, and interactive breakdown bars.

- [ ] **Step 3: Build frontend static pages**

Run: `cd frontend && npm run build`
Expected: 3 pages built with 0 errors.

- [ ] **Step 4: Commit Task 2**

```bash
git add frontend/src/pages/library.astro frontend/src/pages/analyze.astro
git commit -m "feat(ui): display match reason pills and synchronized similarity factors on library and analyze pages"
```

---

### Task 3: Full Test Suite & End-to-End Verification

**Files:**
- Test: Full PyTest suite
- Test: Frontend static build

- [ ] **Step 1: Run full PyTest test suite**

Run: `uv run pytest tests/`
Expected: 68/68 passed.

- [ ] **Step 2: Run frontend build**

Run: `cd frontend && npm run build`
Expected: PASS (0 errors).

- [ ] **Step 3: Commit verification**

```bash
git commit --allow-empty -m "chore: verify end-to-end similarity reasons and library sync"
```
