# Design Specification: Library Similarity Sync & Match Reasons Engine

**Date:** 2026-08-11  
**Author:** Antigravity Team  
**Status:** Proposed  
**Target Files:** `kisholens/ml/similarity.py`, `kisholens/api/main.py`, `frontend/src/pages/library.astro`, `frontend/src/pages/analyze.astro`, `tests/ml/test_similarity.py`

---

## 1. Executive Summary & Problem Statement

Currently, the Analyze Prose page uses the newly overhauled 5-factor similarity engine, but the Library page could retrieve stale or unhydrated match objects from `data/stats_cache.json` that lack feature breakdowns. Furthermore, newcomers and users need clear, accessible explanations for **why** each novel was recommended as a similar work (e.g., whether it matched due to sentence cadence, genre archetype, plot themes, or literary tradition).

---

## 2. Technical Architecture

### 2.1 Backend Reason Synthesis (`kisholens/ml/similarity.py`)
In `find_top_matches()`, alongside `similarity_score` and `breakdown`, compute a prioritized array of textual reasons (`reasons: List[str]`):
- **Prose & Style (`style >= 0.80`)**:
  - `style >= 0.88`: `"Similar prose style & sentence structure"`
  - `0.75 <= style < 0.88`: `"Comparable sentence cadence"`
- **Genre & Archetype (`genre >= 0.70` or primary match)**:
  - Exact primary match: `f"Matching primary archetype: {cand_primary}"`
  - In top genres: `f"Shared genre: {cand_primary}"`
  - Overlap: `"Strong genre overlap"`
- **Plot & Narrative Theme (`semantic >= 0.65`)**:
  - `semantic >= 0.80`: `"Closely aligned plot premise & themes"`
  - `0.65 <= semantic < 0.80`: `"Thematic narrative overlap"`
- **Literary Tradition (`territory >= 0.80`)**:
  - `"Shared Classic Literature tradition"` or `"Shared Web Novel territory"`
- **Tropes & Tags (`tags >= 0.60`)**:
  - `"Overlapping narrative tropes"`

*Note: No emojis or icons in strings or markup.*

### 2.2 FastAPI Endpoint Synchronization (`kisholens/api/main.py`)
In `get_novel_stats(novel_id)`:
- Detect if `_cached_novel_stats[novel_id]["top_matches"]` is missing or lacks `breakdown` / `reasons`.
- Dynamically call `find_top_matches(stats, exclude_novel_id=novel_id, top_k=5)` so that the Library endpoint returns the exact same data structure and accuracy as the Analyze Prose endpoint.

### 2.3 Frontend UI: Reason Pills & Detailed Factor Explanations
In both [`frontend/src/pages/library.astro`](file:///Users/jonathan/Documents/KishoLens/frontend/src/pages/library.astro) and [`frontend/src/pages/analyze.astro`](file:///Users/jonathan/Documents/KishoLens/frontend/src/pages/analyze.astro):
- **Compact Reason Pills**: Display clean, border-accented badges for key match reasons below the novel title.
- **Detailed Factor Breakdown**: Compact horizontal factor bars (`STY`, `GEN`, `TAG`, `TER`, `SEM`) with tooltips / expandable detail explaining the specific metric behind each score.
- **Typography & Style**: Pure typographic styling without emojis or icons, matching the sleek dark glassmorphic design system.

---

## 3. Verification & Acceptance Criteria

1. **Backend Matching & Reasons**:
   - Unit tests in `tests/ml/test_similarity.py` verify that `find_top_matches` returns `reasons` with clean text strings for Mystery, Isekai, and Romance queries.
2. **Library API Synchronization**:
   - Calling `/api/novels/{id}/stats` for any novel returns 5 top matches with full `breakdown` and `reasons`.
3. **Frontend Integration**:
   - `npm run build` compiles with 0 errors.
   - Similar novel cards in both Library and Analyze pages display the reason badges and factor breakdown bars.
