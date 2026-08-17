# Descriptive Story Similarity & 4-Pillar Narrative Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the similarity reasoning engine to produce a cohesive Narrative Synthesis Paragraph, a 4-Pillar Narrative Alignment Matrix (Catalyst, Setting, Conflict, Style Cadence), and Shared Trope Chips for both library novels and arbitrary raw user inputs.

**Architecture:** In `kisholens/ml/similarity.py`, implement dynamic story anatomy extraction (`_infer_query_anatomy`), narrative synthesis generation (`_generate_narrative_synthesis`), 4-pillar comparative evaluation (`_compute_4pillar_breakdown`), and trope matching (`_extract_shared_tropes`). Update frontend drawers in `library.astro` and `analyze.astro` to render the new structured narrative callouts and 4-pillar cards.

**Tech Stack:** Python 3.14 (FastAPI, SentenceTransformers, NumPy, pytest), Astro (Vanilla CSS/JS, HTML5).

## Global Constraints

- Must support both Database Novels (with rich metadata) and Raw User Input Text (with no metadata, dynamically inferring taxonomy and tropes).
- All similarity calculations must remain sub-millisecond per candidate pair.
- Frontend drawers must support both Dark Mode and Light Mode.
- All existing tests in `tests/` must pass.

---

### Task 1: Backend Story Anatomy & 4-Pillar Narrative Reasoning Engine

**Files:**
- Modify: `kisholens/ml/similarity.py`
- Create: `tests/ml/test_descriptive_similarity.py`

**Interfaces:**
- Produces: `narrative_reasoning` dictionary attached to every item in `find_top_matches()`:
  - `narrative_synthesis`: `str`
  - `pillars`: `dict` with `catalyst`, `setting`, `conflict`, `style_cadence`
  - `shared_tropes`: `list[str]`

- [ ] **Step 1: Write the failing unit tests for descriptive similarity reasoning**

```python
# tests/ml/test_descriptive_similarity.py
import pytest
from kisholens.ml.similarity import find_top_matches

def test_descriptive_similarity_for_database_novel():
    # Test matching for Noble Reincarnation (ID 1)
    matches = find_top_matches(target_novel_id=1, limit=3)
    assert len(matches) > 0
    top = matches[0]
    
    assert "narrative_reasoning" in top
    reasoning = top["narrative_reasoning"]
    assert "narrative_synthesis" in reasoning
    assert len(reasoning["narrative_synthesis"]) > 20
    
    assert "pillars" in reasoning
    pillars = reasoning["pillars"]
    assert "catalyst" in pillars
    assert "setting" in pillars
    assert "conflict" in pillars
    assert "style_cadence" in pillars
    
    assert "shared_tropes" in reasoning
    assert isinstance(reasoning["shared_tropes"], list)


def test_descriptive_similarity_for_raw_user_text():
    # Test matching for arbitrary user text without title or synopsis
    raw_text = """
    In a flash of blinding azure light, I opened my eyes in an ornate palace chamber.
    The grand duke stared down with cold calculation. "You have awakened, my son," he murmured.
    My previous life as an ordinary salaryman was gone; I was now the third prince in an empire teetering on civil war.
    """
    matches = find_top_matches(query_text=raw_text, limit=3)
    assert len(matches) > 0
    top = matches[0]
    
    assert "narrative_reasoning" in top
    reasoning = top["narrative_reasoning"]
    assert "narrative_synthesis" in reasoning
    assert "pillars" in reasoning
    assert pillars["catalyst"]["score"] > 0
    assert pillars["setting"]["score"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/ml/test_descriptive_similarity.py -v`
Expected: FAIL with missing `narrative_reasoning`

- [ ] **Step 3: Implement dynamic story anatomy inference and 4-pillar reasoner in `kisholens/ml/similarity.py`**

Implement:
1. `_infer_query_anatomy(query_text, query_semantic, query_features)`
2. `_generate_narrative_synthesis(q_anat, c_anat, s_sim, g_sim, is_user_input)`
3. `_compute_4pillar_breakdown(q_anat, c_anat, q_m, c_m, s_sim, g_sim, sty_sim)`
4. `_extract_shared_tropes(q_anat, c_anat)`
5. Integrate into `find_top_matches()` to attach `narrative_reasoning` to each returned candidate.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/ml/test_descriptive_similarity.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 1**

```bash
git add kisholens/ml/similarity.py tests/ml/test_descriptive_similarity.py
git commit -m "feat(similarity): implement hybrid narrative synthesis and 4-pillar story alignment reasoning"
```

---

### Task 2: Batch Recalculation Script & Cache Hydration

**Files:**
- Modify: `scripts/recalculate_all_similarities.py`

**Interfaces:**
- Consumes: `_infer_query_anatomy`, `_generate_narrative_synthesis`, `_compute_4pillar_breakdown`, `_extract_shared_tropes`
- Produces: Hydrated `data/stats_cache.json` with `narrative_reasoning` for all 10,320 novels.

- [x] **Step 1: Update `scripts/recalculate_all_similarities.py` to embed `narrative_reasoning`**

Include full `narrative_reasoning` dictionary in each top match when batch generating `data/stats_cache.json`.

- [x] **Step 2: Execute batch recalculation across all 10,320 novels**

Run: `./.venv/bin/python scripts/recalculate_all_similarities.py`
Expected: Successfully updated all 10,320 novels in `data/stats_cache.json`.

- [x] **Step 3: Commit Task 2**

```bash
git add scripts/recalculate_all_similarities.py
git commit -m "feat(cache): batch hydrated 4-pillar narrative reasoning across all 10,320 novels"
```

---

### Task 3: Frontend Visual Drawer Integration in Library & Analyze Pages

**Files:**
- Modify: `frontend/src/pages/library.astro`
- Modify: `frontend/src/pages/analyze.astro`
- Modify: `frontend/src/styles/global.css`

**Interfaces:**
- Consumes: `top_matches[i].narrative_reasoning` from `/api/novels/{id}/stats` and `/api/analyze`
- Produces: Rendered Narrative Synthesis Callout, 4-Pillar Narrative Alignment Cards, and Shared Trope Chips.

- [ ] **Step 1: Update `library.astro` and `analyze.astro` rendering functions**

In `renderSimilarNovels()`:
1. Render **📖 Narrative Synthesis Callout** (`.narrative-synthesis-box`).
2. Render **🏛️ 4-Pillar Narrative Alignment Grid** (`.pillars-grid` with `.pillar-card` for Catalyst, Setting, Conflict, and Style Cadence).
3. Render **🏷️ Shared Trope Chips** (`.shared-trope-chip`).
4. Retain side-by-side metric comparison table for deep stylometric inspection.

- [ ] **Step 2: Add high-contrast responsive styling in `global.css`**

Add CSS for `.narrative-synthesis-box`, `.pillars-grid`, `.pillar-card`, `.pillar-score-bar`, `.shared-trope-chip` for both dark and light modes.

- [ ] **Step 3: Verify frontend compilation and visual appearance in browser**

Run: `npm --prefix frontend run build`
Run: `agent-browser open http://localhost:4321/library` and `agent-browser open http://localhost:4321/analyze`
Verify: Drawer opens smoothly with synthesis text, 4-pillar breakdown, and trope badges.

- [ ] **Step 4: Commit Task 3**

```bash
git add frontend/src/pages/library.astro frontend/src/pages/analyze.astro frontend/src/styles/global.css
git commit -m "feat(frontend): render narrative synthesis and 4-pillar alignment cards in similarity drawer"
```

---

### Task 4: Full End-to-End Regression Verification

**Files:**
- Verify: `tests/`
- Verify: `frontend/`

- [ ] **Step 1: Run full backend test suite**

Run: `./.venv/bin/pytest`
Expected: 100% tests passing.

- [ ] **Step 2: Test live API endpoints**

Run: `curl -s http://localhost:8000/api/novels/1/stats | jq '.top_matches[0].narrative_reasoning'`
Verify: Contains narrative synthesis, pillars, and shared tropes.

- [ ] **Step 3: Final Commit**

```bash
git commit --allow-empty -m "chore: completed and verified descriptive story similarity engine"
```
