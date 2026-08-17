# Editorial Matrix & Natural Literary Narrative Reasoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the spacious, zero-truncation Editorial Matrix UI and the human-grade literary narrative reasoning engine across the KishoLens library, analysis drawer, and backend ML similarity pipeline.

**Architecture:** 
- Backend ML (`kisholens/ml/similarity.py`): Natural editorial commentary generator eliminating AI clichés and repetitive tag echoes.
- Batch script (`scripts/recalculate_all_similarities.py`): Hydrates all 10,320 novels in `data/stats_cache.json` with fresh editorial matrix data.
- Frontend (`library.astro`, `analyze.astro`, `global.css`): Zero-truncation multi-line capsule tags, progress meters, thematic icons, and light/dark theme styling.

**Tech Stack:** Python 3.14, FastAPI, SentenceTransformers, Astro 5.16, TypeScript, Vanilla CSS.

## Global Constraints
- Zero text truncation (`...` / ellipsis) on narrative pillar values.
- Ban robotic filler phrases (*"thematic beats"*, *"anchored by a catalyst"*, *"socio-political hierarchy"*, *"richly drawn backdrop"*).
- All tests in `tests/ml/test_descriptive_similarity.py` and `pytest` must pass 100%.
- Astro frontend static build must pass in under 300ms with zero errors.

---

### Task 1: Natural Editorial Narrative Synthesis & Refined 4-Pillar Rationale Backend

**Files:**
- Modify: `kisholens/ml/similarity.py:800-1050`
- Test: `tests/ml/test_descriptive_similarity.py`

**Interfaces:**
- Consumes: `_infer_query_anatomy()`, `match_semantic()`, `extract_feature_vector()`
- Produces: `narrative_reasoning` dictionary with `narrative_synthesis`, `pillars`, `shared_tropes` formatted according to the spec.

- [ ] **Step 1: Write unit tests verifying natural human phrasing and zero robotic clichés**

```python
# In tests/ml/test_descriptive_similarity.py
def test_editorial_natural_phrasing():
    from kisholens.ml.similarity import _generate_narrative_synthesis, _compute_4pillar_breakdown
    q_anat = {
        "catalyst": "Reincarnation into Imperial Nobility",
        "setting": "High Fantasy Imperial Court & Aristocracy",
        "conflict": "Imperial Succession & Concealing Overpowered Might",
        "tropes": ["Overpowered Protagonist", "Reincarnation"]
    }
    c_anat = {
        "catalyst": "Villainess Subversion Reincarnation",
        "setting": "Otome Aristocratic Empire",
        "conflict": "Subverting Doom & Aristocratic Ruin",
        "tropes": ["Villainess Route", "Reincarnation"]
    }
    synth = _generate_narrative_synthesis(q_anat, c_anat, 0.92, 0.90, is_user_input=False)
    
    # Assert no robotic fillers
    robotic_words = ["thematic beats", "anchored by a", "socio-political hierarchy", "richly drawn backdrop"]
    for word in robotic_words:
        assert word not in synth.lower(), f"Found robotic filler: {word}"
        
    # Check 4-pillar breakdown
    q_m = {"dialogue_ratio": 0.68, "avg_sentence_len": 10.3}
    c_m = {"dialogue_ratio": 0.63, "avg_sentence_len": 8.7}
    pillars = _compute_4pillar_breakdown(q_anat, c_anat, q_m, c_m, 0.92, 0.90, 0.85)
    for p_key, p_data in pillars.items():
        assert "explanation" in p_data
        for word in robotic_words:
            assert word not in p_data["explanation"].lower()
```

- [ ] **Step 2: Run test to verify it fails or runs**

Run: `./.venv/bin/pytest tests/ml/test_descriptive_similarity.py -v`
Expected: Passes or fails with clear assertion.

- [ ] **Step 3: Implement refined `_generate_narrative_synthesis` and `_compute_4pillar_breakdown`**

Ensure `_generate_narrative_synthesis` and `_compute_4pillar_breakdown` generate natural, human-grade literary commentary.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/ml/test_descriptive_similarity.py -v`
Expected: 100% PASS.

- [ ] **Step 5: Commit**

```bash
git add kisholens/ml/similarity.py tests/ml/test_descriptive_similarity.py
git commit -m "feat(similarity): implement natural editorial synthesis and non-robotic 4-pillar rationale"
```

---

### Task 2: Batch Recomputation of All 10,320 Novels in Disk Cache

**Files:**
- Modify: `scripts/recalculate_all_similarities.py`
- Target: `data/stats_cache.json`

**Interfaces:**
- Consumes: Updated `_infer_query_anatomy`, `_generate_narrative_synthesis`, `_compute_4pillar_breakdown` from `kisholens.ml.similarity`.
- Produces: Updated `data/stats_cache.json` with freshly populated `narrative_reasoning` objects for all 10,320 novels.

- [ ] **Step 1: Run batch recalculator script**

Run: `./.venv/bin/python scripts/recalculate_all_similarities.py`
Expected: Outputs `Successfully updated all 10320 novels in .../data/stats_cache.json`.

- [ ] **Step 2: Validate sample cache entries for Novel 1 and match entries**

Run:
```bash
./.venv/bin/python -c '
import json
with open("data/stats_cache.json") as f:
    cache = json.load(f)
novel1 = cache["1"]
assert len(novel1["top_matches"]) > 0
top1 = novel1["top_matches"][0]
print("Novel 1 Top Match:", top1["title"])
print("Reasoning:", json.dumps(top1["narrative_reasoning"], indent=2))
'
```
Expected: Clean narrative reasoning with no robotic fillers and valid pillar entries.

- [ ] **Step 3: Commit**

```bash
git add scripts/recalculate_all_similarities.py
git commit -m "chore(cache): recalculate all 10,320 novel matches with editorial narrative reasoning"
```

---

### Task 3: Editorial Matrix Frontend Layout & Styling

**Files:**
- Modify: `frontend/src/pages/library.astro`
- Modify: `frontend/src/pages/analyze.astro`
- Modify: `frontend/src/styles/global.css`

**Interfaces:**
- Consumes: `m.narrative_reasoning` payload from API response.
- Produces: Clean, multi-line wrapping Editorial Matrix cards with icons, score progress bars, capsule chips, and light/dark theme support.

- [ ] **Step 1: Update `global.css` to remove text-overflow ellipsis from capsule values and support natural multi-line wrapping**

Ensure `.capsule-value` uses `word-break: break-word; white-space: normal; line-height: 1.35;`.

- [ ] **Step 2: Update `library.astro` and `analyze.astro` drawer rendering**

Render the Editorial Matrix with spacious capsule layout, progress bars, and thematic icons.

- [ ] **Step 3: Build frontend to verify build correctness**

Run: `npm --prefix frontend run build`
Expected: 3 static pages built in < 300ms with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/library.astro frontend/src/pages/analyze.astro frontend/src/styles/global.css
git commit -m "feat(frontend): implement zero-truncation Editorial Matrix layout with natural multi-line wrapping"
```

---

### Task 4: Full Verification & Live API Validation

**Files:**
- Test: All backend tests and frontend static build.

- [ ] **Step 1: Run full pytest suite**

Run: `./.venv/bin/pytest -v`
Expected: 76+ tests passing 100%.

- [ ] **Step 2: Verify live API endpoint with TestClient**

Run:
```bash
./.venv/bin/python -c '
from fastapi.testclient import TestClient
from kisholens.api.main import app
client = TestClient(app)
res = client.get("/api/novels/1/stats")
assert res.status_code == 200
data = res.json()
assert "top_matches" in data
print("Live API verified successfully!")
'
```
Expected: `Live API verified successfully!`

- [ ] **Step 3: Commit any final polish**

```bash
git status
```
