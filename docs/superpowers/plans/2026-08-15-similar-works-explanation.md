# Granular Similar Works & Stylistic Match Explanations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform similar works / doppelgänger cards into granular, evidence-based comparisons featuring color-coded metric delta chips, taxonomy catalyst badges, and an interactive side-by-side metric comparison drawer.

**Architecture:** Extend backend similarity calculation in `kisholens/ml/similarity.py` to extract dimension-level differences and structured badges; update frontend renderers in `library.astro` and `analyze.astro` with interactive accordion comparison drawers and adaptive light/dark theme CSS in `global.css`.

**Tech Stack:** Python (FastAPI, NumPy, SQLModel), Astro, Vanilla CSS, TypeScript/JS, agent-browser.

## Global Constraints

- Must preserve all existing similarity scoring mathematics and weights (30% style, 35% genre, 20% semantic, 5% tags, 10% territory).
- Match badges must be specific, quantitative, and free of generic clichés or emoji decorations.
- The UI must adapt seamlessly to Light and Dark mode using the established KishoLens design system tokens.
- All builds must pass `cd frontend && npm run build` with 0 errors.

---

### Task 1: Backend Granular Match Badges & Metric Comparisons Generation

**Files:**
- Modify: `kisholens/ml/similarity.py:520-584`
- Test: `tests/test_similarity.py`

**Interfaces:**
- Consumes: `query_features: dict`, `query_text: Optional[str]`, `candidate_items`
- Produces: Each candidate dict containing `similarity_score: float`, `match_badges: List[dict]`, `metric_comparisons: List[dict]`, `reasons: List[str]`, `breakdown: dict`.

- [ ] **Step 1: Write the failing unit test for match badges and metric comparisons**

```python
# In tests/test_similarity.py
def test_granular_match_badges_and_comparisons():
    from kisholens.ml.similarity import find_top_matches
    q_feats = {
        "en_dialogue_ratio": 0.65,
        "en_avg_sentence_length": 11.2,
        "en_ttr": 0.46,
        "en_sensory_body_density": 0.70,
        "en_theme_explication_ratio": 2.8,
        "genre": "Fantasy, Isekai",
        "territory": "Web Novel"
    }
    matches = find_top_matches(q_feats, query_text="Hero summoned to another world", top_k=3)
    assert len(matches) > 0
    top = matches[0]
    assert "match_badges" in top
    assert isinstance(top["match_badges"], list)
    assert len(top["match_badges"]) >= 1
    assert "type" in top["match_badges"][0]
    assert "metric_comparisons" in top
    assert isinstance(top["metric_comparisons"], list)
    assert len(top["metric_comparisons"]) >= 1
    assert "metric" in top["metric_comparisons"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_similarity.py -k test_granular_match_badges_and_comparisons -v`
Expected: FAIL with missing `match_badges` / `metric_comparisons`.

- [ ] **Step 3: Implement `_compute_match_badges` and `_compute_metric_comparisons` in `kisholens/ml/similarity.py`**

Compute detailed metric comparisons:
- Dialogue ratio: `{ "metric": "Dialogue Density", "query": f"{q_dlg*100:.1f}%", "candidate": f"{c_dlg*100:.1f}%", "match": f"{max(0, 100 - abs(q_dlg-c_dlg)*200):.0f}%" }`
- Sentence cadence: `{ "metric": "Sentence Cadence", "query": f"{q_asl:.1f} w/s", "candidate": f"{c_asl:.1f} w/s", "match": f"{max(0, 100 - abs(q_asl-c_asl)*10):.0f}%" }`
- Lexical richness: `{ "metric": "Lexical Richness (TTR)", "query": f"{q_ttr:.2f}", "candidate": f"{c_ttr:.2f}", "match": f"{max(0, 100 - abs(q_ttr-c_ttr)*300):.0f}%" }`
- Visceral emotion: `{ "metric": "Visceral Somatic Imagery", "query": f"{q_sbd*100:.1f}%", "candidate": f"{c_sbd*100:.1f}%", "match": f"{max(0, 100 - abs(q_sbd-c_sbd)*200):.0f}%" }`
- Thematic depth: `{ "metric": "Thematic Explicitness", "query": f"{q_theme:.2f}", "candidate": f"{c_theme:.2f}", "match": f"{max(0, 100 - abs(q_theme-c_theme)*20):.0f}%" }`

Generate `match_badges`:
- `type: "metric"`, `label: "Dialogue"`, `detail: f"{q_dlg*100:.0f}% ≈ {c_dlg*100:.0f}%"`, `tier: "cyan"`
- `type: "metric"`, `label: "Cadence"`, `detail: f"{q_asl:.1f} ≈ {c_asl:.1f} w/s"`, `tier: "purple"`
- `type: "taxonomy"`, `label: "Catalyst"`, `detail: catalyst_name`, `tier: "amber"`
- `type: "trope"`, `label: "Archetype"`, `detail: primary_genre`, `tier: "emerald"`

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_similarity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kisholens/ml/similarity.py tests/test_similarity.py
git commit -m "feat(similarity): generate granular metric delta badges and side-by-side comparison metadata"
```

---

### Task 2: Doppelgänger Component & Interactive Comparison Drawer in `library.astro` and `global.css`

**Files:**
- Modify: `frontend/src/pages/library.astro:4390-4460`
- Modify: `frontend/src/styles/global.css`

**Interfaces:**
- Consumes: `m.match_badges`, `m.metric_comparisons`, `m.breakdown`
- Produces: Expandable card UI with color-coded chip rows and interactive why-this-matched drawer.

- [ ] **Step 1: Add CSS styles for categorized match pills, why-this-matched button, and metric comparison table in `global.css`**

Add styles for:
- `.reason-pill.tier-cyan`, `.reason-pill.tier-purple`, `.reason-pill.tier-amber`, `.reason-pill.tier-emerald`
- `.doppelganger-why-btn` with hover scale and chevron rotation
- `.doppelganger-drawer` collapsible animation and glassmorphic background
- `.metric-compare-table`, `.metric-compare-row`, `.metric-compare-bar`

- [ ] **Step 2: Update `renderSimilarNovels(matches)` in `library.astro`**

Render:
- `m.match_badges` as categorized pills.
- Add `<button type="button" class="doppelganger-why-btn" data-drawer-id="drawer-${m.id}">Why this matched <span class="why-chevron">▾</span></button>`.
- Render `<div id="drawer-${m.id}" class="doppelganger-drawer" style="display: none;">` with `m.metric_comparisons`.
- Wire up inline click toggle handler preventing bubbling to novel selection.

- [ ] **Step 3: Run frontend build to verify compilation**

Run: `cd frontend && npm run build`
Expected: PASS with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/library.astro frontend/src/styles/global.css
git commit -m "feat(library): implement rich categorized match chips and interactive why-this-matched drawer"
```

---

### Task 3: Doppelgänger Component & Interactive Comparison Drawer in `analyze.astro`

**Files:**
- Modify: `frontend/src/pages/analyze.astro:3040-3110`

**Interfaces:**
- Consumes: `m.match_badges`, `m.metric_comparisons`, `m.breakdown`
- Produces: Expandable card UI in analyze prose output pane.

- [ ] **Step 1: Update `renderSimilarNovels(matches)` in `analyze.astro`**

Align `renderSimilarNovels` with `library.astro` to render `match_badges` and the expandable `Why this matched` drawer.

- [ ] **Step 2: Run frontend build to verify compilation**

Run: `cd frontend && npm run build`
Expected: PASS with 0 errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/analyze.astro
git commit -m "feat(analyze): implement rich categorized match chips and interactive why-this-matched drawer"
```

---

### Task 4: End-to-End Visual Verification & Polish

**Files:**
- Test via `agent-browser` on `http://localhost:4321/library` and `http://localhost:4321/analyze`

- [ ] **Step 1: Test `/library` in Light and Dark mode**
  - Select novel, verify `match_badges` render with quantitative deltas.
  - Click `Why this matched ▾`, verify drawer expands and shows aligned comparison values.
- [ ] **Step 2: Test `/analyze` in Light and Dark mode**
  - Submit sample prose, verify `match_badges` and drawer operate seamlessly.
- [ ] **Step 3: Run full verification test suite**
  - `uv run pytest`
  - `cd frontend && npm run build`
