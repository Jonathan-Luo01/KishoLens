# Natural Multi-Archetype Editorial Synthesis & 4-Pillar Similarity Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Elevate similarity synthesis and 4-pillar narrative breakdown descriptions across all web novels and classic literature to eliminate cookie-cutter, repetitive, or synthetic phrasing.

**Architecture:** Refine story anatomy extraction by removing search context pollution, expanding taxonomy across 25+ distinct narrative archetypes, generating contrastive editorial synthesis, and computing nuanced 4-pillar explanations based on metric differentials.

**Tech Stack:** Python 3.12, Pytest, Astro, TypeScript, agent-browser.

## Global Constraints
- Do NOT commit or push to git (user constraint).
- Maintain 100% test pass rate with pytest (`./.venv/bin/pytest tests/ml/test_descriptive_similarity.py`).
- Full support for dark and light mode UI in `library.astro` and `analyze.astro`.

---

### Task 1: NLP Story Anatomy & 25+ Archetypes Parser (Backend)

**Files:**
- Modify: `kisholens/ml/similarity.py:700-900`
- Test: `tests/ml/test_descriptive_similarity.py`

**Interfaces:**
- Produces: `extract_dynamic_story_anatomy(query_features, query_semantic, query_text) -> dict` containing `catalyst`, `setting`, `conflict`, `tropes`.

- [ ] **Step 1: Write failing unit test for diverse subgenres**

In `tests/ml/test_descriptive_similarity.py`, add `test_dynamic_story_anatomy_subgenres` checking that:
- Slice of Life / Cozy novel produces pastoral/everyday catalyst and conflict (NOT `"Territorial Warfare"`).
- LitRPG / Hunter novel produces System Awakening catalyst and Gate Conquest conflict.
- Otome / Villainess novel produces Subversion catalyst and Death Flag conflict.
- Mystery novel produces Investigation catalyst and Unmasking Conspirators conflict.
- Xianxia novel produces Meridian/Cultivation catalyst and Sect Ascendancy conflict.

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/ml/test_descriptive_similarity.py -k test_dynamic_story_anatomy_subgenres -v`
Expected: FAIL (due to `territory` context pollution returning `"Territorial Warfare"` for Slice of Life).

- [ ] **Step 3: Implement clean context extraction & 25+ archetypes in `similarity.py`**

In `extract_dynamic_story_anatomy`:
- Remove `territory` from `context` string to prevent regex pollution.
- Expand catalyst, setting, conflict, and trope regex mappings with rich granular subgenres.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/ml/test_descriptive_similarity.py -k test_dynamic_story_anatomy_subgenres -v`
Expected: PASS.

---

### Task 2: Human-Grade Editorial Synthesis & 4-Pillar Card Rationale (Backend)

**Files:**
- Modify: `kisholens/ml/similarity.py:905-1120`
- Test: `tests/ml/test_descriptive_similarity.py`

**Interfaces:**
- Produces: `_generate_narrative_synthesis` and `_compute_4pillar_breakdown`.

- [ ] **Step 1: Write unit test asserting zero cookie-cutter repetitions and rich pillar commentary**

In `tests/ml/test_descriptive_similarity.py`, add `test_narrative_synthesis_and_pillar_diversity` asserting:
- No banned robotic phrases (`"thematic beats"`, `"richly drawn backdrop"`, `"factional friction and purposeful protagonist progression"`).
- 10 distinct novel pairs produce 10 unique, bespoke synthesis paragraphs and distinct pillar explanations.

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/ml/test_descriptive_similarity.py -k test_narrative_synthesis_and_pillar_diversity -v`
Expected: FAIL.

- [ ] **Step 3: Upgrade `_generate_narrative_synthesis` & `_compute_4pillar_breakdown`**

In `kisholens/ml/similarity.py`:
- Implement dynamic contrastive narrative synthesis that directly compares how Candidate approaches the story compared to Query.
- Implement rich, metric-driven rationale for Catalyst, Setting, Conflict, and Style Cadence (contrasting dialogue ratio and sentence length).

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/ml/test_descriptive_similarity.py -v`
Expected: ALL PASS (100%).

---

### Task 3: Batch Cache Recalculation & Cache Verification

**Files:**
- Run: `scripts/recalculate_all_similarities.py`
- Target: `data/stats_cache.json`

- [ ] **Step 1: Execute batch recalculation**

Run: `./.venv/bin/python scripts/recalculate_all_similarities.py`
Expected: Processes all 10,320 novels and updates `data/stats_cache.json`.

- [ ] **Step 2: Verify sample diversity in disk cache**

Run python verification script to check that novels across Slice of Life, Romance, Fantasy, Action, Mystery have rich, distinct, non-robotic synthesis and card rationale.

---

### Task 4: Visual Verification via `agent-browser`

- [ ] **Step 1: Test `/library` across multiple distinct novels (Novel 1, Novel 50, Novel 100, Novel 500)** in both Dark and Light modes.
- [ ] **Step 2: Test `/analyze` with Mystery, Fantasy, and Isekai drafts** in both Dark and Light modes.
- [ ] **Step 3: Confirm all cards render with zero truncation, high contrast, and natural editorial voice.**

