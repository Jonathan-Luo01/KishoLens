# Design Specification: Dual-Vector + Semantic Concept Architecture

## 1. Overview & Objectives

This specification defines the refinement of KishoLens's classification architecture in `kisholens/ml/` to resolve **Premise Dilution**.

### Problem
Previously, Inciting Events (such as *Isekai & Regression*, *System Initialization*, and *Cultivation Awakening*) suffered confidence score degradation when averaged across mid-to-late story chapters whose prose shifts into daily world/plot interactions. Additionally, relying on regex string-matching to boost scores was brittle for creative prose.

### Solution
Implement a **Dual-Vector + Semantic Concept Architecture**:
1. **Dual-Scope Vector Generation**: Generate an **Introductory Vector** ($V_{\text{intro}}$) for premise/inciting event evaluation, and a **Sustained Vector** ($V_{\text{sustained}}$) for persistent world/setting and narrative plot evaluation.
2. **Pure Concept Vectors**: Define dense, canonical semantic descriptions for inciting events, embed them using `all-MiniLM-L6-v2`, and calculate proportional confidence boosts based on semantic density rather than regex string matching.
3. **Structured Classification Output**: Evaluate Inciting Events, World Settings, and Narrative Plots independently with graceful threshold fallback (`< 0.55`).

---

## 2. File & Module Structure

The implementation creates three dedicated modules in `kisholens/ml/` while keeping `kisholens/ml/semantic_match.py` as a backward-compatible adapter:

```
kisholens/ml/
├── embeddings.py      # Dual-scope vector generation & dynamic weight redistribution
├── centroids.py       # Centroid loading & Pure Concept Vector definitions (INCITING_CONCEPTS)
├── analyzer.py        # Independent scoring (Intro vs Sustained), dynamic concept boost, & thresholding
└── semantic_match.py  # Backward-compatible adapter for existing API endpoints
```

---

## 3. Detailed Specification

### Task 1: Dual-Scope Vector Generation (`kisholens/ml/embeddings.py`)

Generate two 384-dimensional float32 unit vectors ($V_{\text{intro}}$ and $V_{\text{sustained}}$) from text samples using `all-MiniLM-L6-v2`.

#### Scenario A: Synopsis is Present (Database / Full Input)
* **Intro Vector**:
  $$V_{\text{intro}} = \text{Normalize}(0.60 \cdot V_{\text{synopsis}} + 0.40 \cdot V_{\text{Ch1}})$$
* **Sustained Vector**:
  $$V_{\text{sustained}} = \text{Normalize}(0.10 \cdot V_{\text{synopsis}} + 0.10 \cdot V_{\text{Ch1}} + 0.40 \cdot V_{\text{Ch10}} + 0.40 \cdot V_{\text{Ch20}})$$

#### Scenario B: Synopsis is Missing (Raw User Text Paste)
* **Intro Vector**:
  $$V_{\text{intro}} = \text{Normalize}(1.0 \cdot V_{\text{Ch1 (or First 500 words)}})$$
* **Sustained Vector**:
  $$V_{\text{sustained}} = \text{Normalize}(0.20 \cdot V_{\text{Ch1}} + 0.40 \cdot V_{\text{Ch10}} + 0.40 \cdot V_{\text{Ch20}})$$

*(Note: `Ch1`, `Ch10`, and `Ch20` represent proportional positional samples across the text volume: Beginning $\approx 0\%$, Middle $\approx 50\%$, End $\approx 100\%$).*

---

### Task 2: Pure Concept Vector Definitions (`kisholens/ml/centroids.py`)

Define dense semantic concept descriptions for inciting setup events:

```python
INCITING_CONCEPTS = {
    "Isekai & Regression": (
        "The protagonist dies and is reincarnated, opens their eyes and finds themselves "
        "in a fantasy/game/other world, transmigrated into a novel or game as a villainess or mob character, "
        "summoned to another world as a hero, or regresses back in time to their past life for a second chance at changing their fate."
    ),
    "System Initialization": (
        "A mysterious system interface suddenly appears before the protagonist's eyes, "
        "granting them a status window, levels, skills, and quests. The world undergoes an apocalyptic evolution "
        "or shifts into a game-like reality with dungeons and monsters."
    ),
    "Cultivation Awakening": (
        "The protagonist discovers a heaven-defying cheat artifact, awakens a supreme spiritual root, "
        "or repairs their crippled meridians to begin their journey on the path of cultivation, martial arts, and immortality."
    )
}
```

At module import, each concept string is embedded via `all-MiniLM-L6-v2` into a normalized 384D float32 vector ($V_{\text{concept\_<key>}}$).

---

### Task 3: Independent Scoring & Dynamic Multiplier (`kisholens/ml/analyzer.py`)

#### A. Score Worlds & Plots (The States)
Calculate cosine similarity between $V_{\text{sustained}}$ and standard genre centroids (Fantasy, Cultivation, Sci-Fi, Slice of Life, Mystery, etc.).

#### B. Score Inciting Events (The Events)
For each concept in `INCITING_CONCEPTS`:
1. **Base Score**: $S_{\text{base}} = \cos(V_{\text{intro}}, V_{\text{book\_centroid}})$
2. **Concept Density Score**: $S_{\text{concept}} = \cos(V_{\text{intro}}, V_{\text{concept\_vector}})$
3. **Proportional Boost**:
   ```python
   if S_concept > 0.20:
       dynamic_boost = min(0.25, S_concept * 0.50)
       final_score = min(0.99, S_base + dynamic_boost)
   else:
       final_score = S_base
   ```
4. **Graceful Fallback**: If the highest `final_score` across inciting concepts is `< 0.55`, omit `inciting_event` (`"inciting_event": null`).

---

### Task 4: Output Schema Integration

`analyze_prose()` returns the structured taxonomy dictionary:

```json
{
  "inciting_event": {
    "primary": "Isekai & Regression",
    "score": 0.94
  },
  "world_setting": {
    "primary": "Cultivation",
    "score": 0.88
  },
  "narrative_plot": {
    "primary": "Slice of Life",
    "score": 0.83
  },
  "display_label": "Isekai & Regression Cultivation (Slice of Life)"
}
```

---

## 4. Verification & Test Plan

1. **Unit Tests (`tests/ml/test_analyzer.py`)**:
   * Verify Dual-Scope Vector generation formulas for Scenario A (with synopsis) and Scenario B (no synopsis).
   * Test Pure Concept Vector initialization and similarity computation.
   * Verify dynamic boost calculation and 0.55 threshold fallback (returns `null` when no strong inciting setup exists).
   * Test *Noble Reincarnation* sample: verify `inciting_event` returns `"Isekai & Regression"` ($\ge 0.85$).
2. **Regression Testing**:
   * Run `uv run pytest tests/` to ensure all existing test suites pass.
