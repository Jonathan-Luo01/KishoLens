# Design Specification: Granular Similar Works & Stylistic Match Explanations

## 1. Overview & Problem Statement
Currently, the Doppelgänger / Similar Works engine in KishoLens displays high-level composite match percentages (e.g., `94%`) accompanied by generic boilerplate text pills (such as `"Comparable sentence cadence"` or `"Matching primary archetype: Isekai"`). These badges lack granular quantitative evidence and specific narrative details explaining *why* two works share stylistic and narrative affinity.

This feature introduces **multi-dimensional metric delta chips**, **taxonomy/trope alignment badges**, and an **expandable "Why this matched" inspection drawer** on each similar novel card across both `/library` and `/analyze`.

---

## 2. Architecture & Data Flow

```
[Query Features & Text] + [Candidate Novel Metadata]
                      │
                      ▼
        [kisholens/ml/similarity.py]
  - 8D Radar Vector Cosine & L1 distance
  - 19-Metric Feature Deltas (|query - candidate|)
  - 3-Pillar Taxonomy & Trope Intersection
                      │
                      ▼
         [API Response: top_matches]
  - similarity_score (0.01 - 0.99)
  - match_badges: Array<{ type, label, detail, tier }>
  - metric_comparisons: Array<{ name, query_val, cand_val, delta, match_pct }>
  - breakdown: { style, semantic, genre, tags, territory }
                      │
                      ▼
      [Frontend: library.astro & analyze.astro]
  - Granular Badges: [Dialogue: 68% ≈ 65%] [Pacing: 10.3 ≈ 10.8 w/s] [Catalyst: Reincarnation]
  - Expandable "Why this matched ▾" drawer with side-by-side metric comparison table
```

---

## 3. Backend Specification (`kisholens/ml/similarity.py`)

### 3.1 Feature Comparison Logic
For every candidate novel compared against the query:
1. **Dialogue Ratio Delta**: If `abs(q_dialogue - c_dialogue) <= 0.12`, generate badge: `Dialogue: {round(q_dlg*100)}% ≈ {round(c_dlg*100)}%`.
2. **Average Sentence Length (Cadence) Delta**: If `abs(q_asl - c_asl) <= 3.5`, generate badge: `Cadence: {round(q_asl, 1)} ≈ {round(c_asl, 1)} w/s`.
3. **Lexical Diversity (TTR) Delta**: If `abs(q_ttr - c_ttr) <= 0.08`, generate badge: `Vocab: TTR {round(q_ttr, 2)} ≈ {round(c_ttr, 2)}`.
4. **Visceral Emotion (Somatic Density)**: If both have high somatic ratio (> 0.40), generate badge: `Visceral Imagery: {round(c_sbd*100)}%`.
5. **Thematic Depth**: If both have elevated moral/theme density (> 0.40), generate badge: `Thematic Depth: {round(c_theme*100)}%`.
6. **Temporal Complexity / Non-Linearity**: If linearity subversion matches within 0.06: `Temporal Arc: Non-Linear`.
7. **3-Pillar Taxonomy Catalyst & Trope Match**:
   - Extract matching catalyst (`Reincarnation`, `Summons`, `Murder Investigation`, `Betrayal`, `Tournament`).
   - Extract matching setting (`Fantasy Spire`, `Victorian Urban`, `Space Colony`, `Ancient Realm`).
   - Generate badge: `Catalyst: {catalyst}` or `Setting: {setting}`.

### 3.2 Structured Output Schema
Each item in `top_matches` includes:
```json
{
  "id": 102,
  "title": "Noble Reincarnation",
  "author": "Miki Nazuna",
  "genre": "Isekai, Fantasy",
  "territory": "Web Novel",
  "similarity_score": 0.94,
  "match_badges": [
    { "type": "metric", "label": "Dialogue", "detail": "68% ≈ 65%", "tier": "cyan" },
    { "type": "metric", "label": "Cadence", "detail": "10.3 ≈ 10.8 w/s", "tier": "purple" },
    { "type": "taxonomy", "label": "Catalyst", "detail": "Reincarnation", "tier": "amber" },
    { "type": "trope", "label": "Archetype", "detail": "Isekai", "tier": "emerald" }
  ],
  "metric_comparisons": [
    { "metric": "Dialogue Density", "query": "67.6%", "candidate": "64.8%", "match": "96%" },
    { "metric": "Sentence Cadence", "query": "10.3 w/s", "candidate": "10.8 w/s", "match": "95%" },
    { "metric": "Lexical Diversity (TTR)", "query": "0.47", "candidate": "0.45", "match": "95%" },
    { "metric": "Visceral Emotion", "query": "71.1%", "candidate": "68.2%", "match": "96%" },
    { "metric": "Thematic Explicitness", "query": "2.79", "candidate": "2.95", "match": "94%" }
  ],
  "breakdown": {
    "style": 0.94,
    "genre": 0.92,
    "semantic": 0.91,
    "tags": 0.88,
    "territory": 0.95
  }
}
```

---

## 4. Frontend Specification (`library.astro`, `analyze.astro`, `global.css`)

### 4.1 Card Header & Badges
- **Title, Author, Primary Genre Badge**.
- **Categorized Match Badges**:
  - `tier-cyan`: Stylistic metric deltas (`Dialogue: 68% ≈ 65%`).
  - `tier-purple`: Pacing & structural cadence (`Cadence: 10.3 ≈ 10.8 w/s`).
  - `tier-amber`: Taxonomy catalyst / narrative premise (`Catalyst: Reincarnation`).
  - `tier-emerald`: Genre / archetype affiliation (`Archetype: Isekai`).
- **Composite Score & 5-Factor Mini Bar Indicators** (`STY`, `GEN`, `TAG`, `TER`, `SEM`).

### 4.2 Expandable "Why this matched" Deep-Dive Drawer
- Button: `Why this matched ▾`
- When toggled:
  - Expands a clean recessed well (`rgba(240, 249, 255, 0.75)` in light mode, `rgba(0, 0, 0, 0.35)` in dark mode).
  - Displays a side-by-side comparison table showing:
    - **Metric Name**
    - **Your Prose / Selected Work Value**
    - **Matched Novel Value**
    - **Alignment % Bar**

---

## 5. Verification & Testing
1. **Unit Tests**: `uv run pytest tests/test_similarity.py` verifying structured `match_badges` and `metric_comparisons`.
2. **Astro Build**: `cd frontend && npm run build` (0 errors).
3. **Visual Verification**: `agent-browser` screenshots testing both `/library` and `/analyze` in Light and Dark themes, verifying badge contrast, drawer expansion, and accurate metric comparisons.
