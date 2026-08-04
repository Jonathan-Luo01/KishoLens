# Classical Epic vs. Cultivation Guardrail Architecture Design

## 1. Overview & Objective

In KishoLens, Classical Epics (such as *Romance of the Three Kingdoms* or *Journey to the West*) have historically been misclassified as modern Xianxia Cultivation due to introductory Taoist mysticism, heavenly mandates, and magic scrolls in Chapter 1. 

Furthermore, the engine must gracefully handle **Hybrid Edge-Cases** (such as modern Kingdom-Building Cultivation web novels or Historical Fanfictions) where both macro-level military statecraft and hard progression mechanics co-exist.

This design introduces a **Classical Epic vs. Cultivation Guardrail** in `kisholens/pipeline/taxonomy.py` and `kisholens/ml/analyzer.py` that evaluates sustained text against two high-precision lexicons to differentiate "Taoist Flavor / Historical Mythology" from "Hard Progression Cultivation".

---

## 2. Lexicons & Module Shift (`kisholens/pipeline/taxonomy.py`)

We create `kisholens/pipeline/taxonomy.py` as the centralized taxonomy and lexicon registry for KishoLens.

### 2.1 Lexicons

```python
# Modern progression mechanics (Required for true hard Cultivation)
HARD_CULTIVATION_MARKERS = [
    "dantian", "meridian", "qi gathering", "foundation establishment", "nascent soul", 
    "pill refining", "spirit stone", "bottleneck", "breakthrough",
    "丹田", "经脉", "炼气", "筑基", "元婴", "灵石", "突破", "瓶颈"
]

# Macro-level warfare, statecraft, and political mechanics
MILITARY_EPIC_MARKERS = [
    "army", "general", "emperor", "rebellion", "troops", "strategy", 
    "dynasty", "warlord", "mandate of heaven", "cavalry", "imperial court", 
    "marched", "camp", "siege",
    "将军", "軍隊", "军队", "皇帝", "朝廷", "叛乱", "叛亂", "天下", "诸侯", "諸侯", 
    "兵马", "兵馬", "城池", "谋士", "謀士", "官军", "官軍", "大将", "大將"
]
```

### 2.2 Shared Taxonomy Relocation
* Move `GENRE_TAXONOMY`, `ANCHOR_TERMS`, `detect_text_language`, and `scan_anchor_boosts` from `kisholens/ml/semantic_match.py` into `kisholens/pipeline/taxonomy.py`.
* In `kisholens/ml/semantic_match.py`, re-export these symbols to preserve complete backward compatibility.

### 2.3 Pattern Matching Logic
* **English terms**: Case-insensitive with regex word boundaries (`\bterm\b`).
* **Chinese / Japanese (CJK) characters**: Substring containment matching against text.

---

## 3. Disambiguation Guardrail Logic (`kisholens/ml/analyzer.py`)

In `analyze_prose()`, after vector scoring and concept density evaluations, if `Cultivation` is currently winning or scoring highly (`Cultivation score > 0.65` or ranked #1):

1. **Scan Sustained Text**: Count matched items in `HARD_CULTIVATION_MARKERS` and `MILITARY_EPIC_MARKERS` across `full_scan_text` (`ch1_text + ch10_text + ch20_text`).
2. **Apply Guardrail Rules**:

### Scenario A: Pure Classical Epic (The False Positive)
* **Conditions**:
  - `MILITARY_EPIC_MARKERS_COUNT >= 3`
  - `HARD_CULTIVATION_MARKERS_COUNT == 0`
* **Action**:
  - Apply **-0.40 penalty** to `Cultivation` score.
  - Apply **+0.20 boost** to `Historical` score.
  - Set `Historical` as `world_setting` (or top primary genre).
  - Format `display_label`: `"Historical Action / Adventure (Military Epic)"` (or `"(Military Epic)"` tag).

### Scenario B: Hybrid Historical Cultivation (Kingdom Building / Fanfiction)
* **Conditions**:
  - `MILITARY_EPIC_MARKERS_COUNT >= 3`
  - `HARD_CULTIVATION_MARKERS_COUNT >= 2`
* **Action**:
  - **No penalty** applied to `Cultivation`.
  - Forcefully assign `Historical` as the secondary narrative plot / setting (`narrative_plot: {"primary": "Historical / Military", "score": ...}`).
  - Format `display_label`: `"Historical Cultivation (Kingdom Building / Military)"`.

---

## 4. Updated JSON Output Schema

### Scenario A Output (Pure Classical Epic):
```json
{
  "taxonomy": {
    "inciting_event": null,
    "world_setting": { "primary": "Historical", "score": 0.88 },
    "narrative_plot": { "primary": "Action / Adventure", "score": 0.85 },
    "display_label": "Historical Action / Adventure (Military Epic)"
  }
}
```

### Scenario B Output (Hybrid Cultivation):
```json
{
  "taxonomy": {
    "inciting_event": { "primary": "Cultivation Awakening", "score": 0.82 },
    "world_setting": { "primary": "Cultivation", "score": 0.91 },
    "narrative_plot": { "primary": "Historical / Military", "score": 0.84 },
    "display_label": "Historical Cultivation (Kingdom Building / Military)"
  }
}
```

---

## 5. Test Plan

1. **Unit Tests (`tests/pipeline/test_taxonomy.py`)**:
   * Test marker count scanning for English and CJK text.
   * Test `evaluate_epic_cultivation_guardrail()` returns correct scenario verdicts.

2. **Integration Tests (`tests/ml/test_analyzer.py`)**:
   * Verify *Romance of the Three Kingdoms* (Novel ID 231) evaluates to Scenario A (`Historical Action / Adventure (Military Epic)`).
   * Verify a hybrid kingdom-building cultivation sample evaluates to Scenario B (`Historical Cultivation (Kingdom Building / Military)`).
   * Verify standard Xianxia novels (e.g. *A Will Eternal*) retain pure `Cultivation`.

3. **Full PyTest Suite**:
   * Execute `uv run pytest tests/` and confirm all 48+ tests pass.
