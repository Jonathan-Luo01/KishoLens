# Classical Epic vs. Cultivation Guardrail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Disambiguate Classical Historical/Military Epics (e.g. *Romance of the Three Kingdoms*) from modern Xianxia Cultivation, while cleanly supporting Hybrid Historical Cultivation / Kingdom Building fanfiction.

**Architecture:** Create a centralized taxonomy registry in `kisholens/pipeline/taxonomy.py` with `HARD_CULTIVATION_MARKERS` and `MILITARY_EPIC_MARKERS`. Intercept high-scoring Cultivation results in `kisholens/ml/analyzer.py` to enforce Scenario A (Pure Epic: -0.40 Cultivation penalty, +0.20 Historical boost) and Scenario B (Hybrid: set `Historical / Military` secondary plot).

**Tech Stack:** Python 3.10+, PyTest, NumPy, SQLModel, regex.

## Global Constraints

- Python files use type hints and clear docstrings.
- English term pattern matching must be case-insensitive with word boundaries (`\bterm\b`).
- Chinese/Japanese (CJK) pattern matching must check substring containment (`term in text`).
- All PyTest unit tests in `tests/` must pass cleanly without warnings.

---

### Task 1: Create Centralized Taxonomy & Guardrail Evaluator (`kisholens/pipeline/taxonomy.py`)

**Files:**
- Create: `kisholens/pipeline/taxonomy.py`
- Test: `tests/pipeline/test_taxonomy.py`

**Interfaces:**
- Consumes: Raw text string (`sustained_text`)
- Produces: `HARD_CULTIVATION_MARKERS`, `MILITARY_EPIC_MARKERS`, `evaluate_epic_cultivation_guardrail(text: str) -> dict`

- [ ] **Step 1: Write failing unit test for taxonomy marker scanning & guardrail evaluation**

Create `tests/pipeline/test_taxonomy.py`:
```python
from kisholens.pipeline.taxonomy import (
    HARD_CULTIVATION_MARKERS,
    MILITARY_EPIC_MARKERS,
    scan_marker_counts,
    evaluate_epic_cultivation_guardrail,
)

def test_scan_marker_counts():
    text_zh = "将军带领军队和诸侯，在突破关卡时遭到了叛乱。"
    counts = scan_marker_counts(text_zh)
    assert counts["military"] >= 3
    assert counts["cultivation"] == 0

def test_evaluate_epic_cultivation_guardrail_scenario_a():
    text = "The general led the emperor's troops and cavalry against the rebellion near the camp."
    res = evaluate_epic_cultivation_guardrail(text)
    assert res["scenario"] == "A"
    assert res["cultivation_penalty"] == -0.40
    assert res["historical_boost"] == 0.20

def test_evaluate_epic_cultivation_guardrail_scenario_b():
    text = "The emperor's general gathered qi in his dantian to achieve a breakthrough during the siege."
    res = evaluate_epic_cultivation_guardrail(text)
    assert res["scenario"] == "B"
    assert res["cultivation_penalty"] == 0.0
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/pipeline/test_taxonomy.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'kisholens.pipeline.taxonomy'`

- [ ] **Step 3: Write implementation of `kisholens/pipeline/taxonomy.py`**

Create `kisholens/pipeline/taxonomy.py`:
```python
"""
taxonomy.py — Centralized taxonomy definitions, marker lexicons, and guardrail evaluation.
"""

import re
from typing import Dict, Any, List

HARD_CULTIVATION_MARKERS = [
    "dantian", "meridian", "qi gathering", "foundation establishment", "nascent soul", 
    "pill refining", "spirit stone", "bottleneck", "breakthrough",
    "丹田", "经脉", "煉氣", "炼气", "築基", "筑基", "元嬰", "元婴", "靈石", "灵石", "突破", "瓶頸", "瓶颈"
]

MILITARY_EPIC_MARKERS = [
    "army", "general", "emperor", "rebellion", "troops", "strategy", 
    "dynasty", "warlord", "mandate of heaven", "cavalry", "imperial court", 
    "marched", "camp", "siege",
    "将军", "將軍", "军队", "軍隊", "皇帝", "朝廷", "叛乱", "叛亂", "天下", "诸侯", "諸侯", 
    "兵马", "兵馬", "城池", "谋士", "謀士", "官军", "官軍", "大将", "大將"
]

def scan_marker_counts(text: str) -> Dict[str, int]:
    """
    Counts matches for HARD_CULTIVATION_MARKERS and MILITARY_EPIC_MARKERS in text.
    Uses regex word-boundaries for English terms and substring containment for CJK characters.
    """
    low_text = text.lower()
    
    def _count_matches(markers: List[str]) -> int:
        matched = set()
        for pat in markers:
            if re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', pat):
                if pat in text:
                    matched.add(pat)
            else:
                regex = r'\b' + re.escape(pat) + r'\b'
                if re.search(regex, low_text):
                    matched.add(pat)
        return len(matched)

    return {
        "cultivation": _count_matches(HARD_CULTIVATION_MARKERS),
        "military": _count_matches(MILITARY_EPIC_MARKERS),
    }

def evaluate_epic_cultivation_guardrail(text: str) -> Dict[str, Any]:
    """
    Evaluates whether text is Scenario A (Pure Classical Epic) or Scenario B (Hybrid Cultivation).
    """
    counts = scan_marker_counts(text)
    mil_count = counts["military"]
    cult_count = counts["cultivation"]

    if mil_count >= 3 and cult_count == 0:
        return {
            "scenario": "A",
            "cultivation_penalty": -0.40,
            "historical_boost": 0.20,
            "display_tag": "(Military Epic)",
        }
    elif mil_count >= 3 and cult_count >= 2:
        return {
            "scenario": "B",
            "cultivation_penalty": 0.0,
            "historical_boost": 0.0,
            "secondary_plot": "Historical / Military",
            "display_tag": "(Kingdom Building / Military)",
        }
    
    return {
        "scenario": "NONE",
        "cultivation_penalty": 0.0,
        "historical_boost": 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pipeline/test_taxonomy.py`
Expected: PASS (`3 passed`)

- [ ] **Step 5: Commit Task 1**

```bash
git add kisholens/pipeline/taxonomy.py tests/pipeline/test_taxonomy.py
git commit -m "feat(taxonomy): add taxonomy lexicons and classical epic guardrail evaluator"
```

---

### Task 2: Intercept Scoring & Format Display Payload (`kisholens/ml/analyzer.py`)

**Files:**
- Modify: `kisholens/ml/analyzer.py:45-115`
- Test: `tests/ml/test_analyzer.py`

**Interfaces:**
- Consumes: `evaluate_epic_cultivation_guardrail(full_scan_text)` from `kisholens/pipeline/taxonomy.py`
- Produces: Disambiguated `analyze_prose()` payload containing scenario-adjusted scores and display labels.

- [ ] **Step 1: Write failing unit test for classical epic guardrail in `test_analyzer.py`**

Update `tests/ml/test_analyzer.py`:
```python
from kisholens.ml.analyzer import analyze_prose

def test_analyze_prose_classical_epic_scenario_a():
    ch1 = "The emperor's general marched the army and cavalry to suppress the rebellion at the imperial court."
    ch10 = "The warlord gathered troops and siege engines to claim the mandate of heaven."
    ch20 = "The strategy of the imperial general defeated the warlord's army."
    
    res = analyze_prose(None, ch1, ch10, ch20, title="Romance of the Three Kingdoms")
    assert res["world_setting"]["primary"] in ["Historical", "Action / Adventure"]
    assert "(Military Epic)" in res["display_label"]

def test_analyze_prose_hybrid_cultivation_scenario_b():
    ch1 = "The emperor's general gathered qi in his dantian to breakthrough foundation establishment during the siege."
    ch10 = "The warlord's nascent soul army used spirit stones to march on the imperial court."
    ch20 = "With his meridian unblocked, the general led the troops to victory."
    
    res = analyze_prose(None, ch1, ch10, ch20, title="Kingdom Building Cultivator")
    assert res["narrative_plot"]["primary"] == "Historical / Military"
    assert "(Kingdom Building / Military)" in res["display_label"]
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/ml/test_analyzer.py`
Expected: FAIL with `AssertionError`

- [ ] **Step 3: Modify `kisholens/ml/analyzer.py` to integrate guardrail**

In `kisholens/ml/analyzer.py`:
```python
from kisholens.pipeline.taxonomy import evaluate_epic_cultivation_guardrail

# Inside analyze_prose():
    full_scan_text = f"{title or ''} {synopsis or ''} {ch1_text} {ch10_text or ''} {ch20_text or ''}".strip()
    anchor_boosts = scan_anchor_boosts(full_scan_text)

    # Base sustained score calculation
    sustained_scores: Dict[str, float] = {}
    for i, gname in enumerate(genres):
        raw = float(sustained_sims[i])
        boosted = raw + anchor_boosts.get(gname, 0.0)
        sustained_scores[gname] = round(_calibrate(boosted), 4)

    # Evaluate Epic vs. Cultivation Guardrail
    guardrail = evaluate_epic_cultivation_guardrail(full_scan_text)

    if guardrail["scenario"] == "A":
        # Pure Classical Epic: Apply -0.40 Cultivation penalty and +0.20 Historical boost
        sustained_scores["Cultivation"] = round(max(0.01, sustained_scores.get("Cultivation", 0.50) + guardrail["cultivation_penalty"]), 4)
        sustained_scores["Historical"] = round(min(0.99, sustained_scores.get("Historical", 0.50) + guardrail["historical_boost"]), 4)

    # Sort sustained scores
    sorted_sustained = sorted(sustained_scores.items(), key=lambda x: x[1], reverse=True)
    world_gname, world_score = sorted_sustained[0]
    plot_gname, plot_score = sorted_sustained[1] if len(sorted_sustained) > 1 else (world_gname, world_score)

    if guardrail["scenario"] == "B":
        plot_gname = "Historical / Military"

    display_parts = []
    if inciting_payload and guardrail["scenario"] != "A":
        display_parts.append(inciting_payload["primary"])
    display_parts.append(world_gname)
    if guardrail["scenario"] == "A":
        display_parts.append("Action / Adventure")
        display_parts.append("(Military Epic)")
    elif guardrail["scenario"] == "B":
        display_parts.append("(Kingdom Building / Military)")
    else:
        display_parts.append(f"({plot_gname})")
    
    display_label = " ".join(display_parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_analyzer.py`
Expected: PASS

- [ ] **Step 5: Commit Task 2**

```bash
git add kisholens/ml/analyzer.py tests/ml/test_analyzer.py
git commit -m "feat(ml): integrate classical epic vs cultivation guardrail into prose analyzer"
```

---

### Task 3: Backward-Compatible Re-exports (`kisholens/ml/semantic_match.py`)

**Files:**
- Modify: `kisholens/ml/semantic_match.py:55-170`
- Test: `tests/ml/test_semantic_match.py`

**Interfaces:**
- Consumes: `GENRE_TAXONOMY`, `scan_anchor_boosts` from `kisholens/pipeline/taxonomy.py`
- Produces: Re-exported taxonomy symbols for existing modules.

- [ ] **Step 1: Update `kisholens/ml/semantic_match.py` to re-export from `kisholens/pipeline/taxonomy.py`**

In `kisholens/ml/semantic_match.py`:
```python
from kisholens.pipeline.taxonomy import (
    GENRE_TAXONOMY,
    ANCHOR_TERMS,
    detect_text_language,
    scan_anchor_boosts,
)
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/`
Expected: PASS (`48+ passed`)

- [ ] **Step 3: Commit Task 3**

```bash
git add kisholens/ml/semantic_match.py
git commit -m "refactor(ml): re-export taxonomy definitions from pipeline/taxonomy.py"
```

---

### Task 4: End-to-End Verification & Database Cache Update

**Files:**
- Modify: `scratch/update_stats_cache.py`
- Run: `uv run pytest tests/`

- [ ] **Step 1: Run complete PyTest suite**

Run: `uv run pytest tests/`
Expected: ALL TESTS PASS cleanly with zero errors.

- [ ] **Step 2: Verify Romance of the Three Kingdoms empirical prediction**

Run:
```bash
.venv/bin/python -c "
from sqlmodel import Session, select
from kisholens.models import Novel, Chapter, get_engine
from kisholens.ml.analyzer import analyze_prose

engine = get_engine()
with Session(engine) as session:
    novel = session.get(Novel, 231)
    chs = session.exec(select(Chapter).where(Chapter.novel_id == 231).order_by(Chapter.chapter_number)).all()
    ch1 = chs[0].text_zh or chs[0].text_en or ''
    ch10 = chs[len(chs)//2].text_zh or chs[len(chs)//2].text_en or ''
    ch20 = chs[-1].text_zh or chs[-1].text_en or ''
    res = analyze_prose(None, ch1, ch10, ch20, title=novel.title)
    print('Display Label:', res['display_label'])
    print('World Setting:', res['world_setting'])
"
```
Expected: `Display Label: Historical Action / Adventure (Military Epic)`

- [ ] **Step 3: Commit final plan execution**

```bash
git add scratch/update_stats_cache.py
git commit -m "chore: update stats cache script for classical epic guardrail"
```
