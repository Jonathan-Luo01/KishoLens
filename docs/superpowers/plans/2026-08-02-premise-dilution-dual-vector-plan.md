# Dual-Vector + Semantic Concept Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor KishoLens classification engine with a Dual-Vector + Pure Concept Vector Architecture to solve Premise Dilution.

**Architecture:** Create modular ML units in `kisholens/ml/`: `embeddings.py` generates $V_{\text{intro}}$ and $V_{\text{sustained}}$ using dynamic weight redistribution; `centroids.py` holds pure concept vectors ($V_{\text{concept\_<key>}}$); `analyzer.py` calculates independent scores, applies dynamic concept density multipliers, enforces a 0.55 confidence threshold fallback, and returns structured taxonomy results. `semantic_match.py` acts as a backward-compatible adapter.

**Tech Stack:** Python 3.10+, PyTorch, `sentence-transformers` (`all-MiniLM-L6-v2`), NumPy, PyTest.

## Global Constraints
- Target Files: `kisholens/ml/embeddings.py`, `kisholens/ml/centroids.py`, `kisholens/ml/analyzer.py`, `kisholens/ml/semantic_match.py`.
- Sentence Transformer Model: `all-MiniLM-L6-v2` (384-dimensional float32 embeddings).
- Thresholds: Concept density boost threshold = 0.20; Inciting Event confidence fallback threshold = 0.55.
- Zero place-holders: Complete code in every step and TDD with PyTest.

---

### Task 1: Dual-Scope Vector Generator (`kisholens/ml/embeddings.py`)

**Files:**
- Create: `kisholens/ml/embeddings.py`
- Test: `tests/ml/test_embeddings.py`

**Interfaces:**
- Produces: `generate_dual_vectors(synopsis: Optional[str], ch1_text: str, ch10_text: Optional[str] = None, ch20_text: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]`
- Produces: `embed_single_text(text: str) -> np.ndarray`

- [ ] **Step 1: Write the failing unit tests for dual-scope vector generation**

```python
# tests/ml/test_embeddings.py
import numpy as np
import pytest
from kisholens.ml.embeddings import embed_single_text, generate_dual_vectors

def test_embed_single_text():
    text = "The protagonist woke up in an unfamiliar fantasy world."
    vec = embed_single_text(text)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    # Verify L2 norm is 1.0
    assert pytest.approx(np.linalg.norm(vec), abs=1e-3) == 1.0

def test_generate_dual_vectors_with_synopsis():
    synopsis = "A high school student dies in a truck accident and wakes up as an imperial prince."
    ch1 = "I opened my eyes to a grand canopy bed inside the Thirteenth Prince Palace."
    ch10 = "I was swinging my sword in the yard and having tea with the court knight."
    ch20 = "The demon sword awakened as the palace guards watched in awe."

    v_intro, v_sustained = generate_dual_vectors(synopsis, ch1, ch10, ch20)
    assert v_intro.shape == (384,)
    assert v_sustained.shape == (384,)
    assert pytest.approx(np.linalg.norm(v_intro), abs=1e-3) == 1.0
    assert pytest.approx(np.linalg.norm(v_sustained), abs=1e-3) == 1.0

def test_generate_dual_vectors_without_synopsis():
    ch1 = "I opened my eyes to a grand canopy bed inside the Thirteenth Prince Palace."
    ch10 = "I was swinging my sword in the yard."
    ch20 = "The demon sword awakened."

    v_intro, v_sustained = generate_dual_vectors(None, ch1, ch10, ch20)
    assert v_intro.shape == (384,)
    assert v_sustained.shape == (384,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_embeddings.py -v`
Expected: FAIL with ModuleNotFoundError or function missing.

- [ ] **Step 3: Implement dual-vector generation in `kisholens/ml/embeddings.py`**

```python
# kisholens/ml/embeddings.py
"""
embeddings.py — Dual-scope vector generation for KishoLens.

Computes normalized 384D MiniLM embeddings for Intro (V_intro) and Sustained (V_sustained) prose contexts.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

# Use lazy loading for sentence-transformers to avoid slow module import times
_model = None

def get_transformer_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0 or np.isnan(norm):
        return vec
    return (vec / norm).astype(np.float32)

def embed_single_text(text: str) -> np.ndarray:
    if not text or not text.strip():
        return np.zeros(384, dtype=np.float32)
    model = get_transformer_model()
    # SentenceTransformer encode returns ndarray
    vec = model.encode(text.strip(), convert_to_numpy=True)
    return _normalize(vec.astype(np.float32))

def generate_dual_vectors(
    synopsis: Optional[str],
    ch1_text: str,
    ch10_text: Optional[str] = None,
    ch20_text: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate V_intro and V_sustained using dynamic weight redistribution based on synopsis presence.
    """
    v_syn = embed_single_text(synopsis) if synopsis and synopsis.strip() else None
    v_ch1 = embed_single_text(ch1_text) if ch1_text and ch1_text.strip() else np.zeros(384, dtype=np.float32)
    v_ch10 = embed_single_text(ch20_text) if ch10_text and ch10_text.strip() else v_ch1
    v_ch20 = embed_single_text(ch20_text) if ch20_text and ch20_text.strip() else v_ch10

    if v_syn is not None:
        # Scenario A: Synopsis is Present
        v_intro = 0.60 * v_syn + 0.40 * v_ch1
        v_sustained = 0.10 * v_syn + 0.10 * v_ch1 + 0.40 * v_ch10 + 0.40 * v_ch20
    else:
        # Scenario B: Synopsis is Missing
        v_intro = 1.0 * v_ch1
        v_sustained = 0.20 * v_ch1 + 0.40 * v_ch10 + 0.40 * v_ch20

    return _normalize(v_intro), _normalize(v_sustained)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_embeddings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kisholens/ml/embeddings.py tests/ml/test_embeddings.py
git commit -m "feat(ml): implement dual-scope vector generation in embeddings.py"
```

---

### Task 2: Pure Concept Vectors & Centroid Manager (`kisholens/ml/centroids.py`)

**Files:**
- Create: `kisholens/ml/centroids.py`
- Test: `tests/ml/test_centroids.py`

**Interfaces:**
- Consumes: `embed_single_text` from `kisholens/ml/embeddings.py`
- Produces: `INCITING_CONCEPTS: dict[str, str]`
- Produces: `get_concept_vector(concept_name: str) -> np.ndarray`
- Produces: `get_inciting_concept_vectors() -> dict[str, np.ndarray]`

- [ ] **Step 1: Write failing unit test for pure concept vectors**

```python
# tests/ml/test_centroids.py
import numpy as np
import pytest
from kisholens.ml.centroids import (
    INCITING_CONCEPTS,
    get_concept_vector,
    get_inciting_concept_vectors,
)

def test_inciting_concepts_dict():
    assert "Isekai & Regression" in INCITING_CONCEPTS
    assert "System Initialization" in INCITING_CONCEPTS
    assert "Cultivation Awakening" in INCITING_CONCEPTS

def test_get_concept_vector():
    vec = get_concept_vector("Isekai & Regression")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert pytest.approx(np.linalg.norm(vec), abs=1e-3) == 1.0

def test_get_inciting_concept_vectors():
    vecs = get_inciting_concept_vectors()
    assert len(vecs) == 3
    assert "Isekai & Regression" in vecs
    assert vecs["Isekai & Regression"].shape == (384,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_centroids.py -v`
Expected: FAIL with ModuleNotFoundError or function missing.

- [ ] **Step 3: Implement Pure Concept Vectors in `kisholens/ml/centroids.py`**

```python
# kisholens/ml/centroids.py
"""
centroids.py — Pure Concept Vectors and Book Centroid manager.
"""

from __future__ import annotations
from typing import Dict
import numpy as np
from kisholens.ml.embeddings import embed_single_text

INCITING_CONCEPTS: Dict[str, str] = {
    "Isekai & Regression": (
        "The protagonist dies and is reincarnated, opens their eyes and finds themselves in a fantasy/game/other world, "
        "transmigrated into a novel or game as a villainess or mob character, summoned to another world as a hero, "
        "or regresses back in time to their past life for a second chance at changing their fate."
    ),
    "System Initialization": (
        "A mysterious system interface suddenly appears before the protagonist's eyes, granting them a status window, "
        "levels, skills, and quests. The world undergoes an apocalyptic evolution or shifts into a game-like reality "
        "with dungeons and monsters."
    ),
    "Cultivation Awakening": (
        "The protagonist discovers a heaven-defying cheat artifact, awakens a supreme spiritual root, "
        "or repairs their crippled meridians to begin their journey on the path of cultivation, martial arts, and immortality."
    ),
}

_concept_vector_cache: Dict[str, np.ndarray] = {}

def get_concept_vector(concept_name: str) -> np.ndarray:
    """Return normalized 384D concept embedding for a given concept name."""
    if concept_name not in _concept_vector_cache:
        text = INCITING_CONCEPTS.get(concept_name, "")
        _concept_vector_cache[concept_name] = embed_single_text(text)
    return _concept_vector_cache[concept_name]

def get_inciting_concept_vectors() -> Dict[str, np.ndarray]:
    """Return dictionary of all pre-computed concept vectors."""
    for name in INCITING_CONCEPTS:
        get_concept_vector(name)
    return _concept_vector_cache
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_centroids.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kisholens/ml/centroids.py tests/ml/test_centroids.py
git commit -m "feat(ml): implement pure concept vector definitions in centroids.py"
```

---

### Task 3: Independent Prose Analyzer (`kisholens/ml/analyzer.py`)

**Files:**
- Create: `kisholens/ml/analyzer.py`
- Test: `tests/ml/test_analyzer.py`

**Interfaces:**
- Consumes: `generate_dual_vectors` from `kisholens/ml/embeddings.py`
- Consumes: `get_inciting_concept_vectors` from `kisholens/ml/centroids.py`
- Consumes: centroids from `kisholens/ml/build_centroids.py`
- Produces: `analyze_prose(synopsis: Optional[str], ch1_text: str, ch10_text: Optional[str] = None, ch20_text: Optional[str] = None, title: Optional[str] = None) -> dict`

- [ ] **Step 1: Write failing unit test for independent prose analyzer**

```python
# tests/ml/test_analyzer.py
import pytest
from kisholens.ml.analyzer import analyze_prose

def test_analyze_prose_isekai_novel():
    synopsis = "Reincarnated into another world as the 13th Imperial Prince with divine stats."
    ch1 = "I opened my eyes to a grand canopy bed inside the Thirteenth Prince Palace."
    ch10 = "I held the demon sword and relaxed in the imperial garden."
    ch20 = "The demon sword unleashed its power."

    res = analyze_prose(synopsis, ch1, ch10, ch20, title="Noble Reincarnation")
    assert "inciting_event" in res
    assert "world_setting" in res
    assert "narrative_plot" in res
    assert "display_label" in res

    # Verify inciting event is Isekai & Regression with high score
    inciting = res["inciting_event"]
    assert inciting is not None
    assert inciting["primary"] == "Isekai & Regression"
    assert inciting["score"] >= 0.70

def test_analyze_prose_fallback_threshold():
    # Non-inciting text (pure Victorian domestic conversation with no setup event)
    ch1 = "The tea was served cold in the parlor as Mr. Bennett discussed the evening news."
    ch10 = "Lady Catherine walked through the garden complaining about the weather."
    ch20 = "They sat quietly by the fireplace in the drawing room."

    res = analyze_prose(None, ch1, ch10, ch20, title="Quiet Tea Room")
    # Should fall back to None when inciting score < 0.55
    assert res["inciting_event"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_analyzer.py -v`
Expected: FAIL with ModuleNotFoundError or function missing.

- [ ] **Step 3: Implement `analyze_prose` in `kisholens/ml/analyzer.py`**

```python
# kisholens/ml/analyzer.py
"""
analyzer.py — Dual-Vector + Dynamic Semantic Concept Prose Analyzer for KishoLens.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np

from kisholens.ml.embeddings import generate_dual_vectors
from kisholens.ml.centroids import get_inciting_concept_vectors
from kisholens.ml.semantic_match import _load_with_cache, DEFAULT_DATA_DIR, scan_anchor_boosts

def analyze_prose(
    synopsis: Optional[str],
    ch1_text: str,
    ch10_text: Optional[str] = None,
    ch20_text: Optional[str] = None,
    title: Optional[str] = None,
    data_dir: str = DEFAULT_DATA_DIR,
) -> Dict[str, Any]:
    """
    Perform independent dual-vector classification across Inciting Events, World Settings, and Narrative Plots.
    """
    v_intro, v_sustained = generate_dual_vectors(synopsis, ch1_text, ch10_text, ch20_text)
    g_centroids, g_meta, t_centroids, t_meta = _load_with_cache(data_dir)

    if g_centroids is None or g_meta is None:
        return {}

    genres = g_meta["genres"]
    g_norms = np.linalg.norm(g_centroids, axis=1, keepdims=True)
    g_safe_norms = np.where(g_norms == 0, 1.0, g_norms)
    norm_g_centroids = g_centroids / g_safe_norms

    # Mean centering to remove cross-genre baseline noise
    g_mean = norm_g_centroids.mean(axis=0, keepdims=True)
    g_sub = norm_g_centroids - g_mean

    # Sustained similarities for World & Plot states
    sustained_sims = np.dot(v_sustained, g_sub.T)
    sustained_sims = sustained_sims - np.mean(sustained_sims)

    # Intro similarities for Base Inciting Event scores
    intro_sims = np.dot(v_intro, g_sub.T)
    intro_sims = intro_sims - np.mean(intro_sims)

    # Scan title/synopsis/ch1 text for macro boosts
    full_scan_text = f"{title or ''} {synopsis or ''} {ch1_text}".strip()
    anchor_boosts = scan_anchor_boosts(full_scan_text)

    def _calibrate(s: float, k: float = 5.5) -> float:
        return float(1.0 / (1.0 + np.exp(-k * s)))

    # Compute genre scores for Sustained states
    sustained_scores: Dict[str, float] = {}
    for i, gname in enumerate(genres):
        raw = float(sustained_sims[i])
        boosted = raw + anchor_boosts.get(gname, 0.0)
        sustained_scores[gname] = round(_calibrate(boosted), 4)

    # 1. Inciting Event Evaluation
    concept_vecs = get_inciting_concept_vectors()
    inciting_results = []

    for concept_name, concept_vec in concept_vecs.items():
        # Map concept to closest book centroid (e.g. Isekai, Progression Fantasy, Cultivation)
        if "Isekai" in concept_name:
            target_g = "Isekai"
        elif "System" in concept_name:
            target_g = "Progression Fantasy"
        else:
            target_g = "Cultivation"

        idx = genres.index(target_g) if target_g in genres else 0
        s_base = _calibrate(float(intro_sims[idx]) + anchor_boosts.get(target_g, 0.0))
        
        # Concept Density Score: Cosine similarity between V_intro and V_concept
        s_concept = float(np.dot(v_intro, concept_vec))

        if s_concept > 0.20:
            dynamic_boost = min(0.25, s_concept * 0.50)
            final_score = round(min(0.99, s_base + dynamic_boost), 4)
        else:
            final_score = round(s_base, 4)

        inciting_results.append((concept_name, final_score))

    inciting_results.sort(key=lambda x: x[1], reverse=True)
    best_inciting_name, best_inciting_score = inciting_results[0]

    # Graceful Fallback Threshold check (< 0.55)
    if best_inciting_score < 0.55:
        inciting_payload = None
    else:
        inciting_payload = {"primary": best_inciting_name, "score": best_inciting_score}

    # 2. World Setting & Narrative Plot
    sorted_sustained = sorted(sustained_scores.items(), key=lambda x: x[1], reverse=True)
    world_gname, world_score = sorted_sustained[0]
    
    # Pick top narrative plot distinct from world setting
    plot_gname, plot_score = sorted_sustained[1] if len(sorted_sustained) > 1 else (world_gname, world_score)

    display_parts = []
    if inciting_payload:
        display_parts.append(inciting_payload["primary"])
    display_parts.append(world_gname)
    display_parts.append(f"({plot_gname})")
    display_label = " ".join(display_parts)

    return {
        "inciting_event": inciting_payload,
        "world_setting": {"primary": world_gname, "score": world_score},
        "narrative_plot": {"primary": plot_gname, "score": plot_score},
        "display_label": display_label,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ml/test_analyzer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kisholens/ml/analyzer.py tests/ml/test_analyzer.py
git commit -m "feat(ml): implement dual-vector prose analyzer in analyzer.py"
```

---

### Task 4: Backward-Compatible Adapter & API Integration (`kisholens/ml/semantic_match.py` & `kisholens/api/main.py`)

**Files:**
- Modify: `kisholens/ml/semantic_match.py`
- Modify: `kisholens/api/main.py`
- Test: `tests/ml/test_semantic_match.py`
- Test: `tests/ml/test_api_semantic.py`

**Interfaces:**
- Consumes: `analyze_prose` from `kisholens/ml/analyzer.py`

- [ ] **Step 1: Write failing unit test for adapter integration**

```python
# tests/ml/test_semantic_adapter.py
from kisholens.ml.semantic_match import match_semantic

def test_match_semantic_adapter():
    text = "Reincarnated into another world as the 13th Imperial Prince with divine stats."
    res = match_semantic(text, title="Noble Reincarnation")
    assert res is not None
    assert "genre" in res
    assert "genre_scores" in res
    assert "taxonomy" in res
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_semantic_adapter.py -v`
Expected: FAIL with "taxonomy" key missing or assertion error.

- [ ] **Step 3: Update `kisholens/ml/semantic_match.py` to adapt `analyze_prose`**

In `kisholens/ml/semantic_match.py`, update `match_semantic` to delegate to `analyze_prose` and attach the `"taxonomy"` field into the returned dict:

```python
# kisholens/ml/semantic_match.py
from kisholens.ml.analyzer import analyze_prose

def match_semantic(
    text: str,
    title: Optional[str] = None,
    synopsis: Optional[str] = None,
    data_dir: str = DEFAULT_DATA_DIR,
) -> Optional[dict]:
    # Extract positional chapter texts if formatted with double newlines
    paragraphs = text.split("\n\n")
    ch1 = paragraphs[0] if paragraphs else text
    ch10 = paragraphs[len(paragraphs) // 2] if len(paragraphs) > 2 else None
    ch20 = paragraphs[-1] if len(paragraphs) > 2 else None

    taxonomy = analyze_prose(synopsis, ch1, ch10, ch20, title=title, data_dir=data_dir)
    if not taxonomy:
        return None

    # Maintain backward compatibility with existing API schemas
    world_primary = taxonomy["world_setting"]["primary"]
    world_score = taxonomy["world_setting"]["score"]

    return {
        "genre": world_primary,
        "genre_confidence": world_score,
        "territory": "Web Novel Territory",
        "territory_confidence": 0.95,
        "genre_scores": [{"genre": world_primary, "score": world_score, "raw_score": world_score}],
        "territory_scores": [{"territory": "Web Novel Territory", "score": 0.95, "raw_score": 0.95}],
        "taxonomy": taxonomy,
    }
```

- [ ] **Step 4: Run all test suites to verify passing status**

Run: `uv run pytest tests/ -v`
Expected: PASS (All tests pass cleanly).

- [ ] **Step 5: Commit**

```bash
git add kisholens/ml/semantic_match.py tests/ml/test_semantic_adapter.py
git commit -m "refactor(ml): adapt match_semantic to dual-vector prose analyzer"
```
