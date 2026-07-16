# Code Refinement & Deduplication (Approach 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deduplicate database ingestion, share sentiment keyword lists, extract diversity variance calculations, and streamline pipeline preview features in KishoLens.

**Architecture:** Modify `kisholens/ml/features.py`, `kisholens/api/main.py`, and `kisholens/pipeline/main.py` to share assets and leverage reusable functions.

**Tech Stack:** Python 3.10+, FastAPI, SQLModel.

## Global Constraints
* Keep current dependencies (aiohttp, lxml) unchanged.
* Do not make unilateral changes to unrelated code.
* Ensure all existing tests and imports are verified correct.

---

### Task 1: Unify math and sentiment constants in `kisholens/ml/features.py`

**Files:**
- Modify: `kisholens/ml/features.py`

**Interfaces:**
- Produces:
  - `EN_POS_WORDS`, `EN_NEG_WORDS`, `JA_POS_WORDS`, `JA_NEG_WORDS`, `ZH_POS_WORDS`, `ZH_NEG_WORDS` (lists of strings)
  - `compute_narrative_feature_diversity(vals: List[float]) -> float`

- [ ] **Step 1: Define shared constants and diversity math helper**
  Add constants and helper function near the top of `kisholens/ml/features.py` (after dynamic library flags):
  ```python
  EN_POS_WORDS = ["good", "great", "joy", "happy", "love", "hope", "bright", "beautiful", "triumph", "warm"]
  EN_NEG_WORDS = ["bad", "dark", "grief", "hate", "fear", "pain", "cold", "loss", "death", "despair"]

  JA_POS_WORDS = ["嬉しい", "楽しい", "美しい", "素晴らしい", "愛する", "成功", "幸せ", "感謝", "満足"]
  JA_NEG_WORDS = ["悲しい", "苦しい", "怒る", "嫌い", "失敗", "痛い", "最悪", "残念", "孤独"]

  ZH_POS_WORDS = ["高兴", "开心", "美丽", "棒", "爱", "成功", "幸福", "感谢", "满意", "喜欢"]
  ZH_NEG_WORDS = ["悲伤", "痛苦", "生气", "讨厌", "失败", "疼", "差", "可惜", "孤独", "难过"]

  def compute_narrative_feature_diversity(vals: List[float]) -> float:
      """Computes narrative feature diversity from a list of metrics using their variance."""
      if not vals:
          return 1.0
      mean_val = sum(vals) / len(vals)
      variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
      return float(1.0 / (1.0 + variance))
  ```

- [ ] **Step 2: Replace inline diversity math in features.py**
  Replace duplicate variance checks in `extract_english_features` (around L237), `extract_japanese_features` (around L350), and `extract_chinese_features` (around L493) with `compute_narrative_feature_diversity`.
  For English:
  ```python
  narrative_feature_diversity = compute_narrative_feature_diversity(vals)
  ```
  For Japanese:
  ```python
  narrative_feature_diversity = compute_narrative_feature_diversity(vals)
  ```
  For Chinese:
  ```python
  narrative_feature_diversity = compute_narrative_feature_diversity(vals)
  ```

- [ ] **Step 3: Run sanity check on ML analyzer**
  Verify compilation and execution by running the ML analysis pipeline:
  Run: `uv run python -m kisholens.ml.main`
  Expected: Success without exceptions; outputs archetype analysis.

- [ ] **Step 4: Commit**
  ```bash
  git add kisholens/ml/features.py
  git commit -m "refactor(ml): share sentiment constants and diversity helper"
  ```

---

### Task 2: Clean up VADER and sentiment keyword lists in `kisholens/api/main.py`

**Files:**
- Modify: `kisholens/api/main.py`

**Interfaces:**
- Consumes:
  - `EN_POS_WORDS`, `EN_NEG_WORDS`, `JA_POS_WORDS`, `JA_NEG_WORDS`, `ZH_POS_WORDS`, `ZH_NEG_WORDS` from `kisholens.ml.features`
  - `_init_nlp_resources` from `kisholens.ml.features`

- [ ] **Step 1: Import constants and _init_nlp_resources**
  Modify imports at the top of `kisholens/api/main.py` to pull:
  ```python
  from kisholens.ml.features import (
      JA_POS_WORDS, JA_NEG_WORDS,
      ZH_POS_WORDS, ZH_NEG_WORDS,
      EN_POS_WORDS, EN_NEG_WORDS,
      _init_nlp_resources
  )
  ```

- [ ] **Step 2: Simplify VADER loader in get_novel_arc**
  Replace L294-304 with a call to `_init_nlp_resources()`:
  ```python
          sia = None
          if lang == "en":
              try:
                  from nltk.sentiment.vader import SentimentIntensityAnalyzer
                  _init_nlp_resources()
                  sia = SentimentIntensityAnalyzer()
              except Exception:
                  pass
  ```

- [ ] **Step 3: Reuse constants in score_sentence**
  Replace duplicate sentiment lists in `score_sentence` (L306-330):
  ```python
          def score_sentence(s: str) -> float:
              if lang == "en":
                  if sia:
                      return sia.polarity_scores(s)["compound"]
                  pos = len(re.findall(
                      r'\b(' + '|'.join(EN_POS_WORDS) + r')\b',
                      s.lower()
                  ))
                  neg = len(re.findall(
                      r'\b(' + '|'.join(EN_NEG_WORDS) + r')\b',
                      s.lower()
                  ))
                  return (pos - neg) / (pos + neg + 1)
              elif lang == "ja":
                  pos = sum(s.count(w) for w in JA_POS_WORDS)
                  neg = sum(s.count(w) for w in JA_NEG_WORDS)
                  return (pos - neg) / (pos + neg + 1)
              else:  # zh
                  pos = sum(s.count(w) for w in ZH_POS_WORDS)
                  neg = sum(s.count(w) for w in ZH_NEG_WORDS)
                  return (pos - neg) / (pos + neg + 1)
  ```

- [ ] **Step 4: Verify API compilation**
  Run: `uv run python -c "from kisholens.api.main import app; print('API compiled OK')"`
  Expected: "API compiled OK" output.

- [ ] **Step 5: Commit**
  ```bash
  git add kisholens/api/main.py
  git commit -m "refactor(api): reuse shared sentiment constants and VADER init"
  ```

---

### Task 3: Unify Novel lookup/creation and features extraction in `kisholens/pipeline/main.py`

**Files:**
- Modify: `kisholens/pipeline/main.py`

**Interfaces:**
- Consumes:
  - `extract_english_features`, `extract_japanese_features`, `extract_chinese_features` from `kisholens.ml.features`

- [ ] **Step 1: Define _get_or_create_novel helper**
  Add the following helper function at module scope in `kisholens/pipeline/main.py`:
  ```python
  def _get_or_create_novel(session: Session, title: str, author: str, source: str, cache: dict) -> int:
      novel_key = (title, author)
      if novel_key not in cache:
          statement = select(Novel).where(Novel.title == title, Novel.author == author)
          existing_novel = session.exec(statement).first()
          if existing_novel:
              cache[novel_key] = existing_novel.id
          else:
              novel = Novel(title=title, author=author, source=source)
              session.add(novel)
              session.commit()
              session.refresh(novel)
              cache[novel_key] = novel.id
              print(f"Added Novel: '{title}' by {author} (ID: {novel.id})")
      return cache[novel_key]
  ```

- [ ] **Step 2: Replace inline lookup blocks**
  In `run_etl` (around L59-71 and L133-146), replace the duplicate Novel lookup and insert logic blocks with:
  ```python
  novel_id = _get_or_create_novel(session, series_title, author, source, novels_cache)
  ```

- [ ] **Step 3: Simplify extract_features**
  Import feature extractors from `kisholens.ml.features` and rewrite `extract_features` (L10-38) to use them:
  ```python
  from kisholens.ml.features import (
      extract_english_features,
      extract_japanese_features,
      extract_chinese_features
  )

  def extract_features(text: str, lang: str = "en"):
      """Computes baseline preview features by delegating to the unified extractors in ml.features."""
      if lang == "en":
          f = extract_english_features(text)
          return {
              "token_count": f.get("word_count", 0),
              "sentence_count": f.get("sentence_count", 0),
              "punctuation_density": f.get("punc_density", 0.0),
              "dialogue_ratio": f.get("dialogue_ratio", 0.0)
          }
      elif lang == "ja":
          f = extract_japanese_features(text)
          return {
              "token_count": f.get("char_count", 0),
              "sentence_count": f.get("sentence_count", 0),
              "punctuation_density": f.get("punc_density", 0.0),
              "dialogue_ratio": f.get("dialogue_ratio", 0.0)
          }
      else:  # zh
          f = extract_chinese_features(text)
          return {
              "token_count": f.get("char_count", 0),
              "sentence_count": f.get("sentence_count", 0),
              "punctuation_density": f.get("punc_density", 0.0),
              "dialogue_ratio": f.get("dialogue_ratio", 0.0)
          }
  ```

- [ ] **Step 4: Run ETL Pipeline check**
  Run: `uv run python -m kisholens.pipeline.main`
  Expected: Ingests 5 chapters of royalroad, scribblehub, syosetu, gutenberg, cnnovel; prints feature logs successfully.

- [ ] **Step 5: Commit**
  ```bash
  git add kisholens/pipeline/main.py
  git commit -m "refactor(pipeline): extract novel helper and reuse ml feature extractors"
  ```
