# Spec: Code Refinement & Deduplication (Approach 2)

## Overview
This specification details the code-level refactoring and deduplication plan for the KishoLens codebase. Based on user selection, we are pursuing **Approach 2**, focusing on deduplicating source code logic, sharing assets, and aligning functional correctness while leaving third-party dependency requirements (`aiohttp` and `lxml`) intact.

---

## 1. Objectives & Requirements
* **Unify Novel Ingestion paths**: Extract the duplicate Novel lookup and SQLModel insert blocks in `pipeline/main.py` into a helper function.
* **Reuse Sentiment Word Lists**: Unify the Japanese, Chinese, and English sentiment keyword lists, defining them once in `ml/features.py` and referencing them in `api/main.py`.
* **Deduplicate Diversity Math**: Extract the narrative feature diversity variance math formula used across English, Japanese, and Chinese feature extractions in `ml/features.py` into a helper function.
* **Clean up VADER Checks**: Clean up the redundant inline VADER download block in the sentiment arc calculation inside `api/main.py` by calling `_init_nlp_resources` from `ml/features.py` instead.
* **Streamline Preview Features**: Clean up the regex-based `extract_features` function in `pipeline/main.py` by importing and calling the unified extractors in `ml/features.py`.

---

## 2. Design Details

### 2.1 Novel Lookup Helper in `pipeline/main.py`
We will introduce a helper `_get_or_create_novel`:
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
And replace the inline blocks in `run_etl` with this function call.

### 2.2 Shared Sentiment Lists & VADER Init
In `ml/features.py`, we will define module-level constants:
```python
EN_POS_WORDS = ["good", "great", "joy", "happy", "love", "hope", "bright", "beautiful", "triumph", "warm"]
EN_NEG_WORDS = ["bad", "dark", "grief", "hate", "fear", "pain", "cold", "loss", "death", "despair"]

JA_POS_WORDS = ["嬉しい", "楽しい", "美しい", "素晴らしい", "愛する", "成功", "幸せ", "感謝", "満足"]
JA_NEG_WORDS = ["悲しい", "苦しい", "怒る", "嫌い", "失敗", "痛い", "最悪", "残念", "孤独"]

ZH_POS_WORDS = ["高兴", "开心", "美丽", "棒", "爱", "成功", "幸福", "感谢", "满意", "喜欢"]
ZH_NEG_WORDS = ["悲伤", "痛苦", "生气", "讨厌", "失败", "疼", "差", "可惜", "孤独", "难过"]
```
These will be imported in `api/main.py` and reused in `score_sentence` inside `get_novel_arc`.
We will also clean up the inline VADER downloader in `api/main.py`:
```python
        sia = None
        if lang == "en":
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                from kisholens.ml.features import _init_nlp_resources
                _init_nlp_resources()
                sia = SentimentIntensityAnalyzer()
            except Exception:
                pass
```

### 2.3 Diversity Math Helper
In `ml/features.py`, we will add `compute_narrative_feature_diversity`:
```python
def compute_narrative_feature_diversity(vals: List[float]) -> float:
    """Computes narrative feature diversity from a list of metrics using their variance."""
    if not vals:
        return 1.0
    mean_val = sum(vals) / len(vals)
    variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
    return float(1.0 / (1.0 + variance))
```
This helper will replace the duplicate calculations in `extract_english_features`, `extract_japanese_features`, and `extract_chinese_features`.

---

## 3. Implementation Plan
1. Define shared constants and the math helper in `kisholens/ml/features.py`.
2. Clean up `kisholens/api/main.py` to import and use the shared constants and VADER initialization.
3. Clean up `kisholens/pipeline/main.py` to define the Novel helper and call it, and call the unified features extractors.
4. Verify the changes by running backend compilation and pipeline ingestion.
