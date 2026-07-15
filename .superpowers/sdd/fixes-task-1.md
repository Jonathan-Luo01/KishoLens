# Fixes Report: Backend Baseline Data Endpoint Integration (Task 1)

Resolved issues in the backend baseline data endpoint (`kisholens/api/main.py`).

## Summary of Changes

1. **Severe Performance Bottleneck (Scale/DB Load & NLP) & DB Query Optimization**
   - Optimized the scale check to avoid fetching all chapters from the database before checking the count.
   - Now, a rapid SQL COUNT query is run first:
     ```python
     count = session.exec(select(func.count(Chapter.id))).one()
     if count > 30:
         _cached_baselines[lang] = fallbacks
         return _cached_baselines[lang]
     ```
   - Only when the count is 30 or less do we fetch all chapters from the database:
     ```python
     chapters = session.exec(select(Chapter)).all()
     ```
   - This prevents loading all chapters into memory when there are many chapters, resolving the high database load/scale bottleneck.

2. **Language-Specific Baseline Segregation**
   - Updated `get_baseline_stats(lang)` to accept the target language string `lang` (defaulting to `"en"`).
   - Filtered and extracted features specifically matching that language (checking `ch.text_en` for English, `ch.text_ja` for Japanese, and `ch.text_zh` for Chinese).
   - Defined language-specific fallback baseline constants for `en`, `ja`, and `zh` based on:
     - `en`: Gutenberg (TTR=33.9%, Dialogue=34.9%, AvgSentLen=11.4); Webnovel (TTR=36.1%, Dialogue=22.3%, AvgSentLen=10.8)
     - `ja`: Gutenberg (TTR=22.0%, Dialogue=15.0%, AvgSentLen=25.0); Webnovel (TTR=28.8%, Dialogue=40.2%, AvgSentLen=35.7)
     - `zh`: Gutenberg (TTR=0.7%, Dialogue=1.5%, AvgSentLen=13.5); Webnovel (TTR=3.1%, Dialogue=29.4%, AvgSentLen=22.1)

3. **Cache Override Bug Fix**
   - Updated `_cached_baselines` from `None` to a dictionary mapping language codes directly (e.g. `_cached_baselines = {}`).
   - Integrated logic to resolve each language's fallback baselines independently. In the case where one category (e.g., Gutenberg or Webnovel) is missing or empty in the database, the fallback metrics for that specific category are loaded individually without discarding or overwriting the calculated metrics of the other category.

## Impact Analysis & verification
- **Impact Analysis**: Executed GitNexus impact analysis on `get_baseline_stats`. The only direct caller detected is `post_analyze` with a LOW risk level.
- **Verification**: Ran `DISABLE_HANLP=1 uv run python scratch/test_api_baselines.py` and confirmed:
  `ALL BASELINE API TESTS PASSED!`
- **Detect Changes**: Ran `node .gitnexus/run.cjs detect_changes` to verify affected files/symbols.
