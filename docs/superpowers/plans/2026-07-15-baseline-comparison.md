# Baseline Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate classic literature and web novel baseline comparative metrics into the `/api/analyze` API and the `/analyze` frontend dashboard.

**Architecture:** The backend computes average baseline metrics from the SQLite database (or falls back to pre-calculated baseline constants) and returns them in the `POST /api/analyze` response. The frontend reads this data and renders label indicators below the progress bars of the respective cards.

**Tech Stack:** FastAPI, SQLModel, Astro, CSS, Vanilla JS

## Global Constraints
- Do not commit mock or placeholder logic.
- Ensure all styling aligns with the existing dark theme/glassmorphism design.

---

### Task 1: Backend Baseline Data Endpoint Integration

**Files:**
- Modify: `kisholens/api/main.py`
- Create: `/Users/jonathan/.gemini/antigravity-cli/brain/44415173-3aac-40be-9614-3ddd98be293c/scratch/test_api_baselines.py`

**Interfaces:**
- Consumes: none
- Produces: `POST /api/analyze` response payload containing `baselines` data

- [ ] **Step 1: Write the failing API test script**
  Create the test script at `/Users/jonathan/.gemini/antigravity-cli/brain/44415173-3aac-40be-9614-3ddd98be293c/scratch/test_api_baselines.py`:
  ```python
  import sys
  from fastapi.testclient import TestClient
  
  sys.path.append("/Users/jonathan/Documents/KishoLens")
  from kisholens.api.main import app
  
  client = TestClient(app)
  
  def test_analyze_baselines_returned():
      response = client.post("/api/analyze", json={
          "text": "This is a simple sentence to test whether the baselines are returned by the API.",
          "lang": "en",
          "title": "Baseline Test"
      })
      assert response.status_code == 200
      data = response.json()
      assert "baselines" in data
      baselines = data["baselines"]
      assert "gutenberg" in baselines
      assert "webnovel" in baselines
      
      # Assert specific metrics are present
      for source in ("gutenberg", "webnovel"):
          metrics = baselines[source]
          assert "ttr" in metrics
          assert "dialogue_ratio" in metrics
          assert "avg_sentence_len" in metrics
          assert isinstance(metrics["ttr"], float)
          assert isinstance(metrics["dialogue_ratio"], float)
          assert isinstance(metrics["avg_sentence_len"], float)
  
  if __name__ == "__main__":
      try:
          test_analyze_baselines_returned()
          print("ALL BASELINE API TESTS PASSED!")
      except AssertionError as e:
          print("Baseline test verification failed:", e)
          sys.exit(1)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `uv run python /Users/jonathan/.gemini/antigravity-cli/brain/44415173-3aac-40be-9614-3ddd98be293c/scratch/test_api_baselines.py`
  Expected: FAIL with `AssertionError` (since `"baselines"` key is missing from response).

- [ ] **Step 3: Implement cached baseline logic and update endpoint in main.py**
  Add the module global `_cached_baselines`, the `get_baseline_stats()` helper, and update `post_analyze()` response payload in [main.py](file:///Users/jonathan/Documents/KishoLens/kisholens/api/main.py):
  
  ```python
  _cached_baselines = None
  
  def get_baseline_stats():
      global _cached_baselines
      if _cached_baselines is not None:
          return _cached_baselines
          
      try:
          with Session(engine) as session:
              chapters = session.exec(select(Chapter)).all()
              novels = session.exec(select(Novel)).all()
              novels_map = {n.id: n for n in novels}
              
              gutenberg_feats = []
              webnovel_feats = []
              
              for ch in chapters:
                  novel = novels_map.get(ch.novel_id)
                  if not novel:
                      continue
                  
                  if ch.text_en:
                      f = extract_english_features(ch.text_en)
                  elif ch.text_ja:
                      f = extract_japanese_features(ch.text_ja)
                  elif ch.text_zh:
                      f = extract_chinese_features(ch.text_zh)
                  else:
                      continue
                  
                  if novel.source == 'gutenberg':
                      gutenberg_feats.append(f)
                  else:
                      webnovel_feats.append(f)
              
              def avg_dict(lst):
                  if not lst:
                      return None
                  keys = lst[0].keys()
                  return {k: sum(d.get(k, 0) for d in lst) / len(lst) for k in keys}
                  
              _cached_baselines = {
                  "gutenberg": avg_dict(gutenberg_feats),
                  "webnovel": avg_dict(webnovel_feats)
              }
      except Exception as e:
          print(f"Error computing baselines from DB: {e}")
          
      # Fallback defaults in case DB query is empty/fails
      if not _cached_baselines or not _cached_baselines["gutenberg"] or not _cached_baselines["webnovel"]:
          _cached_baselines = {
              "gutenberg": {
                  "ttr": 0.173,
                  "dialogue_ratio": 0.182,
                  "avg_sentence_len": 12.464
              },
              "webnovel": {
                  "ttr": 0.306,
                  "dialogue_ratio": 0.235,
                  "avg_sentence_len": 12.699
              }
          }
      return _cached_baselines
  ```
  
  And update `@app.post("/api/analyze")` response payload:
  ```python
      baselines = get_baseline_stats()
      
      # format features for matcher
      agg = {f"{lang}_{k}": v for k, v in features.items()}
      archetype = match_archetype(agg)
      
      return {
          "status": "success",
          "detected_lang": lang,
          "features": features,
          "archetype": {
              "archetype": archetype["closest_trope"],
              "confidence": archetype["confidence"],
              "description": f"Classification: {archetype['territory']}. Closest matched writing archetype based on stylistic features."
          },
          "baselines": {
              "gutenberg": {
                  "ttr": baselines["gutenberg"].get("ttr", 0.173) if baselines["gutenberg"] else 0.173,
                  "dialogue_ratio": baselines["gutenberg"].get("dialogue_ratio", 0.182) if baselines["gutenberg"] else 0.182,
                  "avg_sentence_len": baselines["gutenberg"].get("avg_sentence_len", 12.464) if baselines["gutenberg"] else 12.464
              },
              "webnovel": {
                  "ttr": baselines["webnovel"].get("ttr", 0.306) if baselines["webnovel"] else 0.306,
                  "dialogue_ratio": baselines["webnovel"].get("dialogue_ratio", 0.235) if baselines["webnovel"] else 0.235,
                  "avg_sentence_len": baselines["webnovel"].get("avg_sentence_len", 12.699) if baselines["webnovel"] else 12.699
              }
          }
      }
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `DISABLE_HANLP=1 uv run python /Users/jonathan/.gemini/antigravity-cli/brain/44415173-3aac-40be-9614-3ddd98be293c/scratch/test_api_baselines.py`
  Expected: PASS with "ALL BASELINE API TESTS PASSED!"

- [ ] **Step 5: Stage API changes**
  Run: `git add kisholens/api/main.py`

---

### Task 2: Frontend `/analyze` Baseline UI Integration

**Files:**
- Modify: `frontend/src/pages/analyze.astro`

- [ ] **Step 1: Add comparative labels markup to metric cards**
  In [analyze.astro](file:///Users/jonathan/Documents/KishoLens/frontend/src/pages/analyze.astro):
  - In Vocabulary Richness card:
    ```html
                <div class="progress-bar-container">
                  <div class="progress-bar-fill" id="ttr-progress"></div>
                </div>
                <div class="baseline-comparison">
                  <span>Classic Lit: <strong id="ttr-gutenberg-val">0.0%</strong></span>
                  <span>Web Novel: <strong id="ttr-webnovel-val">0.0%</strong></span>
                </div>
    ```
  - In Dialogue Ratio card:
    ```html
                <div class="progress-bar-container">
                  <div class="progress-bar-fill" id="dialogue-progress"></div>
                </div>
                <div class="baseline-comparison">
                  <span>Classic Lit: <strong id="dialogue-gutenberg-val">0.0%</strong></span>
                  <span>Web Novel: <strong id="dialogue-webnovel-val">0.0%</strong></span>
                </div>
    ```
  - In Avg Sentence Length card:
    ```html
                <span class="metric-value" id="metric-sent-len">0.0 words</span>
                <div class="baseline-comparison" style="margin-top: 0.5rem; border-top: 1px solid rgba(255, 255, 255, 0.03); padding-top: 0.3rem;">
                  <span>Classic Lit: <strong id="sent-gutenberg-val">0.0w</strong></span>
                  <span>Web Novel: <strong id="sent-webnovel-val">0.0w</strong></span>
                </div>
    ```

- [ ] **Step 2: Add CSS styles for the comparison text**
  Add the `.baseline-comparison` styles in `<style>` block in [analyze.astro](file:///Users/jonathan/Documents/KishoLens/frontend/src/pages/analyze.astro):
  ```css
        .baseline-comparison {
          display: flex;
          justify-content: space-between;
          font-size: 0.72rem;
          color: var(--color-muted);
          margin-top: 0.4rem;
          border-top: 1px solid rgba(255, 255, 255, 0.03);
          padding-top: 0.3rem;
        }
        .baseline-comparison strong {
          color: var(--color-text);
          font-weight: 500;
        }
  ```

- [ ] **Step 3: Update renderResults function in analyze.astro**
  Update the script inside `renderResults(data)` to format and render baseline values:
  ```javascript
          const base = data.baselines;
          const unit = data.detected_lang === "en" ? "w" : "c";
  
          // Vocabulary Richness Baselines
          document.getElementById("ttr-gutenberg-val").textContent = `${((base.gutenberg.ttr || 0) * 100).toFixed(1)}%`;
          document.getElementById("ttr-webnovel-val").textContent = `${((base.webnovel.ttr || 0) * 100).toFixed(1)}%`;
  
          // Dialogue Ratio Baselines
          document.getElementById("dialogue-gutenberg-val").textContent = `${((base.gutenberg.dialogue_ratio || 0) * 100).toFixed(1)}%`;
          document.getElementById("dialogue-webnovel-val").textContent = `${((base.webnovel.dialogue_ratio || 0) * 100).toFixed(1)}%`;
  
          // Avg Sentence Length Baselines
          document.getElementById("sent-gutenberg-val").textContent = `${(base.gutenberg.avg_sentence_len || 0).toFixed(1)}${unit}`;
          document.getElementById("sent-webnovel-val").textContent = `${(base.webnovel.avg_sentence_len || 0).toFixed(1)}${unit}`;
  ```

- [ ] **Step 4: Verify Astro compile succeeds**
  Run: `npm run build --prefix frontend`
  Expected: exit 0 (compile successfully)

- [ ] **Step 5: Stage Frontend changes**
  Run: `git add frontend/src/pages/analyze.astro`

---

### Task 3: E2E Integration Verification

- [ ] **Step 1: Restart servers and verify visually**
  Restart the backend server and frontend development environment, paste custom text, trigger analysis, and confirm that the Classic Lit and Web Novel baseline values render beautifully underneath their respective metrics.
