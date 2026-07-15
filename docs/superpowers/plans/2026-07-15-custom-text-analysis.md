# Custom Text Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a custom text analysis feature with a POST API endpoint and a split-pane `/analyze` Astro frontend page.

**Architecture:** A POST route `/api/analyze` parses user-submitted prose, runs NLP style feature extraction, and matches it to a prose archetype. The Astro frontend uses a responsive split-pane layout to submit inputs and render live metrics.

**Tech Stack:** FastAPI, SQLModel, Astro, CSS, Vanilla JS

## Global Constraints
- Do not commit mock or placeholder logic.
- Ensure all interactive elements have unique, descriptive HTML IDs.
- Keep typography, colors, and layout highly aesthetic (matching dark theme, glassmorphism, Inter/Outfit fonts).

---

### Task 1: Backend POST /api/analyze Endpoint

**Files:**
- Modify: `kisholens/api/main.py`
- Create: `/Users/jonathan/.gemini/antigravity-cli/brain/44415173-3aac-40be-9614-3ddd98be293c/scratch/test_api_analyze.py`

**Interfaces:**
- Consumes: `extract_english_features`, `extract_japanese_features`, `extract_chinese_features`, `match_archetype` from `kisholens.ml.features`
- Produces: `POST /api/analyze` accepting `AnalysisRequest` and returning `detected_lang`, `features`, and `archetype`

- [ ] **Step 1: Write the failing API test script**
  Create the test script at `/Users/jonathan/.gemini/antigravity-cli/brain/44415173-3aac-40be-9614-3ddd98be293c/scratch/test_api_analyze.py`:
  ```python
  import sys
  from fastapi.testclient import TestClient
  
  # Add project path to python path
  sys.path.append("/Users/jonathan/Documents/KishoLens")
  from kisholens.api.main import app
  
  client = TestClient(app)
  
  def test_english_auto_detection():
      response = client.post("/api/analyze", json={
          "text": "This is a simple English sentence to test the auto detection capabilities.",
          "lang": "auto",
          "title": "English Test"
      })
      assert response.status_code == 200
      data = response.json()
      assert data["status"] == "success"
      assert data["detected_lang"] == "en"
      assert "features" in data
      assert "archetype" in data
  
  def test_japanese_auto_detection():
      response = client.post("/api/analyze", json={
          "text": "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
          "lang": "auto",
          "title": "Japanese Test"
      })
      assert response.status_code == 200
      data = response.json()
      assert data["detected_lang"] == "ja"
  
  def test_chinese_auto_detection():
      response = client.post("/api/analyze", json={
          "text": "这是测试自动检测中文句子的功能。非常简单且直接。",
          "lang": "auto",
          "title": "Chinese Test"
      })
      assert response.status_code == 200
      data = response.json()
      assert data["detected_lang"] == "zh"
  
  if __name__ == "__main__":
      try:
          test_english_auto_detection()
          test_japanese_auto_detection()
          test_chinese_auto_detection()
          print("ALL API TESTS PASSED!")
      except AssertionError as e:
          print("Test verification failed:", e)
          sys.exit(1)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `uv run python /Users/jonathan/.gemini/antigravity-cli/brain/44415173-3aac-40be-9614-3ddd98be293c/scratch/test_api_analyze.py`
  Expected: FAIL with Import/Module/Route errors (since endpoint `/api/analyze` is not defined).

- [ ] **Step 3: Implement the endpoint in kisholens/api/main.py**
  Add the Pydantic schema, the regex language detection helper, and the endpoint in [main.py](file:///Users/jonathan/Documents/KishoLens/kisholens/api/main.py):
  
  ```python
  import re
  
  class AnalysisRequest(BaseModel):
      text: str
      lang: str = "auto"
      title: str = "Untitled"
  
  def detect_language(text: str) -> str:
      if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
          return "ja"
      if re.search(r'[\u4e00-\u9fff]', text):
          return "zh"
      return "en"
  
  @app.post("/api/analyze")
  def post_analyze(request: AnalysisRequest):
      if not request.text.strip():
          raise HTTPException(status_code=400, detail="Text content cannot be empty")
      
      lang = request.lang
      if lang == "auto":
          lang = detect_language(request.text)
          
      if lang == "en":
          features = extract_english_features(request.text)
      elif lang == "ja":
          features = extract_japanese_features(request.text)
      elif lang == "zh":
          features = extract_chinese_features(request.text)
      else:
          raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")
          
      # format features for matcher
      agg = {f"{lang}_{k}": v for k, v in features.items()}
      archetype = match_archetype(agg)
      
      return {
          "status": "success",
          "detected_lang": lang,
          "features": features,
          "archetype": archetype
      }
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `uv run python /Users/jonathan/.gemini/antigravity-cli/brain/44415173-3aac-40be-9614-3ddd98be293c/scratch/test_api_analyze.py`
  Expected: PASS with "ALL API TESTS PASSED!"

- [ ] **Step 5: Commit/Stage API changes**
  Run: `git add kisholens/api/main.py`

---

### Task 2: Frontend `/analyze` Page Creation

**Files:**
- Modify: `frontend/src/pages/index.astro`
- Create: `frontend/src/pages/analyze.astro`

- [ ] **Step 1: Update Landing Page buttons**
  Modify [index.astro](file:///Users/jonathan/Documents/KishoLens/frontend/src/pages/index.astro#L33-L36):
  ```html
          <div class="hero__actions">
            <a href="/analyze" class="btn btn--primary" id="analyze-btn">Analyze Prose</a>
            <a href="/library" class="btn btn--ghost" id="explore-btn">Explore Library</a>
          </div>
  ```

- [ ] **Step 2: Create the /analyze Astro page**
  Create [analyze.astro](file:///Users/jonathan/Documents/KishoLens/frontend/src/pages/analyze.astro) containing the split-pane markup, styling, and JavaScript logic:
  
  ```html
  ---
  import "../styles/global.css";
  ---
  
  <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <meta name="description" content="Paste your prose and analyze stylistic pacing, dialogue ratios, and prose archetypes." />
      <title>Analyze Prose — KishoLens</title>
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet" />
      <style>
        .app-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 2rem 1.5rem 4rem 1.5rem;
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }
        header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          border-bottom: 1px solid var(--color-border);
          padding-bottom: 1.5rem;
        }
        .brand-title {
          font-family: var(--font-display);
          font-size: 1.8rem;
          font-weight: 700;
          text-decoration: none;
          color: var(--color-text);
        }
        .brand-title span {
          background: linear-gradient(135deg, var(--color-accent), var(--color-accent-2));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .analyze-layout {
          display: grid;
          grid-template-columns: 1.2fr 1fr;
          gap: 2rem;
        }
        @media (max-width: 900px) {
          .analyze-layout {
            grid-template-columns: 1fr;
          }
        }
        .card {
          background: rgba(255, 255, 255, 0.02);
          border: 1px solid var(--color-border);
          border-radius: var(--radius);
          padding: 1.5rem;
          backdrop-filter: blur(12px);
        }
        .form-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          margin-bottom: 1.25rem;
        }
        label {
          font-size: 0.85rem;
          font-weight: 500;
          color: var(--color-text);
        }
        input[type="text"], select, textarea {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid var(--color-border);
          border-radius: 8px;
          padding: 0.75rem;
          color: var(--color-text);
          font-family: var(--font-sans);
          font-size: 0.95rem;
          outline: none;
          transition: border-color var(--transition);
        }
        input[type="text"]:focus, select:focus, textarea:focus {
          border-color: var(--color-accent);
        }
        textarea {
          resize: vertical;
          min-height: 250px;
        }
        .counter-row {
          display: flex;
          justify-content: space-between;
          font-size: 0.8rem;
          color: var(--color-muted);
          margin-top: -0.75rem;
          margin-bottom: 1.25rem;
        }
        .placeholder-text {
          color: var(--color-muted);
          text-align: center;
          padding: 5rem 1rem;
          font-style: italic;
        }
        .error-alert {
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: rgb(248, 113, 113);
          padding: 0.75rem 1rem;
          border-radius: 8px;
          font-size: 0.9rem;
          margin-bottom: 1.25rem;
          display: none;
        }
        
        /* Loading Shimmer Spinner */
        .loading-container {
          display: none;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 5rem 1rem;
          gap: 1.5rem;
        }
        .spinner {
          width: 50px;
          height: 50px;
          border: 3px solid rgba(124, 106, 255, 0.1);
          border-top: 3px solid var(--color-accent);
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
  
        /* Results Section Styling */
        #results-container {
          display: none;
          flex-direction: column;
          gap: 1.5rem;
        }
        .archetype-banner {
          background: linear-gradient(135deg, rgba(124, 106, 255, 0.1), rgba(192, 132, 252, 0.05));
          border: 1px solid var(--color-accent);
          box-shadow: 0 0 15px rgba(124, 106, 255, 0.1);
          border-radius: var(--radius);
          padding: 1.5rem;
          position: relative;
          overflow: hidden;
        }
        .archetype-banner::before {
          content: '';
          position: absolute;
          top: 0; left: 0; right: 0; height: 3px;
          background: linear-gradient(90deg, var(--color-accent), var(--color-accent-2));
        }
        .archetype-title-row {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          margin-bottom: 0.5rem;
        }
        .archetype-name {
          font-family: var(--font-display);
          font-size: 1.5rem;
          font-weight: 700;
          color: #fff;
        }
        .archetype-confidence {
          font-size: 0.9rem;
          font-weight: 600;
          color: var(--color-accent-2);
          background: rgba(192, 132, 252, 0.1);
          padding: 0.15rem 0.5rem;
          border-radius: 999px;
        }
        .archetype-description {
          font-size: 0.95rem;
          color: var(--color-text);
          line-height: 1.5;
        }
        .metric-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1rem;
        }
        @media (max-width: 500px) {
          .metric-grid {
            grid-template-columns: 1fr;
          }
        }
        .metric-card {
          background: rgba(255, 255, 255, 0.01);
          border: 1px solid var(--color-border);
          border-radius: 8px;
          padding: 1rem;
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }
        .metric-label {
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: var(--color-muted);
        }
        .metric-value {
          font-family: var(--font-display);
          font-size: 1.4rem;
          font-weight: 600;
          color: #fff;
        }
        .progress-bar-container {
          width: 100%;
          height: 6px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 3px;
          overflow: hidden;
          margin-top: 0.5rem;
        }
        .progress-bar-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--color-accent), var(--color-accent-2));
          width: 0%;
          transition: width 0.5s ease-out;
        }
      </style>
    </head>
    <body>
      <div class="hero__glow" aria-hidden="true"></div>
      <div class="app-container">
        <header>
          <a href="/" class="brand-title">Kish<span>o</span>Lens</a>
          <a href="/" class="btn btn--ghost" id="back-btn">← Back</a>
        </header>
  
        <div class="analyze-layout">
          <!-- Left Pane: Input Form -->
          <div class="card">
            <h2 style="font-family: var(--font-display); margin-bottom: 1.5rem;">Input Passage</h2>
            <form id="analyze-form">
              <div class="form-group">
                <label for="passage-title">Passage Title (Optional)</label>
                <input type="text" id="passage-title" placeholder="e.g. Chapter 1, Draft 2" />
              </div>
              <div class="form-group">
                <label for="passage-lang">Language</label>
                <select id="passage-lang">
                  <option value="auto">Auto-detect (recommended)</option>
                  <option value="en">English</option>
                  <option value="ja">Japanese</option>
                  <option value="zh">Chinese</option>
                </select>
              </div>
              <div class="form-group">
                <label for="passage-text">Prose Content</label>
                <textarea id="passage-text" placeholder="Paste your writing here (at least 20 words recommended)..." required></textarea>
              </div>
              <div class="counter-row">
                <span id="char-counter">0 characters</span>
                <span id="word-counter">0 words</span>
              </div>
              <button type="submit" class="btn btn--primary" id="analyze-submit" style="width: 100%;">Analyze Prose</button>
            </form>
          </div>
  
          <!-- Right Pane: Analysis Results -->
          <div class="card" style="display: flex; flex-direction: column;">
            <h2 style="font-family: var(--font-display); margin-bottom: 1.5rem;">Stylistic Analysis</h2>
            
            <div class="error-alert" id="error-box"></div>
  
            <!-- State 1: Ready -->
            <div class="placeholder-text" id="ready-state">
              Pasted text analysis will appear here. Paste writing in the input pane and click "Analyze Prose".
            </div>
  
            <!-- State 2: Loading -->
            <div class="loading-container" id="loading-state">
              <div class="spinner"></div>
              <p style="color: var(--color-muted);">Extracting stylistic pacing metrics...</p>
            </div>
  
            <!-- State 3: Success -->
            <div id="results-container">
              <!-- Archetype Card -->
              <div class="archetype-banner">
                <div class="archetype-title-row">
                  <span class="archetype-name" id="result-archetype">Suspenseful Pacing</span>
                  <span class="archetype-confidence" id="result-confidence">Match: 82%</span>
                </div>
                <p class="archetype-description" id="result-description">
                  Fast pacing, low descriptive complexity, high dialogue ratio. Focuses heavily on action-beats.
                </p>
              </div>
  
              <!-- Metrics Cards Grid -->
              <div class="metric-grid">
                <div class="metric-card">
                  <span class="metric-label">Language Used</span>
                  <span class="metric-value" style="color: var(--color-accent-2);" id="metric-lang">English</span>
                </div>
                <div class="metric-card">
                  <span class="metric-label">Vocabulary Richness</span>
                  <span class="metric-value" id="metric-ttr">0.00%</span>
                  <span style="font-size: 0.75rem; color: var(--color-muted);">Type-Token Ratio</span>
                  <div class="progress-bar-container">
                    <div class="progress-bar-fill" id="ttr-progress"></div>
                  </div>
                </div>
                <div class="metric-card">
                  <span class="metric-label">Dialogue Ratio</span>
                  <span class="metric-value" id="metric-dialogue">0.0%</span>
                  <span style="font-size: 0.75rem; color: var(--color-muted);">Spoken vs. Exposition</span>
                  <div class="progress-bar-container">
                    <div class="progress-bar-fill" id="dialogue-progress"></div>
                  </div>
                </div>
                <div class="metric-card">
                  <span class="metric-label">Avg Sentence Length</span>
                  <span class="metric-value" id="metric-sent-len">0.0 words</span>
                </div>
                <div class="metric-card">
                  <span class="metric-label">Sentence Count</span>
                  <span class="metric-value" id="metric-sent-count">0</span>
                </div>
                <div class="metric-card">
                  <span class="metric-label">Pacing Score</span>
                  <span class="metric-value" id="metric-pacing">0.00</span>
                  <span style="font-size: 0.75rem; color: var(--color-muted);">High value = slower pacing</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
  
      <script is:inline>
        const API_URL = "http://127.0.0.1:8000";
        const textarea = document.getElementById("passage-text");
        const charCounter = document.getElementById("char-counter");
        const wordCounter = document.getElementById("word-counter");
        const form = document.getElementById("analyze-form");
        const errorBox = document.getElementById("error-box");
        
        const readyState = document.getElementById("ready-state");
        const loadingState = document.getElementById("loading-state");
        const resultsContainer = document.getElementById("results-container");
  
        // 1. Text Area character and word counter
        textarea.addEventListener("input", () => {
          const text = textarea.value;
          charCounter.textContent = `${text.length} characters`;
          
          const words = text.trim() === "" ? 0 : text.trim().split(/\s+/).length;
          wordCounter.textContent = `${words} words`;
        });
  
        // 2. Submit form to API
        form.addEventListener("submit", async (e) => {
          e.preventDefault();
          errorBox.style.display = "none";
          readyState.style.display = "none";
          resultsContainer.style.display = "none";
          loadingState.style.display = "flex";
  
          const title = document.getElementById("passage-title").value || "Untitled";
          const lang = document.getElementById("passage-lang").value;
          const text = textarea.value;
  
          try {
            const res = await fetch(`${API_URL}/api/analyze`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ title, lang, text })
            });
  
            if (!res.ok) {
              const errData = await res.json();
              throw new Error(errData.detail || "Server failed to process analysis");
            }
  
            const data = await res.json();
            renderResults(data);
          } catch (err) {
            loadingState.style.display = "none";
            errorBox.textContent = `Analysis error: ${err.message}`;
            errorBox.style.display = "block";
            readyState.style.display = "block";
          }
        });
  
        function renderResults(data) {
          loadingState.style.display = "none";
          
          // Set lang text
          const langMap = { "en": "English", "ja": "Japanese", "zh": "Chinese" };
          document.getElementById("metric-lang").textContent = langMap[data.detected_lang] || data.detected_lang;
  
          // Set archetype
          const arch = data.archetype;
          document.getElementById("result-archetype").textContent = arch.archetype || "Unknown Style";
          document.getElementById("result-confidence").textContent = `Match: ${Math.round(arch.confidence * 100)}%`;
          document.getElementById("result-description").textContent = arch.description || "";
  
          // Set metrics
          const feats = data.features;
          
          // TTR (vocabulary richness)
          const ttr = (feats.type_token_ratio || 0) * 100;
          document.getElementById("metric-ttr").textContent = `${ttr.toFixed(1)}%`;
          document.getElementById("ttr-progress").style.width = `${ttr}%`;
  
          // Dialogue Ratio
          const dialogue = (feats.dialogue_ratio || 0) * 100;
          document.getElementById("metric-dialogue").textContent = `${dialogue.toFixed(1)}%`;
          document.getElementById("dialogue-progress").style.width = `${dialogue}%`;
  
          // Sentence metrics
          document.getElementById("metric-sent-count").textContent = feats.sentence_count || 0;
          
          const avgSentLen = feats.avg_sentence_length || 0;
          const unit = data.detected_lang === "en" ? "words" : "chars";
          document.getElementById("metric-sent-len").textContent = `${avgSentLen.toFixed(1)} ${unit}`;
  
          // Pacing score
          const pacing = feats.pacing_score || 0;
          document.getElementById("metric-pacing").textContent = pacing.toFixed(2);
  
          resultsContainer.style.display = "flex";
        }
      </script>
    </body>
  </html>
  ```

- [ ] **Step 3: Verify the Astro build**
  Run: `npm run build --prefix frontend`
  Expected: exit code 0 (successful compilation)

---

### Task 3: E2E Integration and Manual Checkout

- [ ] **Step 1: Run the Backend & Frontend servers**
  Launch servers in background tasks or manual run processes, verifying logs show both running on port 8000 and port 4321 respectively.

- [ ] **Step 2: Access `/analyze` and perform a test paste**
  Open browser link, click "Analyze Prose", type custom test paragraphs, verify metrics update successfully and match visual expectations.
