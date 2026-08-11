# Enhanced Dashboard Loading State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide an engaging, responsive loading experience with shimmering skeleton placeholders (metrics, radar, sentiment arc, pacing barcode, doppelgänger cards) and a dynamic 3-stage progress indicator during the ~8–10 second novel metrics fetch.

**Architecture:** Implement CSS shimmer & pulse keyframe animations, build modular skeleton renderers in `frontend/src/pages/library.astro`, and integrate a multi-stage loading timeline into `selectNovel` with request race-condition guards and smooth crossfade transitions.

**Tech Stack:** Astro, Vanilla CSS3 animations, SVG vector placeholders, Chart.js.

## Global Constraints
- Target File: `frontend/src/pages/library.astro`
- Design Tokens: Dark glassmorphism (`rgba(255,255,255,0.03)` to `0.10`), Cyan `#06b6d4`, Purple `#a855f7`, Emerald `#10b981`.
- Verification: `cd frontend && npm run build` and `uv run pytest tests/` must pass cleanly with 0 errors.
- No third-party UI libraries: use pure vanilla CSS and SVG.

---

### Task 1: Skeleton Design Tokens & CSS Shimmer Keyframe Animations

**Files:**
- Modify: `frontend/src/pages/library.astro` (in `<style>` section around line 150-250)

**Interfaces:**
- Produces CSS classes: `.skeleton-shimmer`, `.skeleton-box`, `.skeleton-card`, `.skeleton-progress-wrapper`, `.skeleton-progress-fill`, `.skeleton-stage-pill`, `.skeleton-stage-dot`, `.skeleton-chart-frame`, `.skeleton-svg-wave`.

- [ ] **Step 1: Add skeleton CSS rules and animations to `library.astro`**

```css
/* ── Skeleton Loading UI & Shimmer Keyframes ── */
@keyframes skeleton-shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

@keyframes skeleton-pulse-glow {
  0%, 100% {
    border-color: rgba(148, 163, 184, 0.12);
    box-shadow: 0 0 0 0 rgba(6, 182, 212, 0);
  }
  50% {
    border-color: rgba(6, 182, 212, 0.35);
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.10);
  }
}

@keyframes pulse-dot {
  0%, 100% { transform: scale(1); opacity: 0.8; }
  50% { transform: scale(1.3); opacity: 1; }
}

.skeleton-shimmer {
  background: linear-gradient(
    90deg,
    rgba(255, 255, 255, 0.03) 0%,
    rgba(255, 255, 255, 0.09) 50%,
    rgba(255, 255, 255, 0.03) 100%
  );
  background-size: 200% 100%;
  animation: skeleton-shimmer 1.8s infinite ease-in-out;
}

.skeleton-progress-container {
  width: 100%;
  margin-bottom: 1.25rem;
  background: rgba(15, 23, 42, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.15);
  border-radius: 12px;
  padding: 1rem 1.25rem;
  backdrop-filter: blur(12px);
}

.skeleton-progress-bar-bg {
  width: 100%;
  height: 6px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 999px;
  overflow: hidden;
  margin-top: 0.75rem;
}

.skeleton-progress-bar-fill {
  height: 100%;
  width: 15%;
  background: linear-gradient(90deg, #06b6d4, #a855f7, #10b981);
  background-size: 200% 100%;
  border-radius: 999px;
  transition: width 0.4s ease-out;
  animation: skeleton-shimmer 2s infinite linear;
}

.skeleton-stage-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.skeleton-stage-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.8125rem;
  font-weight: 600;
  color: #38bdf8;
  background: rgba(56, 189, 248, 0.1);
  border: 1px solid rgba(56, 189, 248, 0.25);
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
}

.skeleton-stage-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #38bdf8;
  animation: pulse-dot 1.2s infinite ease-in-out;
}

.skeleton-card {
  background: rgba(15, 23, 42, 0.45);
  border: 1px solid rgba(148, 163, 184, 0.1);
  border-radius: 12px;
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.skeleton-chart-frame {
  background: rgba(15, 23, 42, 0.5);
  border: 1px solid rgba(148, 163, 184, 0.15);
  border-radius: 14px;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  animation: skeleton-pulse-glow 3s infinite ease-in-out;
}

.fade-in-content {
  animation: fadeIn 0.35s ease-out forwards;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}
```

- [ ] **Step 2: Verify CSS build**

Run: `cd /Users/jonathan/Documents/KishoLens/frontend && npm run build`
Expected: Build finishes with zero errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/library.astro
git commit -m "feat(ui): add skeleton shimmer animations and design tokens to library dashboard"
```

---

### Task 2: Implement Skeleton Layout Builders in `library.astro`

**Files:**
- Modify: `frontend/src/pages/library.astro` (Client script section)

**Interfaces:**
- Produces:
  - `renderSkeletonDashboard(title: string): string`
  - `renderSkeletonSimilarNovels(): string`

- [ ] **Step 1: Write `renderSkeletonDashboard` and `renderSkeletonSimilarNovels` functions**

```javascript
function renderSkeletonDashboard(novelTitle) {
  return `
    <div class="skeleton-progress-container">
      <div class="skeleton-stage-header">
        <div class="skeleton-stage-pill" id="skeletonStagePill">
          <span class="skeleton-stage-dot" id="skeletonStageDot"></span>
          <span id="skeletonStageText">Step 1/3: Extracting Stylistic Metrics...</span>
        </div>
        <span style="font-size: 0.75rem; color: #94a3b8; font-family: monospace;" id="skeletonElapsedTime">0.0s elapsed</span>
      </div>
      <p id="skeletonSubtext" style="font-size: 0.8125rem; color: #cbd5e1; margin: 0.5rem 0 0 0;">
        Parsing sentence structure, syntactic depth, and lexical richness...
      </p>
      <div class="skeleton-progress-bar-bg">
        <div class="skeleton-progress-bar-fill" id="skeletonProgressFill" style="width: 15%;"></div>
      </div>
    </div>

    <!-- Category Tabs Skeleton -->
    <div style="display: flex; gap: 0.5rem; margin-bottom: 1.25rem; flex-wrap: wrap;">
      <div class="skeleton-shimmer" style="height: 32px; width: 64px; border-radius: 8px;"></div>
      <div class="skeleton-shimmer" style="height: 32px; width: 110px; border-radius: 8px;"></div>
      <div class="skeleton-shimmer" style="height: 32px; width: 95px; border-radius: 8px;"></div>
      <div class="skeleton-shimmer" style="height: 32px; width: 105px; border-radius: 8px;"></div>
      <div class="skeleton-shimmer" style="height: 32px; width: 100px; border-radius: 8px;"></div>
    </div>

    <!-- Metrics Cards Skeleton Grid -->
    <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
      ${[1,2,3,4,5,6].map(() => `
        <div class="skeleton-card">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="skeleton-shimmer" style="height: 14px; width: 55%; border-radius: 4px;"></div>
            <div class="skeleton-shimmer" style="height: 18px; width: 18px; border-radius: 50%;"></div>
          </div>
          <div class="skeleton-shimmer" style="height: 28px; width: 40%; border-radius: 6px; margin: 0.25rem 0;"></div>
          <div class="skeleton-shimmer" style="height: 8px; width: 100%; border-radius: 4px;"></div>
        </div>
      `).join('')}
    </div>

    <!-- Visualizations Skeletons Grid -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem;">
      <!-- Radar Skeleton -->
      <div class="skeleton-chart-frame">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div class="skeleton-shimmer" style="height: 16px; width: 45%; border-radius: 4px;"></div>
          <div class="skeleton-shimmer" style="height: 14px; width: 25%; border-radius: 4px;"></div>
        </div>
        <div style="height: 260px; display: flex; align-items: center; justify-content: center; position: relative;">
          <svg width="200" height="200" viewBox="0 0 200 200" style="opacity: 0.35;">
            <polygon points="100,20 180,60 180,140 100,180 20,140 20,60" fill="none" stroke="#06b6d4" stroke-width="1.5" stroke-dasharray="4 4" />
            <polygon points="100,50 150,75 150,125 100,150 50,125 50,75" fill="none" stroke="#a855f7" stroke-width="1.5" />
            <circle cx="100" cy="100" r="4" fill="#06b6d4" />
            <line x1="100" y1="20" x2="100" y2="180" stroke="rgba(255,255,255,0.1)" stroke-width="1" />
            <line x1="20" y1="60" x2="180" y2="140" stroke="rgba(255,255,255,0.1)" stroke-width="1" />
            <line x1="20" y1="140" x2="180" y2="60" stroke="rgba(255,255,255,0.1)" stroke-width="1" />
          </svg>
        </div>
      </div>

      <!-- Sentiment Arc Skeleton -->
      <div class="skeleton-chart-frame">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div class="skeleton-shimmer" style="height: 16px; width: 50%; border-radius: 4px;"></div>
          <div class="skeleton-shimmer" style="height: 14px; width: 30%; border-radius: 4px;"></div>
        </div>
        <div style="height: 260px; display: flex; flex-direction: column; justify-content: space-between; padding: 1rem 0;">
          <svg width="100%" height="160" viewBox="0 0 300 120" preserveAspectRatio="none" style="opacity: 0.4;">
            <line x1="0" y1="60" x2="300" y2="60" stroke="rgba(255,255,255,0.15)" stroke-width="1" stroke-dasharray="2 2" />
            <line x1="75" y1="0" x2="75" y2="120" stroke="rgba(255,255,255,0.08)" stroke-width="1" />
            <line x1="150" y1="0" x2="150" y2="120" stroke="rgba(255,255,255,0.08)" stroke-width="1" />
            <line x1="225" y1="0" x2="225" y2="120" stroke="rgba(255,255,255,0.08)" stroke-width="1" />
            <path d="M 0 70 Q 75 20, 150 70 T 300 40" fill="none" stroke="#a855f7" stroke-width="2.5" stroke-linecap="round" />
          </svg>
          <div style="display: flex; justify-content: space-between;">
            <div class="skeleton-shimmer" style="height: 10px; width: 18%; border-radius: 3px;"></div>
            <div class="skeleton-shimmer" style="height: 10px; width: 18%; border-radius: 3px;"></div>
            <div class="skeleton-shimmer" style="height: 10px; width: 18%; border-radius: 3px;"></div>
            <div class="skeleton-shimmer" style="height: 10px; width: 18%; border-radius: 3px;"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Pacing Barcode Skeleton -->
    <div class="skeleton-chart-frame" style="margin-top: 1.5rem;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="skeleton-shimmer" style="height: 16px; width: 35%; border-radius: 4px;"></div>
        <div class="skeleton-shimmer" style="height: 14px; width: 20%; border-radius: 4px;"></div>
      </div>
      <div style="display: flex; align-items: flex-end; gap: 3px; height: 70px; padding: 0.5rem 0; overflow: hidden;">
        ${Array.from({length: 48}).map((_, i) => `
          <div class="skeleton-shimmer" style="flex: 1; height: ${20 + ((i * 17) % 55)}px; border-radius: 2px;"></div>
        `).join('')}
      </div>
    </div>
  `;
}

function renderSkeletonSimilarNovels() {
  return `
    <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem;">
      ${[1,2,3].map(() => `
        <div class="skeleton-card">
          <div class="skeleton-shimmer" style="height: 14px; width: 70%; border-radius: 4px;"></div>
          <div class="skeleton-shimmer" style="height: 12px; width: 45%; border-radius: 4px;"></div>
          <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
            <div class="skeleton-shimmer" style="height: 16px; width: 35%; border-radius: 4px;"></div>
            <div class="skeleton-shimmer" style="height: 16px; width: 25%; border-radius: 4px;"></div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}
```

- [ ] **Step 2: Verify frontend compilation**

Run: `cd /Users/jonathan/Documents/KishoLens/frontend && npm run build`
Expected: Passes with 0 errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/library.astro
git commit -m "feat(ui): implement skeleton dashboard and recommendation builders"
```

---

### Task 3: Multi-Stage Progress Controller & Integration in `selectNovel`

**Files:**
- Modify: `frontend/src/pages/library.astro` (inside `selectNovel` around line 2190-2300)

**Interfaces:**
- Consumes: `renderSkeletonDashboard`, `renderSkeletonSimilarNovels`
- Produces: dynamic stage transition lifecycle with timer cleanup and `currentLoadId` race-condition token.

- [ ] **Step 1: Integrate stage timer and skeleton activation into `selectNovel`**

```javascript
let currentLoadingTimer = null;
let currentLoadId = 0;

async function selectNovel(id, title) {
  const loadId = ++currentLoadId;
  if (currentLoadingTimer) {
    clearInterval(currentLoadingTimer);
    currentLoadingTimer = null;
  }

  // Highlight active card in grid
  document.querySelectorAll(".card-novel").forEach(c => {
    if (c.getAttribute("data-novel-id") == id) {
      c.classList.add("active");
    } else {
      c.classList.remove("active");
    }
  });

  // Sync dropdown
  const dropdown = document.getElementById("novelSelect");
  if (dropdown) dropdown.value = id;

  currentStats = null;
  currentArc = null;

  const statsContent = document.getElementById("stats-content");
  const titleElement = document.getElementById("novel-details-title");
  const chartsGrid = document.getElementById("chartsGrid");
  const similarContainer = document.getElementById("similarNovelsContainer");

  if (titleElement) titleElement.innerText = `${title} — Analyzing Stylistic DNA...`;
  if (chartsGrid) chartsGrid.style.display = "none"; // Hide real charts while skeleton is active
  if (statsContent) statsContent.innerHTML = renderSkeletonDashboard(title);
  if (similarContainer) similarContainer.innerHTML = renderSkeletonSimilarNovels();

  // Multi-stage progress animation timeline
  const startTime = Date.now();
  currentLoadingTimer = setInterval(() => {
    if (loadId !== currentLoadId) {
      clearInterval(currentLoadingTimer);
      return;
    }
    const elapsed = (Date.now() - startTime) / 1000;
    const elapsedEl = document.getElementById("skeletonElapsedTime");
    const pillEl = document.getElementById("skeletonStagePill");
    const textEl = document.getElementById("skeletonStageText");
    const dotEl = document.getElementById("skeletonStageDot");
    const subtextEl = document.getElementById("skeletonSubtext");
    const progressFill = document.getElementById("skeletonProgressFill");

    if (elapsedEl) elapsedEl.innerText = `${elapsed.toFixed(1)}s elapsed`;

    if (elapsed < 3.0) {
      // Step 1: Metrics
      const pct = Math.min(45, 15 + (elapsed / 3.0) * 30);
      if (progressFill) progressFill.style.width = `${pct}%`;
      if (textEl) textEl.innerText = "Step 1/3: Extracting Stylistic Metrics...";
      if (subtextEl) subtextEl.innerText = "Parsing sentence structure, syntactic depth, and lexical richness...";
    } else if (elapsed < 7.0) {
      // Step 2: Arc
      const pct = Math.min(80, 45 + ((elapsed - 3.0) / 4.0) * 35);
      if (progressFill) progressFill.style.width = `${pct}%`;
      if (pillEl) {
        pillEl.style.color = "#c084fc";
        pillEl.style.background = "rgba(192, 132, 252, 0.1)";
        pillEl.style.borderColor = "rgba(192, 132, 252, 0.25)";
      }
      if (dotEl) dotEl.style.background = "#c084fc";
      if (textEl) textEl.innerText = "Step 2/3: Mapping Kishotenketsu Arc...";
      if (subtextEl) subtextEl.innerText = "Calculating 4-phase quantile emotional trajectories across prose...";
    } else {
      // Step 3: Synthesis & Charts
      const pct = Math.min(96, 80 + ((elapsed - 7.0) / 5.0) * 16);
      if (progressFill) progressFill.style.width = `${pct}%`;
      if (pillEl) {
        pillEl.style.color = "#34d399";
        pillEl.style.background = "rgba(52, 211, 153, 0.1)";
        pillEl.style.borderColor = "rgba(52, 211, 153, 0.25)";
      }
      if (dotEl) dotEl.style.background = "#34d399";
      if (textEl) textEl.innerText = "Step 3/3: Synthesizing Archetype Radar...";
      if (subtextEl) subtextEl.innerText = "Extracting multi-factor similarity vectors and rendering visualizations...";
    }
  }, 100);

  try {
    const [statsRes, arcRes] = await Promise.all([
      fetch(`${API_URL}/api/novels/${id}/stats`),
      fetch(`${API_URL}/api/novels/${id}/arc`)
    ]);

    if (loadId !== currentLoadId) return; // Stale request discarded
    if (currentLoadingTimer) {
      clearInterval(currentLoadingTimer);
      currentLoadingTimer = null;
    }

    if (!statsRes.ok) throw new Error("Stats request failed");
    if (!arcRes.ok) throw new Error("Arc request failed");

    const stats = await statsRes.json();
    const arc = await arcRes.json();

    if (loadId !== currentLoadId) return;

    currentStats = stats;
    currentArc = arc;

    if (titleElement) titleElement.innerText = `${title} Metrics`;

    if (Object.keys(stats).length === 0) {
      statsContent.innerHTML = `<p class="placeholder-text">No text metrics found for this novel.</p>`;
      return;
    }

    // Render live dashboard with fade-in class
    statsContent.innerHTML = `<div class="fade-in-content">${renderEnhancedMetricsDashboard(stats)}</div>`;

    // Attach event listeners for category tabs and info popovers
    attachDashboardEventListeners(statsContent);

    // Unhide and render charts
    if (chartsGrid) {
      chartsGrid.style.display = "grid";
      chartsGrid.classList.add("fade-in-content");
    }
    renderAll();
    if (stats.top_matches) {
      renderSimilarNovels(stats.top_matches);
    }
  } catch (err) {
    if (loadId !== currentLoadId) return;
    if (currentLoadingTimer) {
      clearInterval(currentLoadingTimer);
      currentLoadingTimer = null;
    }
    console.error("Error loading stats or arc:", err);
    if (titleElement) titleElement.innerText = `${title} (Error)`;
    if (statsContent) {
      statsContent.innerHTML = `
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 1.5rem; text-align: center;">
          <p style="color: #f87171; font-weight: 600; margin: 0 0 0.5rem 0;">Failed to load metrics for this novel.</p>
          <p style="color: #94a3b8; font-size: 0.8125rem; margin: 0 0 1rem 0;">${err.message || "Network request failed"}</p>
          <button class="baseline-btn" onclick="selectNovel(${id}, '${title.replace(/'/g, "\\'")}')">Retry Analysis</button>
        </div>
      `;
    }
  }
}
```

- [ ] **Step 2: Refactor tab/popover event listener attachment into reusable helper `attachDashboardEventListeners`**

- [ ] **Step 3: Run full frontend build**

Run: `cd /Users/jonathan/Documents/KishoLens/frontend && npm run build`
Expected: Build passes with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/library.astro
git commit -m "feat(library): integrate dynamic 3-stage progress loader and skeleton dashboard"
```

---

### Task 4: Verification & End-to-End Testing

**Files:**
- Test: `tests/ml/test_canonical_predictions.py`

- [ ] **Step 1: Run complete backend test suite to verify no regressions**

Run: `uv run pytest tests/ -x -q`
Expected: 63 passed, 0 failures.

- [ ] **Step 2: Run frontend build verification**

Run: `cd frontend && npm run build`
Expected: Built in < 2s with 0 warnings/errors.
