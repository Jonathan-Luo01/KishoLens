# Design Specification: Enhanced Dashboard Loading State (Skeleton UI & Multi-Stage Progress Indicator)

**Date:** 2026-08-10  
**Feature:** Frontend Loading State & Chart Rendering Feedback  
**Target Files:**  
- `frontend/src/pages/library.astro`  
- `frontend/src/pages/analyze.astro` (shared styling/patterns)

---

## 1. Overview & Problem Statement
When exploring novels in KishoLens, fetching comprehensive metrics (`/api/novels/{id}/stats`) and sentence-level sentiment trajectories (`/api/novels/{id}/arc`) as well as rendering Chart.js visualizations can take up to 8–10 seconds.

Currently, selecting a novel in `library.astro` hides the entire charts grid (`display: none`) and replaces the dashboard with a single unstyled text line:
```html
<p class="placeholder-text" style="padding: 1.5rem 1rem;">Fetching stats & rendering charts...</p>
```
This causes the interface to feel unresponsive, empty, or frozen during the fetch duration.

---

## 2. Goals & Success Criteria
1. **Immediate Visual Feedback**: Within 0ms of clicking a novel, display a rich, structured skeleton layout resembling the final dashboard.
2. **Dynamic Progress Stages**: Provide dynamic, descriptive stage indicators ("Extracting metrics...", "Mapping sentiment arc...", "Synthesizing archetype...") that inform the user of background analysis milestones.
3. **Smooth State Transitions**: Transition smoothly from skeleton placeholders to live Chart.js visualizations without jarring layout shifts (CLS < 0.05).
4. **Resilient Cleanup**: Ensure rapid switching between novels properly cancels prior stage timers and prevents race conditions.

---

## 3. Architecture & Visual Components

### 3.1 Skeleton Layout Structure
When `selectNovel(id, title)` is triggered, the details panel renders:

1. **Header & Progress Banner**:
   - Title: `[Novel Title] — Analyzing Stylistic DNA...`
   - Top linear shimmering progress bar (gradient sweep from cyan `#06b6d4` to purple `#a855f7`).
   - Dynamic stage badge with pulsing status dot and activity message.
2. **Category Tabs Skeleton**:
   - 5 pulsing pill skeletons matching the category filter buttons.
3. **Metric Card Grid Skeletons (6 cards)**:
   - Shimmering glassmorphism cards with placeholder label, value, and percentile bar.
4. **Visualization Skeletons**:
   - **Radar Chart Skeleton**: Circular glowing radar wireframe with concentric guides.
   - **Sentiment Arc Skeleton**: Chart canvas placeholder with pulsing bezier wave sine curve and 4 Kishotenketsu section guides (Ki, Sho, Ten, Ketsu).
   - **Pacing Barcode Skeleton**: Horizontal strip of animated bars with varied heights.
   - **Doppelgänger Skeletons**: 3 recommendation card skeletons with miniature horizontal factor breakdown bars.

---

## 4. Multi-Stage Progress Timeline

A lightweight interval manager runs while `Promise.all([fetchStats, fetchArc])` is pending:

| Time Window | Step # | Stage Name | Accent Color | Subtext Description |
| :--- | :--- | :--- | :--- | :--- |
| **0s – 3.0s** | Step 1/3 | Extracting Stylistic Prose Metrics | Cyan (`#06b6d4`) | Parsing syntactic depth, vocabulary diversity, and dialogue ratios... |
| **3.0s – 7.0s** | Step 2/3 | Computing Kishotenketsu Sentiment Arc | Purple (`#a855f7`) | Mapping 4-phase quantile emotional trajectories across narrative... |
| **7.0s – 10s+** | Step 3/3 | Synthesizing Archetype Radar & Doppelgängers | Emerald (`#10b981`) | Computing 5-factor similarity vectors & rendering interactive charts... |

When the network requests resolve, the progress bar snaps to 100% and seamlessly cross-fades into the rendered dashboard.

---

## 5. CSS Animations & Styling Tokens

All styling adheres to the KishoLens dark glassmorphism design system:

```css
/* Shimmer Sweep Animation */
@keyframes skeleton-shimmer {
  0% {
    background-position: -200% 0;
  }
  100% {
    background-position: 200% 0;
  }
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

/* Pulse Glow for Chart Frames */
@keyframes skeleton-pulse-glow {
  0%, 100% {
    border-color: rgba(148, 163, 184, 0.1);
    box-shadow: 0 0 0 0 rgba(6, 182, 212, 0);
  }
  50% {
    border-color: rgba(6, 182, 212, 0.25);
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.08);
  }
}
```

---

## 6. Error Handling & Edge Cases
- **Network Error / Failure**: If either request fails or returns a non-200 status, clear the stage interval, display a structured error banner with a "Retry" button, and reset the skeleton state.
- **Rapid Navigation (Debounce / Cancellation)**: Store a unique `selectionToken` or `AbortController` in `selectNovel`. If the user selects Novel B while Novel A is still loading, discard Novel A's response upon arrival and cancel Novel A's progress timer.
- **Empty Dataset**: If metrics return empty (`{}`), display an empty state banner gracefully without unhandled exceptions.

---

## 7. Verification & Testing Plan
1. **Visual Inspection**:
   - Select a novel in Library (`npm run dev` / Astro frontend).
   - Verify skeleton renders instantly (0ms) across all 4 visualization sections.
   - Verify progress bar advances through Step 1 $\rightarrow$ Step 2 $\rightarrow$ Step 3.
   - Verify smooth transition into live Chart.js charts without UI flash.
2. **Build Verification**:
   - Run `cd frontend && npm run build` to ensure zero Astro/TypeScript compilation errors.
3. **Unit / Integration Tests**:
   - Run `uv run pytest tests/` to ensure backend APIs remain fully compliant.
