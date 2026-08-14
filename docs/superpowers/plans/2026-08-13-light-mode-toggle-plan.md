# Platform-Wide Light Mode Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a platform-wide, system-aware Light Mode Toggle across KishoLens (`/`, `/analyze`, `/library`) with zero flash-of-unstyled-theme (FOUT), dynamic canvas & SVG chart adaptation, and full mobile compatibility.

**Architecture:** A centralized CSS token override system under `html[data-theme="light"]` in `global.css`, coupled with an inline zero-FOUT `<script>` in the `<head>` of all pages that synchronizes with `localStorage` and `prefers-color-scheme`. Dynamic charts (8D Archetype Radar canvas, Kishōtenketsu Sentiment Arc SVG, Pacing Barcode canvas) adapt color palettes via dynamic theme queries and a custom `themechange` event.

**Tech Stack:** Astro, Vanilla CSS, HTML5 Canvas, SVG, JavaScript (ES6+), Agent-Browser for verification.

## Global Constraints

- Design Tokens: Light mode must use clean slate palettes (`--color-bg: #f8fafc;`, `--color-surface: #ffffff;`, `--color-text: #0f172a;`, `--color-text-subtle: #475569;`, `--color-border: rgba(0,0,0,0.09);`).
- Zero-FOUT: Theme initialization script must run synchronously inside `<head>` before body paint.
- Chart Adaptability: Canvas & SVG charts must render with high contrast in both themes without page reload.
- Mobile Compatibility: Header toggle button and layouts must be responsive down to 375px viewports.
- Verification: `cd frontend && npm run build` must pass with 0 errors.

---

### Task 1: Semantic Theme Tokens, Zero-FOUT Head Script & Header Toggle Component

**Files:**
- Modify: `frontend/src/styles/global.css:4-40`
- Modify: `frontend/src/pages/index.astro:20-50`
- Modify: `frontend/src/pages/analyze.astro:20-50`
- Modify: `frontend/src/pages/library.astro:20-50`

**Interfaces:**
- Produces: `html[data-theme="light"]` and `html[data-theme="dark"]` CSS token variables, `#themeToggleBtn` in `.site-header`, and window `themechange` event.

- [ ] **Step 1: Add Light Theme Semantic Tokens and Toggle Button Styles in `global.css`**

```css
/* In frontend/src/styles/global.css */
:root {
  --color-bg: #07090f;
  --color-surface: #0e1220;
  --color-surface-card: #0b0f19;
  --color-surface-hover: rgba(255, 255, 255, 0.04);
  --color-border: rgba(255, 255, 255, 0.08);
  --color-border-subtle: rgba(255, 255, 255, 0.04);
  --color-accent: #7c6aff;
  --color-accent-2: #c084fc;
  --color-accent-cyan: #38bdf8;
  --color-accent-emerald: #34d399;
  --color-text: #e8eaf0;
  --color-text-subtle: #94a3b8;
  --color-muted: #6b7280;
  --header-bg: rgba(2, 6, 23, 0.85);
  --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  --chart-grid: rgba(255, 255, 255, 0.10);
  --chart-text: #cbd5e1;
  --chart-axis: rgba(255, 255, 255, 0.5);
  --input-bg: rgba(0, 0, 0, 0.25);
  --skeleton-bg: rgba(255, 255, 255, 0.04);
  --skeleton-shimmer: rgba(255, 255, 255, 0.08);
}

html[data-theme="light"], :root[data-theme="light"] {
  --color-bg: #f8fafc;
  --color-surface: #ffffff;
  --color-surface-card: #ffffff;
  --color-surface-hover: #f1f5f9;
  --color-border: rgba(0, 0, 0, 0.09);
  --color-border-subtle: rgba(0, 0, 0, 0.05);
  --color-accent: #6366f1;
  --color-accent-2: #a855f7;
  --color-accent-cyan: #0284c7;
  --color-accent-emerald: #059669;
  --color-text: #0f172a;
  --color-text-subtle: #475569;
  --color-muted: #64748b;
  --header-bg: rgba(255, 255, 255, 0.88);
  --card-shadow: 0 4px 24px -2px rgba(15, 23, 42, 0.06), 0 2px 6px -1px rgba(15, 23, 42, 0.04);
  --chart-grid: rgba(15, 23, 42, 0.10);
  --chart-text: #1e293b;
  --chart-axis: #64748b;
  --input-bg: #f8fafc;
  --skeleton-bg: rgba(0, 0, 0, 0.04);
  --skeleton-shimmer: rgba(0, 0, 0, 0.08);
}

/* Theme Toggle Button */
.theme-toggle-btn {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--color-border);
  color: var(--color-text-subtle);
  width: 36px;
  height: 36px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  flex-shrink: 0;
  padding: 0;
}
.theme-toggle-btn:hover {
  background: var(--color-surface-hover);
  color: var(--color-text);
  border-color: var(--color-accent);
  transform: translateY(-1px);
}
html[data-theme="light"] .theme-toggle-btn {
  background: rgba(0, 0, 0, 0.04);
}
.theme-toggle-btn .icon-sun { display: none; }
.theme-toggle-btn .icon-moon { display: block; }
html[data-theme="light"] .theme-toggle-btn .icon-sun { display: block; }
html[data-theme="light"] .theme-toggle-btn .icon-moon { display: none; }
```

- [ ] **Step 2: Add Head Script and Header Markup to `index.astro`, `analyze.astro`, `library.astro`**

In `<head>` of all 3 pages:
```html
<script is:inline>
  (() => {
    const saved = localStorage.getItem('kisholens-theme');
    const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
    const theme = saved || (prefersLight ? 'light' : 'dark');
    document.documentElement.setAttribute('data-theme', theme);
  })();
</script>
```

In `.site-header` next to `.site-nav`:
```html
<button id="themeToggleBtn" class="theme-toggle-btn" type="button" aria-label="Toggle light/dark theme" title="Toggle theme">
  <svg class="icon-sun" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="5"></circle>
    <line x1="12" y1="1" x2="12" y2="3"></line>
    <line x1="12" y1="21" x2="12" y2="23"></line>
    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
    <line x1="1" y1="12" x2="3" y2="12"></line>
    <line x1="21" y1="12" x2="23" y2="12"></line>
    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
  </svg>
  <svg class="icon-moon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
  </svg>
</button>
```

In client scripts of all 3 pages:
```javascript
const themeBtn = document.getElementById('themeToggleBtn');
if (themeBtn) {
  themeBtn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('kisholens-theme', next);
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: next } }));
  });
}
```

- [ ] **Step 3: Run build to verify zero errors**

Run: `cd frontend && npm run build`
Expected: 3 page(s) built in < 200ms with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/styles/global.css frontend/src/pages/index.astro frontend/src/pages/analyze.astro frontend/src/pages/library.astro
git commit -m "feat(theme): add light mode tokens, zero-FOUT head initializer, and header toggle button"
```

---

### Task 2: Landing Page Light Mode Refinements (`index.astro`)

**Files:**
- Modify: `frontend/src/pages/index.astro:80-350`

**Interfaces:**
- Consumes: `html[data-theme="light"]` token definitions.
- Produces: Styled light mode Hero section, live prose inspector cards, demo preview cards, and CTA buttons.

- [ ] **Step 1: Add Light Theme Overrides for Landing Page Components in `index.astro`**

```css
/* In frontend/src/pages/index.astro <style> */
html[data-theme="light"] .hero__glow {
  background:
    radial-gradient(ellipse 60% 50% at 30% 40%, rgba(99, 102, 241, 0.08) 0%, transparent 70%),
    radial-gradient(ellipse 40% 40% at 75% 65%, rgba(168, 85, 247, 0.06) 0%, transparent 65%);
}

html[data-theme="light"] .card,
html[data-theme="light"] .demo-card,
html[data-theme="light"] .metric-tile,
html[data-theme="light"] .sample-inspector-card {
  background: var(--color-surface);
  border-color: var(--color-border);
  box-shadow: var(--card-shadow);
}

html[data-theme="light"] .prose-sample-box {
  background: #f1f5f9;
  border-color: rgba(0, 0, 0, 0.08);
  color: #334155;
}

html[data-theme="light"] .sample-tab-btn {
  background: rgba(0, 0, 0, 0.03);
  border-color: rgba(0, 0, 0, 0.08);
  color: #64748b;
}

html[data-theme="light"] .sample-tab-btn.active {
  background: rgba(99, 102, 241, 0.12);
  border-color: rgba(99, 102, 241, 0.35);
  color: var(--color-accent);
}
```

- [ ] **Step 2: Verify in browser with agent-browser**

Run: `agent-browser open http://localhost:4321 && agent-browser eval "document.getElementById('themeToggleBtn').click()"`
Verify: Background is `#f8fafc`, typography is crisp `#0f172a`, cards have clean borders and subtle shadow.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/index.astro
git commit -m "feat(index): refine light mode styling for landing page and live prose inspector"
```

---

### Task 3: Analyze Prose Page Light Mode & Dynamic Canvas Theming (`analyze.astro`)

**Files:**
- Modify: `frontend/src/pages/analyze.astro:300-600, 1550-1850`

**Interfaces:**
- Consumes: `themechange` event, `document.documentElement.getAttribute('data-theme')`.
- Produces: Dynamic adaptive Canvas Radar, SVG Sentiment Arc, Barcode canvas, form controls, metric cards, and doppelgänger cards.

- [ ] **Step 1: Update Canvas & SVG Chart Functions to Adapt to Current Theme**

In `drawRadar(stats, baseline)`:
```javascript
const isLight = document.documentElement.getAttribute("data-theme") === "light";
const gridColor = isLight ? "rgba(15, 23, 42, 0.10)" : "rgba(255, 255, 255, 0.10)";
const axisColor = isLight ? "rgba(15, 23, 42, 0.15)" : "rgba(255, 255, 255, 0.15)";
const textColor = isLight ? "#334155" : "#cbd5e1";
const ringFill  = isLight ? "rgba(241, 245, 249, 0.6)" : "rgba(255, 255, 255, 0.015)";
```

In `drawArc(arcData, baseline)`:
```javascript
const isLight = document.documentElement.getAttribute("data-theme") === "light";
const axisColor = isLight ? "#cbd5e1" : "rgba(255,255,255,0.15)";
const zeroColor = isLight ? "#94a3b8" : "rgba(255,255,255,0.3)";
const divColor  = isLight ? "#e2e8f0" : "rgba(255,255,255,0.07)";
const labelColor = isLight ? "#64748b" : "rgba(255,255,255,0.45)";
```

In client initialization:
```javascript
window.addEventListener('themechange', () => {
  if (currentStats) {
    renderAll();
  }
});
```

- [ ] **Step 2: Add Light Mode Component Styles for Analyze Page**

```css
html[data-theme="light"] .card,
html[data-theme="light"] .chart-card,
html[data-theme="light"] .doppelganger-card {
  background: var(--color-surface);
  border-color: var(--color-border);
  box-shadow: var(--card-shadow);
}

html[data-theme="light"] textarea,
html[data-theme="light"] input,
html[data-theme="light"] select {
  background: #f8fafc;
  border-color: rgba(0, 0, 0, 0.12);
  color: #0f172a;
}

html[data-theme="light"] .hero-chip {
  background: #ffffff;
  border-color: rgba(0, 0, 0, 0.08);
}
```

- [ ] **Step 3: Verify in browser with agent-browser**

Run: `agent-browser open http://localhost:4321/analyze && agent-browser eval "document.getElementById('themeToggleBtn').click()"`
Verify: Forms, category tab scroll tracks, Radar canvas labels, Sentiment wave, and doppelgänger cards render with high contrast in Light Mode.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/analyze.astro
git commit -m "feat(analyze): implement light mode styling and adaptive canvas chart re-rendering"
```

---

### Task 4: Library Explorer Light Mode & Dynamic Visualizations (`library.astro`)

**Files:**
- Modify: `frontend/src/pages/library.astro:1100-1400, 2900-3200`

**Interfaces:**
- Consumes: `themechange` event, `document.documentElement.getAttribute('data-theme')`.
- Produces: Light mode Library explorer with search bar, territory tabs, novel list cards, skeleton loading states, KPI badges, 3-pillar taxonomy badges, 17-genre drawer, and adaptive charts.

- [ ] **Step 1: Update `drawRadar` and `drawArc` in `library.astro` to adapt to theme**

Apply dynamic theme-aware colors for `radarChart`, `arcChart`, and `barcodeCanvas` in `library.astro`, and attach `window.addEventListener('themechange', () => { if (currentStats) renderAll(); })`.

- [ ] **Step 2: Add Light Mode Styles for Library Explorer Elements**

```css
html[data-theme="light"] .card-novel {
  background: #ffffff;
  border-color: rgba(0, 0, 0, 0.08);
}
html[data-theme="light"] .card-novel:hover {
  background: #f8fafc;
  border-color: var(--color-accent);
}
html[data-theme="light"] .card-novel.active {
  background: rgba(99, 102, 241, 0.08);
  border-color: var(--color-accent);
}
html[data-theme="light"] .territory-tab {
  background: rgba(0, 0, 0, 0.03);
  border-color: rgba(0, 0, 0, 0.07);
  color: #64748b;
}
html[data-theme="light"] .territory-tab.active {
  background: #ffffff;
  color: #0f172a;
  border-color: rgba(0, 0, 0, 0.12);
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
html[data-theme="light"] .kpi-card {
  background: #ffffff;
  border-color: rgba(0, 0, 0, 0.08);
}
html[data-theme="light"] .search-input {
  background: #ffffff;
  border-color: rgba(0, 0, 0, 0.12);
  color: #0f172a;
}
```

- [ ] **Step 3: Verify in browser with agent-browser**

Run: `agent-browser open http://localhost:4321/library && agent-browser eval "document.getElementById('themeToggleBtn').click()"`
Verify: Selecting a novel (e.g. *Noble Reincarnation* or *Tondemo Skill*) displays all charts, taxonomy pillars, and genre drawers seamlessly in Light Mode.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/library.astro
git commit -m "feat(library): refine library explorer, skeleton states, and charts for light mode"
```

---

### Task 5: Mobile Viewport Audit & Full End-to-End Test Suite

**Files:**
- Test all pages in `agent-browser`

- [ ] **Step 1: Run frontend build**

Run: `cd frontend && npm run build`
Expected: 0 errors.

- [ ] **Step 2: Run backend pytest suite**

Run: `uv run pytest tests/`
Expected: 73 passed.

- [ ] **Step 3: Audit mobile viewports in `agent-browser`**

Test at `375x667` (iPhone SE) and `600x900` across `/`, `/analyze`, and `/library` in both Dark and Light modes.
Verify:
- Theme button is easily clickable and fits in header without wrapping navigation awkwardly.
- Contrast ratio meets accessibility standards.
- Refreshing page preserves theme from `localStorage`.

- [ ] **Step 4: Commit and finalize**

```bash
git commit --allow-empty -m "chore: verify platform-wide light mode toggle and mobile responsiveness"
```
