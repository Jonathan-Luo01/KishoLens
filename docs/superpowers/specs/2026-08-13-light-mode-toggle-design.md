# Design Specification: Platform-Wide Light Mode Toggle & Adaptive Visualizations

- **Feature**: Universal Light/Dark Theme Switcher with Zero-FOUT and Adaptive Canvas/SVG Charts
- **Date**: 2026-08-13
- **Author**: Antigravity & Pair Programming Partner
- **Target Surface**: `frontend/src/styles/global.css`, `frontend/src/pages/index.astro`, `frontend/src/pages/analyze.astro`, `frontend/src/pages/library.astro`

---

## 1. Executive Summary

KishoLens is currently styled exclusively in a dark obsidian glassmorphic aesthetic. This feature introduces a comprehensive, system-aware **Light Mode Toggle** across all pages of the application (`/`, `/analyze`, `/library`). The implementation delivers:
1. **Semantic CSS Token System**: Comprehensive mapping of background, surface, card, border, text contrast, and interactive hover states under `html[data-theme="light"]`.
2. **Zero-FOUT (Flash of Unstyled Theme) Script**: Lightweight inline script in `<head>` that instantly applies theme preferences from `localStorage` or `prefers-color-scheme` prior to DOM rendering.
3. **Adaptive Canvas & SVG Chart Rendering**: The 8D Archetype Radar (HTML5 Canvas), Kishōtenketsu Sentiment Arc (SVG), and Pacing Barcode (Canvas) automatically adapt grid lines, text labels, and stroke contrast to the active theme, re-rendering dynamically on theme toggle.
4. **Header Integration & Mobile Responsiveness**: A sleek Sun/Moon animated icon toggle seamlessly integrated into `.site-header` on desktop and mobile viewports ($\le 375\text{px}$).

---

## 2. Token Architecture & Color Palettes

### 2.1 CSS Variables & Semantic Mapping

```css
/* ─── Default (Dark Theme) ─── */
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

/* ─── Light Theme Overrides ─── */
html[data-theme="light"], :root[data-theme="light"] {
  --color-bg: #f8fafc; /* Crisp Slate-50 */
  --color-surface: #ffffff;
  --color-surface-card: #ffffff;
  --color-surface-hover: #f1f5f9;
  --color-border: rgba(0, 0, 0, 0.09);
  --color-border-subtle: rgba(0, 0, 0, 0.05);
  --color-accent: #6366f1;
  --color-accent-2: #a855f7;
  --color-accent-cyan: #0284c7;
  --color-accent-emerald: #059669;
  --color-text: #0f172a; /* Deep Slate-900 */
  --color-text-subtle: #475569; /* Slate-600 */
  --color-muted: #64748b; /* Slate-500 */
  --header-bg: rgba(255, 255, 255, 0.88);
  --card-shadow: 0 4px 24px -2px rgba(15, 23, 42, 0.06), 0 2px 6px -1px rgba(15, 23, 42, 0.04);
  --chart-grid: rgba(15, 23, 42, 0.10);
  --chart-text: #1e293b;
  --chart-axis: #64748b;
  --input-bg: #f8fafc;
  --skeleton-bg: rgba(0, 0, 0, 0.04);
  --skeleton-shimmer: rgba(0, 0, 0, 0.08);
}
```

---

## 3. Component & UI Architecture

### 3.1 Zero-FOUT Head Initializer
Placed directly inside `<head>` on every page before stylesheets execute:
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

### 3.2 Header Theme Toggle Button
A clickable button in `.site-header` containing Sun and Moon SVGs:
```html
<button id="themeToggleBtn" class="theme-toggle-btn" type="button" aria-label="Toggle theme" title="Toggle dark/light theme">
  <svg class="icon-sun" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
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
  <svg class="icon-moon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
  </svg>
</button>
```

---

## 4. Adaptive Chart Theming Engine

### 4.1 Radar Chart Dynamic Canvas Colors
In `drawRadar(currentStats, activeBaseline)`:
```javascript
const isLight = document.documentElement.getAttribute("data-theme") === "light";
const gridColor = isLight ? "rgba(15, 23, 42, 0.10)" : "rgba(255, 255, 255, 0.10)";
const axisColor = isLight ? "rgba(15, 23, 42, 0.15)" : "rgba(255, 255, 255, 0.15)";
const textColor = isLight ? "#334155" : "#cbd5e1";
const ringFill  = isLight ? "rgba(241, 245, 249, 0.6)" : "rgba(255, 255, 255, 0.015)";
```

### 4.2 Kishōtenketsu Sentiment Arc (SVG)
In `drawArc(arcData, baseline)`:
- Axis line: `stroke="${isLight ? '#cbd5e1' : 'rgba(255,255,255,0.15)'}"`
- Zero baseline: `stroke="${isLight ? '#94a3b8' : 'rgba(255,255,255,0.3)'}"`
- Act vertical divider: `stroke="${isLight ? '#e2e8f0' : 'rgba(255,255,255,0.07)'}"`
- Act labels text: `fill="${isLight ? '#64748b' : 'rgba(255,255,255,0.45)'}"`

### 4.3 Barcode & Visualization Listeners
Whenever the toggle button is clicked:
1. `document.documentElement.setAttribute('data-theme', newTheme)`
2. `localStorage.setItem('kisholens-theme', newTheme)`
3. Dispatches `window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: newTheme } }))`
4. All active charts immediately re-render with matching palettes.

---

## 5. Verification Plan

1. **Astro Build Integrity**: `cd frontend && npm run build` (0 compilation or bundling errors).
2. **Desktop Visual Test**: Verify Index, Analyze, and Library pages in both Dark Mode and Light Mode with high contrast and proper card elevation.
3. **Mobile Viewport Test**: Verify headers, navigation pill bars, theme toggle button, and cards on mobile viewports ($375\text{px}$ and $600\text{px}$) in `agent-browser`.
4. **Theme Persistence Test**: Verify that refreshing or navigating between pages preserves the chosen theme without flash.
