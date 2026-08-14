# Elevated Light Mode UI & Sliding Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Elevate the Light Mode UI with a prominent, high-visibility segmented navigation tab bar, add smooth sliding page and active pill animations between Analyze Prose and Library via Astro's native `ClientRouter`, and refine card elevations across the website.

**Architecture:** Astro View Transitions (`ClientRouter`) are integrated across all Astro pages (`index.astro`, `analyze.astro`, `library.astro`). The `.site-nav` component is styled with a prominent segmented capsule in Light Mode, using `transition:name="active-nav-pill"` for shared element active tab sliding. Dynamic page scripts are bound to `astro:page-load` for zero-flicker client-side routing.

**Tech Stack:** Astro 7.0 (`astro:transitions`), Vanilla CSS with CSS Custom Properties, JavaScript.

## Global Constraints

- Native Astro View Transitions must be used via `import { ClientRouter } from 'astro:transitions';`.
- All client-side initializers must safely bind to `document.addEventListener('astro:page-load', ...)` to maintain interactivity during client-side navigation.
- Light Mode colors must maintain high contrast (WCAG AA $\ge 4.5:1$ for text).
- `cd frontend && npm run build` must succeed with 0 errors.
- `uv run pytest tests/` must pass 73/73 tests cleanly.

---

### Task 1: Astro View Transitions Infrastructure & Shared Sliding Pill Active Indicator

**Files:**
- Modify: `frontend/src/pages/index.astro`
- Modify: `frontend/src/pages/analyze.astro`
- Modify: `frontend/src/pages/library.astro`
- Modify: `frontend/src/styles/global.css`

**Interfaces:**
- Consumes: Astro's `ClientRouter` from `astro:transitions`.
- Produces: Seamless client-side route transitions and `transition:name="active-nav-pill"` on active navigation tabs.

- [ ] **Step 1: Add ClientRouter to index.astro, analyze.astro, and library.astro**
In the frontmatter of all 3 Astro pages, import `ClientRouter` and render `<ClientRouter />` inside `<head>`:
```astro
---
import { ClientRouter } from 'astro:transitions';
---
<head>
  ...
  <ClientRouter />
</head>
```

- [ ] **Step 2: Add View Transition & Active Nav Pill CSS in global.css**
In `frontend/src/styles/global.css`, add shared transition naming and animation rules:
```css
/* ─── Astro View Transitions & Shared Nav Indicator ─────────────────── */
.site-nav a.active {
  view-transition-name: active-nav-pill;
}

::view-transition-group(active-nav-pill) {
  animation-duration: 0.3s;
  animation-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}

::view-transition-old(root),
::view-transition-new(root) {
  animation-duration: 0.22s;
  animation-timing-function: ease-in-out;
}
```

- [ ] **Step 3: Test build**
Run: `cd frontend && npm run build`
Expected: 3 page(s) built cleanly with 0 errors.

- [ ] **Step 4: Commit**
```bash
git add frontend/src/pages/index.astro frontend/src/pages/analyze.astro frontend/src/pages/library.astro frontend/src/styles/global.css
git commit -m "feat(transitions): add ClientRouter and shared active nav pill view transition"
```

---

### Task 2: High-Visibility Segmented Navigation Tab Bar & Light Mode Header Refinement

**Files:**
- Modify: `frontend/src/styles/global.css`

**Interfaces:**
- Consumes: CSS tokens `--color-bg-base`, `--color-surface`, `--color-border`, `--color-accent`.
- Produces: Elevated `.site-header`, crisp capsule `.site-nav`, vibrant `.site-nav a.active`, and tactile `#themeToggleBtn`.

- [ ] **Step 1: Refine .site-header and .site-nav styles in global.css**
Update `.site-header`, `.site-nav`, and `.site-nav a` in `frontend/src/styles/global.css`:
```css
.site-header {
  position: sticky;
  top: 0;
  z-index: 100;
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  background: rgba(2, 6, 23, 0.85);
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  padding: 0.75rem 1.5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: background-color var(--transition), border-color var(--transition);
}

[data-theme="light"] .site-header {
  background: rgba(255, 255, 255, 0.92);
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03), 0 4px 12px -2px rgba(15, 23, 42, 0.04);
}

.site-nav {
  display: flex;
  gap: 0.35rem;
  align-items: center;
  background: rgba(255, 255, 255, 0.04);
  padding: 4px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.2);
  transition: background-color var(--transition), border-color var(--transition);
}

[data-theme="light"] .site-nav {
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  box-shadow: 0 2px 8px -1px rgba(15, 23, 42, 0.07), 0 1px 3px rgba(15, 23, 42, 0.04), inset 0 1px 2px rgba(0, 0, 0, 0.03);
}

.site-nav a {
  color: rgba(255, 255, 255, 0.65);
  text-decoration: none;
  font-size: 0.85rem;
  font-weight: 500;
  padding: 6px 16px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  white-space: nowrap;
  line-height: 1.2;
  transition: color 0.15s ease, background 0.15s ease, transform 0.15s ease;
}

[data-theme="light"] .site-nav a {
  color: #475569;
}

.site-nav a:hover {
  color: #ffffff;
  background: rgba(255, 255, 255, 0.08);
}

[data-theme="light"] .site-nav a:hover {
  color: #0f172a;
  background: rgba(0, 0, 0, 0.05);
}

.site-nav a.active {
  color: #ffffff !important;
  background: rgba(124, 106, 255, 0.25) !important;
  border: 1px solid rgba(124, 106, 255, 0.4) !important;
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(124, 106, 255, 0.2);
}

[data-theme="light"] .site-nav a.active {
  color: #ffffff !important;
  background: #4f46e5 !important;
  border: 1px solid #4338ca !important;
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(79, 70, 229, 0.28);
}

.theme-toggle-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 50%;
  width: 38px;
  height: 38px;
  color: #94a3b8;
  cursor: pointer;
  transition: all 0.2s ease;
}

[data-theme="light"] .theme-toggle-btn {
  background: #ffffff;
  border: 1px solid #cbd5e1;
  color: #334155;
  box-shadow: 0 2px 6px -1px rgba(15, 23, 42, 0.08);
}

.theme-toggle-btn:hover {
  transform: translateY(-1px);
  border-color: var(--color-accent);
  color: #ffffff;
}

[data-theme="light"] .theme-toggle-btn:hover {
  border-color: #4f46e5;
  color: #4f46e5;
  background: #f8fafc;
  box-shadow: 0 4px 10px -2px rgba(79, 70, 229, 0.15);
}
```

- [ ] **Step 2: Verify build**
Run: `cd frontend && npm run build`
Expected: 0 errors.

- [ ] **Step 3: Commit**
```bash
git add frontend/src/styles/global.css
git commit -m "feat(ui): refine high-visibility segmented navigation tab bar and theme toggle in light mode"
```

---

### Task 3: ClientRouter Lifecycle Integration (`astro:page-load`) Across All Pages

**Files:**
- Modify: `frontend/src/pages/index.astro`
- Modify: `frontend/src/pages/analyze.astro`
- Modify: `frontend/src/pages/library.astro`

**Interfaces:**
- Consumes: `astro:page-load` event fired by Astro ClientRouter.
- Produces: Idempotent event listener registration across client navigations.

- [ ] **Step 1: Wrap theme toggle in idempotent initializer in global.css/pages**
In `index.astro`, `analyze.astro`, and `library.astro`, update the client `<script>` block to register listeners inside `document.addEventListener('astro:page-load', () => { ... })`.

- [ ] **Step 2: Ensure Analyze page form and redraw listeners bind cleanly on astro:page-load**
In `analyze.astro`, ensure the sample selection buttons, submit handler, category tabs, and `themechange` redraw handlers re-bind on `astro:page-load`.

- [ ] **Step 3: Ensure Library page search, territory filters, and novel cards bind cleanly on astro:page-load**
In `library.astro`, ensure `initExplorer()`, territory tabs, search input debounce, and `themechange` listeners re-bind on `astro:page-load`.

- [ ] **Step 4: Verify build**
Run: `cd frontend && npm run build`
Expected: 0 errors.

- [ ] **Step 5: Commit**
```bash
git add frontend/src/pages/index.astro frontend/src/pages/analyze.astro frontend/src/pages/library.astro
git commit -m "feat(lifecycle): attach client handlers to astro:page-load for seamless SPA navigation"
```

---

### Task 4: Light Mode Card Elevation, Surface Tokens & Visual Hierarchy

**Files:**
- Modify: `frontend/src/styles/global.css`
- Modify: `frontend/src/pages/index.astro`
- Modify: `frontend/src/pages/analyze.astro`
- Modify: `frontend/src/pages/library.astro`

**Interfaces:**
- Consumes: CSS theme classes `:global([data-theme="light"])`.
- Produces: Elevated card surfaces, crisp `#e2e8f0` borders, and rich visual depth in Light Mode.

- [ ] **Step 1: Enhance Light Mode surface tokens and card elevation in global.css**
```css
[data-theme="light"] {
  --color-surface-card: #ffffff;
  --card-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.06), 0 2px 6px -1px rgba(15, 23, 42, 0.04);
  --card-shadow-hover: 0 10px 30px -4px rgba(15, 23, 42, 0.1), 0 4px 10px -2px rgba(15, 23, 42, 0.05);
}

[data-theme="light"] .card,
[data-theme="light"] .explorer-card,
[data-theme="light"] .chart-card,
[data-theme="light"] .doppelganger-card,
[data-theme="light"] .db-stats-card {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  box-shadow: var(--card-shadow) !important;
}

[data-theme="light"] .pillar-card {
  background: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
}
```

- [ ] **Step 2: Verify build**
Run: `cd frontend && npm run build`
Expected: 0 errors.

- [ ] **Step 3: Commit**
```bash
git add frontend/src/styles/global.css frontend/src/pages/index.astro frontend/src/pages/analyze.astro frontend/src/pages/library.astro
git commit -m "feat(ui): elevate card surfaces, pillar cards, and visualizer containers in light mode"
```

---

### Task 5: End-to-End Build, Verification & Visual Navigation Testing

**Files:**
- Test: `tests/` backend suite
- Test: Visual verification via `agent-browser`

- [ ] **Step 1: Run frontend build**
Run: `cd frontend && npm run build`
Expected: 3 page(s) built cleanly with 0 errors.

- [ ] **Step 2: Run backend pytest suite**
Run: `uv run pytest tests/`
Expected: 73 passed.

- [ ] **Step 3: Visual & interaction test via agent-browser**
Run: `agent-browser` across `/`, `/analyze`, and `/library`:
- Verify tab bar capsule prominence in Light Mode.
- Verify smooth sliding capsule pill when navigating from Analyze to Library.
- Verify interactive analysis and novel card selection work cleanly on client transitions.

- [ ] **Step 4: Commit verification**
```bash
git commit --allow-empty -m "chore: verify light mode ui polish, high-visibility tab bar, and sliding navigation"
```
