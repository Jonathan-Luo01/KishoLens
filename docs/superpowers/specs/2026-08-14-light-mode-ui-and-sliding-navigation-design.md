# Elevated Light Mode UI & Sliding Navigation Design Specification

## Overview & Goals
This specification defines the visual and architectural improvements for KishoLens's **Light Mode UI**, focusing on:
1. **High-Visibility Segmented Top Navigation Bar**: A prominent, elevated capsule tab bar in the header with crisp borders, subtle elevation shadows, and a sliding active pill indicator.
2. **Astro View Transitions & Sliding Route Animation**: Seamless, zero-flicker animated page transitions between `/analyze` and `/library` (and `/` homepage) using Astro's native `ClientRouter` / View Transitions API.
3. **Elevated Card & Panel Styling in Light Mode**: Refined card elevation, tactile borders, clear hierarchy, and smooth micro-interactions across the Landing, Analyze, and Library pages.

---

## 1. High-Visibility Segmented Top Navigation Bar

### 1.1 Visual Appearance & Contrast
- **Container (`.site-nav`)**:
  - Light Mode: `#ffffff` or semi-translucent `#f1f5f9` capsule with `border: 1px solid #cbd5e1`, `box-shadow: 0 2px 8px -1px rgba(15, 23, 42, 0.08), 0 1px 3px rgba(15, 23, 42, 0.04)`, `padding: 4px`, `border-radius: 999px`.
  - Dark Mode: `rgba(255, 255, 255, 0.05)` with `border: 1px solid rgba(255, 255, 255, 0.1)`.
- **Navigation Links (`.site-nav a`)**:
  - Inactive Links: `#475569` text, medium weight (500), smooth color transition on hover to `#0f172a` with subtle background tint (`rgba(0, 0, 0, 0.04)`).
  - Active Link (`.site-nav a.active`):
    - Light Mode: `#ffffff` text on vibrant Indigo/Sky accent pill (`background: #4f46e5; border: 1px solid #4338ca; box-shadow: 0 2px 6px rgba(79, 70, 229, 0.25)`).
    - `transition:name="active-nav-pill"` for shared element morphing/sliding across page transitions.
- **Theme Toggle Button (`#themeToggleBtn`)**:
  - Light Mode: `#ffffff` background with `border: 1px solid #cbd5e1`, `color: #334155`, subtle shadow `0 2px 6px rgba(15, 23, 42, 0.06)`, and micro-hover scale `transform: translateY(-1px)`.

---

## 2. Astro View Transitions & Page Sliding Animation

### 2.1 Native Route Transitions (`ClientRouter`)
- Use `import { ClientRouter } from 'astro:transitions';` in `index.astro`, `analyze.astro`, and `library.astro` in the `<head>` element.
- Define View Transition CSS animations in `global.css`:
  - Active Tab Sliding: Using `transition:name="active-nav-pill"` on `.site-nav a.active`, the active pill indicator smoothly glides across tabs during navigation.
  - Page Slide Animation: Smooth cross-fade with directional slide for main content (`transition:animate="slide"` or custom keyframe animation `.page-content`).

### 2.2 Lifecycle & Script Re-attachment (`astro:page-load`)
Because View Transitions maintain the SPA state without full reloads:
- Wrap frontend initializer scripts in `document.addEventListener('astro:page-load', () => { ... })`:
  - Theme toggle event listeners and persistent theme sync.
  - Analyze page sample buttons, submit form handler, and dynamic chart redrawing.
  - Library page search input, territory tabs, genre filters, and novel cards.
  - Global cleanup to prevent duplicate event listeners.

---

## 3. Light Mode Visual Hierarchy & Card Elevation

### 3.1 Card & Panel Elevation
- **Surface Elevation Tokens**:
  - Base Card (`.card`, `.explorer-card`, `.chart-card`, `.doppelganger-card`): `background: #ffffff`, `border: 1px solid #e2e8f0`, `box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.06), 0 2px 6px -1px rgba(15, 23, 42, 0.04)`.
  - Hover State: `border-color: #cbd5e1`, `box-shadow: 0 8px 30px -4px rgba(15, 23, 42, 0.09)`.
- **Section Headers & Accents**:
  - Distinct badge tags, high-contrast pillar cards (`.pillar-card` with crisp `#cbd5e1` borders and `#0f172a` headings), and elevated KPI boxes.

---

## 4. Verification & Testing

1. **Automated Verification**:
   - `cd frontend && npm run build` static compilation passing with 0 errors.
   - `uv run pytest tests/` full test suite passing (73/73 tests).
2. **Visual & Interaction Verification (agent-browser)**:
   - Verify tab bar visibility, contrast, and active indicator in Light Mode.
   - Click between "Analyze Prose" and "Library" to verify smooth View Transition sliding animation and zero FOUT / script errors.
   - Confirm charts, search, and theme toggle remain fully functional after repeated navigation.
