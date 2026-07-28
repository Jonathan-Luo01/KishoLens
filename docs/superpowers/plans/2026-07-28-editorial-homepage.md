# Editorial Literary Studio Homepage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Option A (Editorial Literary Studio) on `frontend/src/pages/index.astro` — a sophisticated, non-stereotypical literary homepage with an interactive prose sample reader widget (3 tabs), 3 numbered editorial feature columns, database stats bar, and client-side tab switching script.

**Architecture:** Astro page component with embedded vanilla CSS design system, responsive grid layout, interactive client-side tab switcher for sample text excerpts and dynamic metrics, and semantic HTML5 elements.

**Tech Stack:** Astro, Vanilla CSS, Inter/Outfit Google Fonts, HTML5, Vanilla JS.

## Global Constraints

- Avoid stereotypical AI aesthetics (neon spotlights, floating badges, marketing fluff).
- Preserve 1-click revertibility guarantee (`frontend/src/pages/index.astro.bak` remains untouched).
- Ensure 100% static build validity (`npm --prefix frontend run build`).

---

### Task 1: Implement Editorial Homepage Layout & Interactive Sample Reader

**Files:**
- Modify: `frontend/src/pages/index.astro`

**Interfaces:**
- Consumes: Spec `docs/superpowers/specs/2026-07-28-editorial-homepage-design.md`
- Produces: Editorial Literary Studio homepage with interactive 3-tab prose sample reader widget

- [ ] **Step 1: Write `frontend/src/pages/index.astro`**

Implement:
1. `<header class="site-header">` (Logo, navigation links)
2. `<section class="hero">` (Clean centered typography, tagline, `Analyze Prose` & `Explore Library` buttons)
3. `<div class="sample-reader">` (3 tabs for *Xianxia Cultivation*, *Classic Gothic*, *Slice of Life* with interactive text excerpt display and live metric pills)
4. `<section class="editorial-grid">` (3 numbered feature columns: 01 Pacing Arcs, 02 Vector Archetype Engine, 03 Multi-Lingual Syntactic Metrics)
5. `<section class="stats-bar">` (2,813 Novels, 17 Genres, 4 Metrics, 3 Languages)
6. Client-side `<script>` attaching tab click event listeners to update excerpt text and metric pills.

- [ ] **Step 2: Verify static compilation with Astro build**

```bash
npm --prefix frontend run build
```
Expected: `3 page(s) built in ... ms. Complete!`

- [ ] **Step 3: Test tab switching and capture screenshot with agent-browser**

```bash
agent-browser open http://localhost:4321/ && agent-browser wait 1500 && agent-browser screenshot /Users/jonathan/.gemini/antigravity-cli/brain/5a00b53a-91af-46a3-9313-2636e2a995b2/editorial_homepage.png && agent-browser close
```
Expected: Screenshot saved showing clean editorial layout and interactive reader widget.

- [ ] **Step 4: Commit changes**

```bash
git add frontend/src/pages/index.astro
git commit -m "feat(ui): implement Editorial Literary Studio homepage with interactive sample reader widget"
```
