# Design Spec: Editorial Literary Studio Homepage (Option A)

**Date**: 2026-07-28  
**Target File**: `frontend/src/pages/index.astro`  
**Backup File**: `frontend/src/pages/index.astro.bak` (1-click reversal guarantee)  
**Status**: Proposed / Pending User Review  

---

## 1. Executive Summary

Redesign the KishoLens homepage (`index.astro`) into an understated, sophisticated **Editorial Literary Studio** layout. Avoids generic AI landing page clichés (neon mesh spotlights, floating pill badges, fake charts) in favor of high-contrast editorial typography, a live interactive prose sample reader widget with 3 tabbed excerpts, 3 numbered capability columns, database statistics, and a minimal footer.

---

## 2. Layout & Component Architecture

### A. Minimal Header Bar (`<header class="site-header">`)
- **Background**: Semi-transparent warm obsidian (`rgba(10, 11, 16, 0.85)`) with `backdrop-filter: blur(12px)` and a subtle `1px solid rgba(255,255,255,0.07)` bottom line.
- **Left**: `Kish`**`o`**`Lens` logo mark with purple accent dot.
- **Right**: Clean navigation links (`/analyze`, `/library`, `#features`).

### B. Hero Section (`<section class="hero">`)
- **Title**: `Kish`<span class="accent">`o`</span>`Lens` (Outfit font, bold, 3.4rem, clean white).
- **Subtitle**: `Stylistic Pacing · Prose Archetypes · Structural Metrics`.
- **Description**: `Advanced narrative intelligence for web novels, light fiction, and classic literature.`
- **Action Buttons**:
  - `Analyze Prose →` (Primary dark purple glass button with subtle hover highlight).
  - `Explore Library` (Secondary border outline button).

### C. Interactive Sample Reader Widget (`<div class="sample-reader">`)
- **3 Tab Buttons**:
  1. `Xianxia Cultivation` (Sample text: *"The spiritual energy of the golden core surged through his meridians..."*)
  2. `Classic Gothic Literature` (Sample text: *"It was a dreary night of November that I beheld the accomplishment of my labours..."*)
  3. `Cozy Slice of Life` (Sample text: *"The morning sun filtered gently into the small tea shop as the kettle began to simmer..."*)
- **Sample Text Box**: Clean monospaced/serif excerpt container displaying the selected passage.
- **Dynamic Metric Pills**: Live metric chips updating instantaneously when tabs are clicked:
  - `Pacing Arc`: `Ki 起 → Shō 承 → Ten 転 → Ketsu 结`
  - `Dialogue Ratio`: `34.2%`
  - `Lexical Richness (TTR)`: `0.84`
  - `Territory Match`: `Web Novel Territory` or `Classic Literature Territory`

### D. 3 Editorial Feature Columns (`<section class="editorial-grid">`)
- **01 / Narrative Pacing Arcs**: 4-quantile Kishōtenketsu sentiment tracking for introduction, development, twist, and reconciliation.
- **02 / Vector Archetype Engine**: 384-dimensional embedding cosine similarity matching against 2,800+ novels & 17 canonical genres.
- **03 / Multi-Lingual Syntactic Metrics**: Deep dependency tree depth, TTR vocabulary diversity, and dialogue ratio across EN, JA, and ZH.

### E. Database Stats Banner (`<section class="stats-bar">`)
- `2,813` Ingested Novels &bull; `17` Canonical Genres &bull; `4` Metric Dimensions &bull; `3` Languages (EN/JA/ZH)

### F. Minimal Footer (`<footer class="site-footer">`)
- Single-line editorial footer with stack credits (`Astro`, `FastAPI`, `PyTorch`, `spaCy`).

---

## 3. Visual & Styling Principles

- **Background Color**: Warm obsidian `#0a0b10` / `#111219`
- **Card Background**: `rgba(20, 22, 32, 0.6)` with `border: 1px solid rgba(255, 255, 255, 0.08)`
- **Accent Color**: Deep violet `#8b5cf6` and indigo `#6366f1` (subtle, non-neon)
- **Typography**: Heading: `Outfit`, Body/UI: `Inter`, Excerpts: `JetBrains Mono` / `Inter`
- **Aesthetic Goal**: Refined literary elegance, high clarity, non-stereotypical AI design.
