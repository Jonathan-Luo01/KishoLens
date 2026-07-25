# Enhanced Visual Metrics Dashboard UI Design

**Date:** 2026-07-24  
**Status:** Approved  
**Target File:** `frontend/src/pages/library.astro`

## Executive Summary
Transform the plaintext metrics card in the KishoLens library into an interactive, visual dashboard featuring hero metric highlights, category filtering, percentile progress meters, benchmark comparison chips against corpus norms, and hover tooltips.

---

## Architecture & Visual Layout

### 1. Component Hierarchy
The metrics panel (`statsContent`) will render three core sections:
1. **Hero Summary Banner**:
   - Primary Archetype Match & Territory badge (with confidence %)
   - Word Count & Size classification badge
   - Lexical Richness (TTR) card
   - Dialogue Density badge
2. **Category Filter Bar**:
   - Interactive pill buttons: `All` | `Structure` | `Prose & Style` | `Theme & Emotion` | `Pacing & Narrative`
3. **Grouped Visual Metric Grid**:
   - 4 categorized sections with distinct color-coded theme accents:
     - **Structure**: Cyan (`#38bdf8`) — Word Count, Sentence Count, Avg Sentence Length, Sentences/Paragraph, Dep Tree Depth
     - **Prose & Style**: Purple (`#a78bfa`) — Dialogue Ratio, Lexical Density (TTR), Adjective Ratio, Verb/Particle Ratio
     - **Theme & Emotion**: Pink (`#f472b6`) — VADER/Lexicon Sentiment, Thematic Explicitness, Visceral Emotion, World Grounding
     - **Pacing & Narrative**: Emerald (`#34d399`) — Temporal Complexity, Time Shifts, Subplot Diversity

---

## Detailed Card Specifications

### Metric Card Anatomy
Each individual metric item contains:
- **Header**: Metric label + info icon trigger for explanation tooltip.
- **Primary Value**: Formatted numerical readout (`3.51`, `68.2%`, etc.).
- **Percentile Meter Bar**: 
  - Smooth 0-100% horizontal progress bar.
  - Filled with a theme-matching gradient glow.
- **Corpus Benchmark Chip**:
  - Compares metric value to the active baseline (`Web Novel` or `Classic Lit`).
  - Formatted as `+18% vs Web Novel` or `-5% vs Classic Lit` (or qualitative descriptor pill `Balanced`, `High Syntax`).

---

## Interactive Features & Micro-Animations
- **Category Filter Toggle**: Clicking a category tab filters visible sections with a subtle fade animation.
- **Hover Tooltips**: Positioned tooltips describing technical NLP terms (e.g. *Dependency Tree Depth: Measures syntactic sentence complexity*).
- **Responsive Grid**: CSS grid adapting smoothly from 1 column on mobile to 2 columns on desktop.

---

## Data Integration & Fallbacks
- Uses existing `stats` object from `/api/novels/{id}/stats` API response:
  - Percentile scores derived from `stats.normalized_radar`
  - Baselines derived from `stats.baselines`
- Graceful `N/A` handling for missing or uncalculated language features.
