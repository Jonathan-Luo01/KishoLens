# Design Spec: Main Page UI Redesign (Option 1)

**Date**: 2026-07-28  
**Target File**: `frontend/src/pages/index.astro`  
**Backup Location**: `frontend/src/pages/index.astro.bak` (for easy 1-click reversal)  
**Status**: Proposed / Pending User Review  

---

## 1. Executive Summary

Redesign the KishoLens homepage (`index.astro`) into a high-impact, state-of-the-art AI application landing page featuring a glassmorphism navbar, ambient background glow, interactive 4-act Kishōtenketsu pacing hero widget, 4-card feature showcase grid, live database statistics banner, and CTA section.

---

## 2. Layout & Component Architecture

### A. Fixed Top Navbar (`<header class="navbar">`)
- **Background**: `rgba(10, 10, 20, 0.8)` with `backdrop-filter: blur(16px)` and subtle bottom border `rgba(255, 255, 255, 0.08)`.
- **Left**: Glowing brand logo `KishoLens` with purple-pink dot accent.
- **Center**: Links to `#features`, `/analyze`, `/library`.
- **Right**: Action button `Analyze Prose →` with hover glow.

### B. Hero Section (`<section class="hero-section">`)
- **Backdrop**: Layered radial gradient spotlights (`#a78bfa` at 15% opacity, `#38bdf8` at 10% opacity) and animated subtle grid.
- **Pill Badge**: `✨ AI-Powered Prose & Pacing Analytics • 2,800+ Novels`.
- **Headline**: `Uncover the Narrative DNA of Web Novels & Classic Literature`.
- **Subtitle**: `Quantify Kishōtenketsu pacing curves, syntactic complexity, dialogue ratios, and prose archetypes across 17 canonical genres and 3 languages.`
- **Action Buttons**:
  - `Analyze Prose` (Primary purple gradient button with hover elevation).
  - `Explore Library` (Secondary glass button with border highlight).

### C. Interactive Hero Demo Card (`<div class="hero-demo-card">`)
- **Card Header**: Live sample indicator (`Live Preview: Chapter 1 Sentiment Arc`).
- **Interactive SVG Chart**: 4-Act Kishōtenketsu curve (Ki 起, Shō 承, Ten 転, Ketsu 结) with animated gradient stroke and glowing node markers.
- **Live Archetype Chip**: `Matched Territory: Web Novel Territory | Archetype: Progression / Xianxia`.
- **Metric Highlights**: Dialogue Ratio (`34.2%`), TTR (`0.75`), Dep Tree Depth (`4.8`).

### D. Capabilities & Features Grid (`<section class="features-section">`)
- **Grid Layout**: 2x2 responsive CSS grid of glassmorphism cards:
  1. **Kishōtenketsu Pacing Arc**: 4-quantile emotional arc analysis comparing text against classic literature and web novel baselines.
  2. **Prose Archetype & Territory Engine**: Cosine similarity matching across 384-dim embeddings against 2,800+ ingested novels.
  3. **17 Canonical Genres**: Multi-label classification for Xianxia, Cultivation, Isekai, Romance, Historical, Sci-Fi, Slice of Life, etc.
  4. **Multi-Lingual Deep NLP**: Native syntactic parsing and feature extraction for English (spaCy), Japanese (SudachiPy), and Chinese (spaCy/HanLP).

### E. Database Statistics Banner (`<section class="stats-banner">`)
- **Stat Item 1**: `2,813+` Ingested Novels
- **Stat Item 2**: `17` Canonical Genres
- **Stat Item 3**: `4` Metric Categories (Structure, Prose, Theme, Pacing)
- **Stat Item 4**: `3` Supported Languages (EN, JA, ZH)

### F. Call to Action & Footer (`<footer class="footer">`)
- **CTA Box**: `Ready to Analyze Your Prose?` with quick direct action button.
- **Footer**: Brand mark, technology stack credits (`Astro`, `FastAPI`, `PyTorch`, `spaCy`), and links.

---

## 3. Revertibility Plan

To ensure the design can be reverted instantly at any time upon user request:
1. Save the existing `index.astro` file to `frontend/src/pages/index.astro.bak`.
2. Commit the original backup file to git.
3. If the user ever requests to revert, a single step restores `index.astro` from `index.astro.bak`.

---

## 4. Visual Styling System

- **Color Palette**:
  - Background: `#09090e` / `#0f0f18`
  - Cards: `rgba(20, 20, 35, 0.6)` with `border: 1px solid rgba(255, 255, 255, 0.08)`
  - Primary Accent: `#a78bfa` (Purple) & `#ec4899` (Pink)
  - Secondary Accent: `#38bdf8` (Cyan) & `#34d399` (Emerald)
  - Text: Primary `#f8fafc`, Secondary `#94a3b8`
- **Typography**: `Outfit` for headings, `Inter` for body & UI elements.
- **Animations**: CSS smooth transitions on hover (`transform: translateY(-4px)`, `box-shadow` glow).
