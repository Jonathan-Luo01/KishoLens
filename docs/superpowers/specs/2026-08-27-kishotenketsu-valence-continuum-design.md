# Design Spec: Kishōtenketsu Narrative Trajectory & Valence Spectrum

## 1. Overview
This design replaces the redundant Ki/Shō/Ten/Ketsu stat cards under the **Kishōtenketsu Sentiment Arc** chart with a comprehensive literary diagnostic panel that eliminates whitespace, enhances narrative interpretability, and achieves vertical height symmetry with the adjacent Archetype Radar card.

---

## 2. Component Architecture

The `#arcBreakdownContainer` inside the Kishōtenketsu Sentiment Arc card (`.chart-card`) consists of three integrated sections:

```
┌────────────────────────────────────────────────────────────┐
│ 1. Narrative Trajectory Card                               │
│    [Dynamic Reversal Arc ∿]                                │
│    "Begins in grounded exposition (-0.39), builds rising  │
│     momentum through development (+0.23), peaks at Ten..." │
├────────────────────────────────────────────────────────────┤
│ 2. Act Transition Deltas                                   │
│    [Ki ➔ Shō: +0.62 Rising] [Shō ➔ Ten: +0.30 Climax]     │
│    [Ten ➔ Ketsu: -0.04 Stable]                             │
├────────────────────────────────────────────────────────────┤
│ 3. Continuous Valence Spectrum Bar                         │
│    [ -1.0 ══════════════ 0.0 ══════════════ +1.0 ]         │
│    ◀ Peril / Tension   Neutral Baseline   Triumph / Hope ▶ │
└────────────────────────────────────────────────────────────┘
```

### A. Section 1: Narrative Trajectory Summary Card
- **Trajectory Badge**: Dynamic literary arc category badge (`Dynamic Reversal Arc`, `Heroic Ascent Arc`, `Tragic Descent Arc`, `High-Volatility Twist Arc`, `Balanced Narrative Arc`).
- **Editorial Trajectory Explanation**: High-level narrative description connecting initial exposition, rising action, climax turn, and ending resolution.

### B. Section 2: 3-Stage Act Transition Deltas
- Computes emotional velocity shifts between consecutive acts:
  1. `Ki ➔ Shō` (Δ1 = Shō - Ki)
  2. `Shō ➔ Ten` (Δ2 = Ten - Shō)
  3. `Ten ➔ Ketsu` (Δ3 = Ketsu - Ten)
- Each pill displays:
  - Act transition label with Lucide `ArrowRight` SVG (`Ki ➔ Shō`)
  - Delta value formatted with sign (`+0.62` / `-0.41`)
  - Movement descriptor (*Rising Friction*, *Climactic Surge*, *Sudden Reversal*, *Cathartic Resolution*, *Lingering Cliffhanger*)
  - Color-coded trend indicator (emerald for positive delta, rose for negative delta, slate for stable).

### C. Section 3: Continuous Valence Spectrum Bar
- A full-width horizontal spectrum bar with a continuous gradient:
  - Left (`-1.0` to `-0.1`): Deep Rose/Coral (`#f43f5e` to `#f97316`) representing **High Stakes, Peril, Conflict & Melancholy**.
  - Center (`0.0`): Neutral Slate (`#64748b`) representing **Neutral Exposition, Worldbuilding & Procedural Dialogue**.
  - Right (`+0.1` to `+1.0`): Sky Blue/Emerald (`#38bdf8` to `#10b981`) representing **Triumph, Romantic Catharsis, Levity & Hope**.
- Annotated baseline ticks and readable milestone labels.

---

## 3. Files to Modify

1. **`frontend/src/styles/global.css`**:
   - Add `.arc-act-transitions-row`, `.transition-pill`, `.valence-spectrum-card`, `.valence-spectrum-bar`, and `.valence-legend-labels` styles with full dark/light theme support.
2. **`frontend/src/pages/library.astro`**:
   - Update `#arcBreakdownContainer` markup and `updateArcBreakdown()` function.
3. **`frontend/src/pages/analyze.astro`**:
   - Update `#arcBreakdownContainer` markup and `updateArcBreakdown()` function.

---

## 4. Verification Plan

1. **Build Verification**: `npm --prefix frontend run build` must compile with 0 errors.
2. **Browser Verification with `agent-browser`**:
   - Test `http://localhost:4321/library` in light and dark mode with multiple novel archetypes (*Noble Reincarnation*, *Of Course I'll Claim Palimony*, *I Swear I Won't Bother You Again*).
   - Test `http://localhost:4321/analyze` with live user prose analysis.
   - Verify that the card heights match symmetrically and all elements fit with zero overflow.
