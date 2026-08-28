# Kishōtenketsu Narrative Trajectory & Valence Spectrum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the redundant Ki/Shō/Ten/Ketsu stat cards under the Kishōtenketsu Sentiment Arc with a dynamic Narrative Trajectory card, a 3-stage Act Transition Deltas row, and a Continuous Valence Spectrum Bar.

**Architecture:** Update `frontend/src/styles/global.css` with responsive layout tokens and glassmorphism styling for the valence spectrum and transition pills; update markup and `updateArcBreakdown()` in both `frontend/src/pages/library.astro` and `frontend/src/pages/analyze.astro`.

**Tech Stack:** Astro, Vanilla JavaScript, CSS3 (Container Queries & Flex/Grid), Lucide SVG icons.

## Global Constraints
- Do NOT commit or push to remote until explicitly requested by the user.
- Ensure 100% theme support for both Light (`[data-theme="light"]`) and Dark themes.
- Symmetrically balance card height with the Archetype Radar on the left.
- `npm --prefix frontend run build` must compile with 0 errors.

---

### Task 1: Design Tokens & CSS Styling in `global.css`

**Files:**
- Modify: `frontend/src/styles/global.css`

**Interfaces:**
- Produces CSS classes: `.arc-act-transitions-row`, `.transition-pill`, `.transition-delta`, `.transition-desc`, `.valence-spectrum-card`, `.valence-spectrum-bar`, `.valence-spectrum-ticks`, `.valence-legend-labels`.

- [ ] **Step 1: Add CSS styling rules for Act Transitions Row and Continuous Valence Spectrum Bar**

```css
/* ─── Kishōtenketsu Act Transitions Row ───────────────────────── */
.arc-act-transitions-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.5rem;
  width: 100%;
  box-sizing: border-box;
}

@media (max-width: 600px) {
  .arc-act-transitions-row {
    grid-template-columns: minmax(0, 1fr);
  }
}

.transition-pill {
  background: rgba(255, 255, 255, 0.025);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 0.45rem 0.6rem;
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  min-width: 0;
  box-sizing: border-box;
  transition: border-color var(--transition);
}

.transition-pill-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.3rem;
}

.transition-pill-label {
  font-size: 0.68rem;
  font-weight: 700;
  color: var(--color-text);
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.transition-delta {
  font-size: 0.72rem;
  font-weight: 700;
  font-family: var(--font-display);
}

.transition-delta.positive { color: #10b981; }
.transition-delta.negative { color: #f43f5e; }
.transition-delta.neutral { color: var(--color-muted); }

.transition-desc {
  font-size: 0.64rem;
  color: var(--color-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ─── Continuous Valence Spectrum Bar ────────────────────────── */
.valence-spectrum-card {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--color-border);
  border-radius: 10px;
  padding: 0.65rem 0.85rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  box-sizing: border-box;
}

[data-theme="light"] .valence-spectrum-card {
  background: rgba(255, 255, 255, 0.9);
  border-color: rgba(203, 213, 225, 0.8);
}

.valence-spectrum-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.4rem;
}

.valence-spectrum-title {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--color-muted);
}

.valence-spectrum-range {
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--color-text-subtle);
}

.valence-spectrum-bar-wrap {
  position: relative;
  width: 100%;
  padding: 0.2rem 0;
}

.valence-spectrum-bar {
  height: 8px;
  width: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #f43f5e 0%, #f97316 25%, #64748b 50%, #38bdf8 75%, #10b981 100%);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.2);
}

.valence-spectrum-ticks {
  display: flex;
  justify-content: space-between;
  margin-top: 0.25rem;
  font-size: 0.64rem;
  font-weight: 600;
  color: var(--color-muted);
}

.valence-legend-labels {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
  font-size: 0.64rem;
  line-height: 1.3;
  margin-top: 0.15rem;
}

.valence-legend-col {
  display: flex;
  flex-direction: column;
}

.valence-legend-col.left { text-align: left; color: #f87171; }
.valence-legend-col.center { text-align: center; color: var(--color-muted); }
.valence-legend-col.right { text-align: right; color: #34d399; }
```

- [ ] **Step 2: Verify frontend compilation**
Run: `npm --prefix frontend run build`
Expected: PASS with 0 errors.

---

### Task 2: Implement in `frontend/src/pages/library.astro`

**Files:**
- Modify: `frontend/src/pages/library.astro`

- [ ] **Step 1: Replace `#arcBreakdownContainer` markup**
Embed Section 1 (Narrative Trajectory), Section 2 (3-Stage Act Transition Deltas with Lucide arrow SVGs), and Section 3 (Continuous Valence Spectrum Bar).

- [ ] **Step 2: Update `updateArcBreakdown()` JavaScript function**
Calculate `delta1 = vSho - vKi`, `delta2 = vTen - vSho`, `delta3 = vKetsu - vTen`.
Update `#transDelta1`, `#transDesc1`, `#transDelta2`, `#transDesc2`, `#transDelta3`, `#transDesc3`, and narrative trajectory badge/desc.

- [ ] **Step 3: Verify build**
Run: `npm --prefix frontend run build`
Expected: PASS with 0 errors.

---

### Task 3: Implement in `frontend/src/pages/analyze.astro`

**Files:**
- Modify: `frontend/src/pages/analyze.astro`

- [ ] **Step 1: Update `#arcBreakdownContainer` markup and `updateArcBreakdown()` in `analyze.astro`**
Mirror the exact same structure as `library.astro`.

- [ ] **Step 2: Verify build**
Run: `npm --prefix frontend run build`
Expected: PASS with 0 errors.

---

### Task 4: End-to-End Verification with `agent-browser`

**Files:**
- Test via browser: `http://localhost:4321/library` & `http://localhost:4321/analyze`

- [ ] **Step 1: Visual check on `/library` in Light and Dark mode**
Open browser, select novels with contrasting arcs (*Noble Reincarnation*, *Of Course I'll Claim Palimony*, *I Swear I Won't Bother You Again*).
Capture screenshots of the uncompressed chart, transition pills, and valence continuum bar.

- [ ] **Step 2: Visual check on `/analyze`**
Run live prose analysis and verify that the valence continuum bar and transition pills render identically.

- [ ] **Step 3: Confirm card height symmetry**
Verify zero blank white space and zero element clipping.
