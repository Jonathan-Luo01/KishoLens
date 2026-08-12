# Metrics Card UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the raw plaintext statistics list in `frontend/src/pages/library.astro` into an interactive visual metrics dashboard featuring a hero summary row, category filter tabs, visual progress bars, corpus benchmark comparison chips, and hover tooltips.

**Architecture:** Refactor `buildStatItem()` in `library.astro` to render structured metric cards enriched with metadata (category, tooltips, normalized percentile progress, baseline comparison). Add CSS rules for glassmorphism metric cards, category filter tabs, micro-progress bars, and benchmark chips.

**Tech Stack:** Astro, HTML5, Vanilla CSS, JS (ES6+)

## Global Constraints
- Target File: `frontend/src/pages/library.astro`
- Design Specification: `docs/superpowers/specs/2026-07-24-metrics-card-ui-design.md`
- Preserve all existing API bindings (`stats.en_*`, `stats.ja_*`, `stats.zh_*`, `stats.normalized_radar`, `stats.baselines`, `stats.archetype_match`).

---

### Task 1: Add Enhanced Metrics Styles & CSS Components

**Files:**
- Modify: `frontend/src/pages/library.astro:550-620` (inside `<style>`)

**Interfaces:**
- Produces: CSS classes for `.metrics-hero`, `.category-tabs`, `.metric-card`, `.percentile-bar`, `.benchmark-chip`, `.metric-tooltip`.

- [ ] **Step 1: Inspect existing `<style>` block in `library.astro`**

View lines 550 to 610 in `frontend/src/pages/library.astro` to locate the current `.stat-grid`, `.stat-item`, `.stat-label`, and `.stat-value` rules.

- [ ] **Step 2: Add CSS rules for the new dashboard components**

Add the following CSS rules to `<style>` in `frontend/src/pages/library.astro`:

```css
/* Enhanced Metrics Dashboard Component Styles */
.metrics-dashboard {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.metrics-hero {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.6rem;
  background: rgba(124, 106, 255, 0.05);
  border: 1px solid rgba(124, 106, 255, 0.2);
  border-radius: 10px;
  padding: 0.85rem;
}

.hero-chip {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.hero-chip-label {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-muted);
}

.hero-chip-val {
  font-family: var(--font-display);
  font-size: 1.15rem;
  font-weight: 700;
  color: #ffffff;
}

.metrics-category-tabs {
  display: flex;
  gap: 0.4rem;
  overflow-x: auto;
  padding-bottom: 0.25rem;
}

.category-tab {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: var(--color-muted);
  border-radius: 20px;
  padding: 0.3rem 0.75rem;
  font-size: 0.75rem;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.2s ease;
}

.category-tab:hover, .category-tab.active {
  background: rgba(124, 106, 255, 0.18);
  border-color: rgba(124, 106, 255, 0.5);
  color: #ffffff;
}

.metric-category-section {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.category-section-title {
  font-family: var(--font-display);
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin-top: 0.4rem;
}

.category-section-title.cat-structure { color: #38bdf8; }
.category-section-title.cat-prose     { color: #a78bfa; }
.category-section-title.cat-theme     { color: #f472b6; }
.category-section-title.cat-pacing    { color: #34d399; }

.metric-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 0.6rem;
}

.metric-card {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 8px;
  padding: 0.65rem 0.8rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  position: relative;
  transition: border-color 0.2s ease, transform 0.2s ease;
}

.metric-card:hover {
  border-color: rgba(124, 106, 255, 0.3);
  transform: translateY(-1px);
}

.metric-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.metric-card-label {
  font-size: 0.7rem;
  font-weight: 500;
  color: #cbd5e1;
}

.metric-info-icon {
  font-size: 0.7rem;
  color: var(--color-muted);
  cursor: help;
  opacity: 0.6;
  transition: opacity 0.15s;
}
.metric-info-icon:hover { opacity: 1.0; }

.metric-val-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.4rem;
}

.metric-primary-val {
  font-family: var(--font-display);
  font-size: 1.15rem;
  font-weight: 700;
  color: #f8fafc;
}

.percentile-bar-bg {
  width: 100%;
  height: 4px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 2px;
  overflow: hidden;
}

.percentile-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.4s ease-out;
}

.bar-fill-structure { background: linear-gradient(90deg, #0284c7, #38bdf8); }
.bar-fill-prose     { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
.bar-fill-theme     { background: linear-gradient(90deg, #db2777, #f472b6); }
.bar-fill-pacing    { background: linear-gradient(90deg, #059669, #34d399); }

.benchmark-chip {
  display: inline-flex;
  align-items: center;
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
  font-size: 0.65rem;
  font-weight: 600;
  width: fit-content;
}

.benchmark-chip.higher {
  background: rgba(34, 197, 94, 0.15);
  color: #4ade80;
  border: 1px solid rgba(34, 197, 94, 0.3);
}

.benchmark-chip.lower {
  background: rgba(239, 68, 68, 0.15);
  color: #fca5a5;
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.benchmark-chip.neutral {
  background: rgba(148, 163, 184, 0.12);
  color: #cbd5e1;
  border: 1px solid rgba(148, 163, 184, 0.2);
}
```

- [ ] **Step 3: Test syntax by validating CSS block**
Run: `npm --prefix frontend run build`
Expected: Build succeeds without CSS syntax errors.

---

### Task 2: Implement Metric Metadata Dictionary & Render Function

**Files:**
- Modify: `frontend/src/pages/library.astro:1750-1865`

**Interfaces:**
- Consumes: `stats` object from `/api/novels/{id}/stats` API response.
- Produces: `renderEnhancedMetricsDashboard(stats)` client function.

- [ ] **Step 1: Define `METRIC_METADATA` map**

Add `METRIC_METADATA` definition in `library.astro` above `renderEnhancedMetricsDashboard`:

```javascript
const METRIC_METADATA = {
  // Structure
  word_count:                  { label: "Word Count", category: "structure", tooltip: "Total words analyzed across sampled chapters" },
  char_count:                  { label: "Character Count", category: "structure", tooltip: "Total characters in text" },
  sentence_count:              { label: "Sentence Count", category: "structure", tooltip: "Total sentences identified" },
  avg_sentence_len:            { label: "Avg Sentence Length", category: "structure", tooltip: "Average words/characters per sentence" },
  avg_sentences_per_paragraph: { label: "Sentences/Paragraph", category: "structure", tooltip: "Paragraph density and pacing structure" },
  dep_tree_depth:              { label: "Dep Tree Depth", category: "structure", tooltip: "Syntactic dependency tree depth (higher = more complex sentence grammar)" },

  // Prose & Style
  dialogue_ratio:              { label: "Dialogue Ratio", category: "prose", isPercent: true, tooltip: "Percentage of prose contained within quotation marks" },
  ttr:                         { label: "Lexical Density (TTR)", category: "prose", tooltip: "Type-Token Ratio: vocabulary diversity and richness" },
  adj_ratio:                   { label: "Adjective Ratio", category: "prose", isPercent: true, tooltip: "Proportion of adjectives relative to total word count" },
  verb_ratio:                  { label: "Verb Ratio", category: "prose", isPercent: true, tooltip: "Action verb frequency relative to word count" },
  particle_ratio:              { label: "Particle Ratio", category: "prose", isPercent: true, tooltip: "Grammatical particle density" },
  kanji_ratio:                 { label: "Kanji Density", category: "prose", isPercent: true, tooltip: "Kanji character density in Japanese text" },

  // Theme & Emotion
  compound_sentiment:          { label: "Sentiment Tone", category: "theme", tooltip: "Overall emotional valence score (-1.0 negative to +1.0 positive)" },
  theme_explication_ratio:     { label: "Thematic Explicitness", category: "theme", tooltip: "Density of explicit moral/theme statements per 10k units" },
  sensory_body_density:        { label: "Visceral Emotion", category: "theme", isPercent: true, tooltip: "Ratio of somatic body sensations vs direct emotion words" },
  outside_world_engagement:    { label: "World Grounding", category: "theme", isPercent: true, tooltip: "Engagement with setting and physical world descriptions" },

  // Pacing & Subplots
  linearity_subversion_score:  { label: "Temporal Complexity", category: "pacing", tooltip: "Non-linear story structure and flashback frequency" },
  temporal_shift_score:        { label: "Time Shifts", category: "pacing", tooltip: "Frequency of time jump markers and scene transitions" },
  narrative_feature_diversity: { label: "Subplot Diversity", category: "pacing", tooltip: "Co-occurrence diversity of secondary entities and subplots" }
};
```

- [ ] **Step 2: Implement `renderEnhancedMetricsDashboard(stats)`**

Replace the plain `stat-grid` assembly loop with `renderEnhancedMetricsDashboard(stats)`:

```javascript
function renderEnhancedMetricsDashboard(stats) {
  const keys = Object.keys(stats);
  const langPrefix = keys.some(k => k.startsWith("en_")) ? "en_" :
                     keys.some(k => k.startsWith("ja_")) ? "ja_" : "zh_";

  const normRadar = stats.normalized_radar || {};
  const baselines = stats.baselines?.radar?.web_novel || stats.baselines?.radar?.classic_lit || [];

  // 1. Hero Summary Header
  const heroHtml = `
    <div class="metrics-hero">
      <div class="hero-chip">
        <span class="hero-chip-label">Archetype</span>
        <span class="hero-chip-val" style="color: #a78bfa;">${stats.archetype_match?.closest_trope || 'Unknown'}</span>
      </div>
      <div class="hero-chip">
        <span class="hero-chip-label">Territory</span>
        <span class="hero-chip-val" style="color: #38bdf8;">${stats.archetype_match?.territory || 'Web Novel'}</span>
      </div>
      <div class="hero-chip">
        <span class="hero-chip-label">Word Count</span>
        <span class="hero-chip-val">${(stats[`${langPrefix}word_count`] || stats[`${langPrefix}char_count`] || 0).toLocaleString()}</span>
      </div>
      <div class="hero-chip">
        <span class="hero-chip-label">Lexical Richness</span>
        <span class="hero-chip-val" style="color: #34d399;">${stats[`${langPrefix}ttr`]?.toFixed(2) || 'N/A'}</span>
      </div>
    </div>
  `;

  // 2. Filter Tabs
  const tabsHtml = `
    <div class="metrics-category-tabs">
      <button class="category-tab active" data-cat="all">All Metrics</button>
      <button class="category-tab" data-cat="structure">Structure</button>
      <button class="category-tab" data-cat="prose">Prose & Style</button>
      <button class="category-tab" data-cat="theme">Theme & Emotion</button>
      <button class="category-tab" data-cat="pacing">Pacing & Narrative</button>
    </div>
  `;

  // 3. Metric Sections Grouping
  const categories = [
    { id: "structure", title: "Structure & Volume", class: "cat-structure", barClass: "bar-fill-structure" },
    { id: "prose", title: "Prose & Style", class: "cat-prose", barClass: "bar-fill-prose" },
    { id: "theme", title: "Theme & Emotion", class: "cat-theme", barClass: "bar-fill-theme" },
    { id: "pacing", title: "Pacing & Narrative", class: "cat-pacing", barClass: "bar-fill-pacing" }
  ];

  let sectionsHtml = "";

  categories.forEach(cat => {
    let cardsInCatHtml = "";
    Object.keys(METRIC_METADATA).forEach(key => {
      const meta = METRIC_METADATA[key];
      if (meta.category !== cat.id) return;

      const rawKey = `${langPrefix}${key}`;
      if (stats[rawKey] === undefined && stats[key] === undefined) return;

      const rawVal = stats[rawKey] !== undefined ? stats[rawKey] : stats[key];
      if (rawVal === null || rawVal === undefined) return;

      let displayVal = meta.isPercent ? `${(rawVal * 100).toFixed(1)}%` :
                       typeof rawVal === 'number' ? (rawVal % 1 !== 0 ? rawVal.toFixed(2) : rawVal.toString()) : rawVal;

      const normScore = normRadar[rawKey] !== undefined ? normRadar[rawKey] :
                        normRadar[key] !== undefined ? normRadar[key] : 0.5;

      const pctWidth = Math.min(100, Math.max(0, Math.round(normScore * 100)));

      // Benchmark comparison chip
      let benchChipHtml = "";
      if (normScore >= 0.65) {
        benchChipHtml = `<span class="benchmark-chip higher">High (+${Math.round((normScore - 0.5) * 100)}% vs Avg)</span>`;
      } else if (normScore <= 0.35) {
        benchChipHtml = `<span class="benchmark-chip lower">Low (-${Math.round((0.5 - normScore) * 100)}% vs Avg)</span>`;
      } else {
        benchChipHtml = `<span class="benchmark-chip neutral">Balanced</span>`;
      }

      cardsInCatHtml += `
        <div class="metric-card" data-category="${cat.id}">
          <div class="metric-card-header">
            <span class="metric-card-label">${meta.label}</span>
            <span class="metric-info-icon" title="${meta.tooltip}">ⓘ</span>
          </div>
          <div class="metric-val-row">
            <span class="metric-primary-val">${displayVal}</span>
            ${benchChipHtml}
          </div>
          <div class="percentile-bar-bg">
            <div class="percentile-bar-fill ${cat.barClass}" style="width: ${pctWidth}%;"></div>
          </div>
        </div>
      `;
    });

    if (cardsInCatHtml) {
      sectionsHtml += `
        <div class="metric-category-section" data-cat-section="${cat.id}">
          <div class="category-section-title ${cat.class}">${cat.title}</div>
          <div class="metric-card-grid">
            ${cardsInCatHtml}
          </div>
        </div>
      `;
    }
  });

  return `
    <div class="metrics-dashboard">
      ${heroHtml}
      ${tabsHtml}
      ${sectionsHtml}
    </div>
  `;
}
```

- [ ] **Step 3: Update `statsContent.innerHTML` call and add Tab click listener**

Update `statsContent.innerHTML = renderEnhancedMetricsDashboard(stats);` and attach category tab event listeners:

```javascript
statsContent.innerHTML = renderEnhancedMetricsDashboard(stats);

// Attach tab filter click handler
const tabBtns = statsContent.querySelectorAll(".category-tab");
tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    tabBtns.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const targetCat = btn.getAttribute("data-cat");
    const sections = statsContent.querySelectorAll("[data-cat-section]");
    sections.forEach(sec => {
      if (targetCat === "all" || sec.getAttribute("data-cat-section") === targetCat) {
        sec.style.display = "flex";
      } else {
        sec.style.display = "none";
      }
    });
  });
});
```

- [ ] **Step 4: Run frontend build to verify compilation**

Run: `npm --prefix frontend run build`
Expected: Build passes with zero errors.

- [ ] **Step 5: Commit changes**

```bash
git add frontend/src/pages/library.astro
git commit -m "feat(ui): implement enhanced visual metrics dashboard with hero stats, category tabs, and benchmark chips"
```
