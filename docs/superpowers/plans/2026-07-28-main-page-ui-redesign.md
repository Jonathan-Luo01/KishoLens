# Main Page UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the KishoLens homepage (`index.astro`) into a high-impact, state-of-the-art AI application landing page with glassmorphism navbar, hero demo widget, 4-card feature grid, and live database stats banner while preserving 1-click revertibility via a backup file.

**Architecture:** Astro component redesign with embedded CSS variables, vanilla CSS glassmorphism, responsive grid layouts, SVG interactive Kishōtenketsu 4-act curve preview, and semantic HTML5 sections.

**Tech Stack:** Astro, Vanilla CSS, Inter/Outfit Google Fonts, SVG, HTML5.

## Global Constraints

- Preserve 1-click revertibility by creating `frontend/src/pages/index.astro.bak`.
- No TailwindCSS or external CSS frameworks; use Astro `<style>` blocks and CSS custom properties.
- Ensure 100% build validity (`npm --prefix frontend run build` must succeed).

---

### Task 1: Create Backup File for Revertibility

**Files:**
- Create: `frontend/src/pages/index.astro.bak`

**Interfaces:**
- Consumes: `frontend/src/pages/index.astro`
- Produces: `frontend/src/pages/index.astro.bak` (exact copy for 1-click restore)

- [ ] **Step 1: Copy original `index.astro` to `index.astro.bak`**

```bash
cp frontend/src/pages/index.astro frontend/src/pages/index.astro.bak
```

- [ ] **Step 2: Verify copy existence**

```bash
ls -la frontend/src/pages/index.astro.bak
```
Expected: File exists with identical byte size as `index.astro`.

- [ ] **Step 3: Commit backup file**

```bash
git add frontend/src/pages/index.astro.bak
git commit -m "chore(backup): create index.astro.bak for 1-click main page UI revertibility"
```

---

### Task 2: Implement Redesigned Homepage UI

**Files:**
- Modify: `frontend/src/pages/index.astro`

**Interfaces:**
- Consumes: Design spec from `docs/superpowers/specs/2026-07-28-main-page-ui-redesign.md`
- Produces: Redesigned modern AI product homepage for KishoLens

- [ ] **Step 1: Write redesigned `frontend/src/pages/index.astro`**

Replace `frontend/src/pages/index.astro` with complete redesigned layout, including:
1. Glassmorphism navbar (`.navbar`)
2. Hero section with pill badge, headline, dual CTA buttons, and ambient glow (`.hero-section`)
3. Interactive SVG 4-act Kishōtenketsu pacing demo card (`.hero-demo-card`)
4. 4-card feature grid (`.features-grid` & `.feature-card`)
5. Database metrics banner (`.stats-banner`)
6. CTA card & modern footer (`.footer`)

- [ ] **Step 2: Run frontend build to verify compilation**

```bash
npm --prefix frontend run build
```
Expected: `3 page(s) built in ... ms. Complete!`

- [ ] **Step 3: Capture screenshot with agent-browser to verify UI aesthetics**

```bash
agent-browser open http://localhost:4321/ && agent-browser wait 2000 && agent-browser screenshot /Users/jonathan/.gemini/antigravity-cli/brain/5a00b53a-91af-46a3-9313-2636e2a995b2/main_page_redesigned.png
```
Expected: Screenshot saved showing ambient glow, demo card, feature grid, and stats.

- [ ] **Step 4: Commit redesigned index page**

```bash
git add frontend/src/pages/index.astro
git commit -m "feat(ui): redesign main page with glassmorphism navbar, hero demo card, feature grid, and stats banner"
```
