# Design Specification: Natural Multi-Archetype Editorial Synthesis & 4-Pillar Similarity Engine

**Date:** 2026-08-28  
**Status:** In Review  
**Target Module:** `kisholens.ml.similarity`, `frontend/src/pages/library.astro`, `frontend/src/pages/analyze.astro`

---

## 1. Executive Summary

KishoLens provides comparative stylometry, story similarity, and narrative anatomy reasoning between novels in the library and user-submitted prose. Currently, the narrative reasoning output can feel repetitive, cookie-cutter, or synthetic across different genres due to a context pollution bug (metadata `territory` triggering a generic conflict keyword) and limited template variations.

This feature overhauls the narrative similarity reasoning engine into a **Human-Grade Editorial Fiction Critic Engine**:
1. **Accurate Story Anatomy Extraction**: Fixes context search pollution, classifying novels across 25+ distinct subgenres and tropes (LitRPG, Xianxia, Dungeon Hunter, Otome Villainess, Cozy Slice of Life, Gothic Mystery, Classical Melodrama, etc.).
2. **Dynamic Editorial Synthesis**: Generates bespoke literary comparisons contrasting the Query's narrative engine against the Candidate's storytelling approach.
3. **Pillar Rationale Inside Cards**: Produces specific explanations for each pillar (Catalyst, Setting, Conflict, Style & Cadence) with metric-driven cadence evaluations and zero boilerplate fillers.

---

## 2. Architecture & Detailed Design

### A. Context Parsing & Fix for `territory` Pollution
- **Issue**: Previously, `context = f"{title} {synopsis} {tags} {p_genre} {territory} {query_text}".lower()` included `"Web Novel Territory"`, matching `r"territory"` in the warfare conflict rule.
- **Fix**: Sanitize `context` to only include the actual text, title, synopsis, tags, and specific genre tags. Extract `territory` separately without polluting lexical regexes.

### B. 25+ Granular Story Archetypes & Narrative Motifs
The engine extracts precise narrative building blocks:
- **Catalysts**:
  - `Overpowered Reincarnation & Blessed Prodigy`
  - `Villainess Fate Subversion & Second Chance`
  - `System Interface & Status Awakening`
  - `Regression & Time Reversal Redo`
  - `Transmigration into Aristocratic Society`
  - `Hero Summoning & Otherworldly Displacement`
  - `Sect Initiation & Foundation Cultivation`
  - `Betrayal & Vengeful Fall from Grace`
  - `Forensic Homicide & Occult Crime Discovery`
  - `Pastoral Relocation & Slow Life Resettlement`
  - `Academy Enrollment & Dormant Magic Awakening`
  - `Guild Bounty Contract & Frontier Exploration`
- **Settings**:
  - `High Fantasy Imperial Court & Noble Salons`
  - `Otome Aristocratic Empire & High Society`
  - `Urban Fantasy & Labyrinthine Monster Gates`
  - `Immortal Martial World & Wilderness Sects`
  - `High Magic Sorcery Academy & Enchanted Guilds`
  - `Dystopian Megacorp & Cybernetic Frontier`
  - `Pastoral Frontier Village & Cozy Tavern`
  - `Victorian Gothic Manor & Fog-Bound Alleys`
  - `Mythic Ancient Realm & Feudal Kingdom`
- **Conflicts**:
  - `Imperial Succession & Concealing Overpowered Might`
  - `Subverting Execution & Dismantling Death Flags`
  - `Climbing Monster Gates & Underdog Ascension`
  - `Sect Hierarchies & Transcending Mortal Limits`
  - `Vengeance & Retributive Justice Against Betrayers`
  - `Romantic Tension & High-Society Courtship`
  - `Deductive Investigation & Unmasking Conspirators`
  - `Domain Management & Pastoral Community Building`
  - `Psychological Dread & Supernatural Survival`

### C. Contrastive Narrative Synthesis Generator
`_generate_narrative_synthesis(q_anat, c_anat, s_sim, g_sim, is_user_input, q_title, c_title)`:
- Contrast the core narrative drive of both works (e.g. *"While both works originate from noble reincarnation, [Candidate] channels high-stakes survival as a villainess scheming to avert public execution, contrasting with [Query]'s focus on effortless imperial dominance."*).
- For user prose comparisons, articulate how the user's stylistic choices and pacing compare to the published novel's structure.

### D. Deep 4-Pillar Card Rationale
1. **Premise & Catalyst**: Explains how the opening inciting incident sets the tone and character motivation.
2. **World Setting**: Explains atmospheric depth, political vs. wilderness focus, and worldbuilding alignment.
3. **Conflict Stakes**: Highlights the central narrative tension driving character choices.
4. **Style & Cadence**: Computes dialogue-to-exposition balance (`dialogue_ratio`) and sentence length tempo (`avg_sentence_len`), contrasting snappy dialogue banter with descriptive expository layering.

---

## 3. Testing & Verification Plan

1. **Unit Tests (`tests/ml/test_descriptive_similarity.py`)**:
   - Verify story anatomy extraction across 15+ varied synopses (Isekai, Cultivation, Dungeon Hunter, Cozy Slice of Life, Mystery, Romance, Sci-Fi).
   - Assert zero occurrences of banned robotic fillers (`"thematic beats"`, `"richly drawn backdrop"`, `"factional friction and purposeful protagonist progression"`).
   - Verify non-empty, distinct explanations for all 4 pillars.
2. **Batch Cache Recomputation**:
   - Run `scripts/recalculate_all_similarities.py` to refresh `data/stats_cache.json`.
3. **Browser Visual Verification**:
   - Inspect `/library` with diverse novel selections (Noble Reincarnation, Slice of Life, Action, Romance) in both Dark and Light modes.
   - Inspect `/analyze` with mystery, fantasy, and isekai samples.
