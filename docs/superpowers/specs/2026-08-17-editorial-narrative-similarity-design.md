# Spec: Editorial Matrix & Natural Literary Narrative Reasoning

## 1. Goal
Revamp the narrative similarity interface and generation engine to eliminate robotic, AI-sounding phrasing, eliminate text truncation (`...`), and provide a spacious, elegant **Editorial Matrix** that displays clear narrative resonance between target works and candidate matches.

---

## 2. Problems Addressed
1. **Truncation & Cramped Layout**: Dual-capsule pills inside narrow cards caused text values like *"Reincarnation into Imperial Nobility"* to truncate with ellipses (`...`).
2. **Formulaic / AI-Sounding Phrasing**: Sentence templates relied on rigid clichés (*"anchored by a catalyst"*, *"thematic beats"*, *"socio-political hierarchy"*, *"richly drawn backdrop"*).
3. **Information Redundancy**: Cards repeated the exact same phrase 3 times across the header, pill, and explanation paragraph.

---

## 3. UI Architecture: The Editorial Matrix

### Visual Structure
- **Grid Layout**: Spacious responsive grid (`minmax(340px, 1fr)`) allowing full multi-line text wrapping with zero truncation.
- **Pillar Card Anatomy**:
  1. **Header**: Thematic Icon (`⚡`, `🏰`, `⚔️`, `✍️`) + Pillar Name + Score Badge (e.g. `74% Alignment`).
  2. **Affinity Progress Bar**: 3px smooth gradient accent bar.
  3. **Comparative Alignment Row**:
     - `Query / Target`: Tag pill with subtle background tint.
     - Directional Indicator `➔`
     - `Matched Novel`: Tag pill with matching accent border.
  4. **Editorial Insight**: 1 crisp, human-grade sentence explaining why the two works resonate, without repeating raw tag labels verbatim.

### Theme & Styling Tokens
- **Catalyst (`⚡`)**: Indigo gradient (`#818cf8` ➔ `#a78bfa`)
- **Setting (`🏰`)**: Sky / Cyan gradient (`#38bdf8` ➔ `#67e8f9`)
- **Conflict (`⚔️`)**: Rose gradient (`#f43f5e` ➔ `#fb7185`)
- **Style Cadence (`✍️`)**: Emerald gradient (`#34d399` ➔ `#6ee7b7`)

---

## 4. Natural Literary Synthesis Engine

### Core Phrasing Rules
1. **No Robotic Fillers**: Ban phrases like *"thematic beats"*, *"anchored by a foundation"*, *"situated in a world where"*, *"socio-political hierarchy"*.
2. **Dynamic Editorial Hooks**:
   - Focus on the protagonist's core dilemma, world dynamics, and tension.
   - For Court / Reincarnation:
     > *"Both stories plunge an exceptionally gifted reincarnator into the dangerous social minefield of royal court politics—one mastering covert empire-building, the other maneuvering to shield family from political ruin."*
   - For User Prose to Novel:
     > *"Your prose shares this novel's royal court atmosphere and reincarnation hook, building suspense around a protagonist maneuvering treacherous noble rivalries."*
   - For Action / Dungeon / System:
     > *"Both narratives center on an underdog protagonist climbing through high-stakes monster raids while keeping god-tier power hidden from rival guilds."*
   - For Slow Life / Romance:
     > *"Both stories offer a warm, character-driven journey balancing pastoral tranquility with humorous noble complications."*

### Natural Pillar Rationale
- **Catalyst**:
  - Match: *"Both narratives launch from an otherworldly rebirth into high aristocracy."*
  - Shift: *"Transitions from royal birthright into villainess subversion."*
- **Setting**:
  - Match: *"High fantasy imperial palaces governed by ruthless noble factions."*
  - Contrast: *"Translates court intrigue to an otome high-society academy."*
- **Conflict**:
  - Match: *"Maneuvering court intrigue, concealing power, and averting aristocratic ruin."*
  - Parallel: *"Balancing imperial succession against urgent survival stakes."*
- **Style Cadence**:
  - Format: *"Dialogue-driven scene rhythm (68%) with snappy, fast-paced sentence flow (10.3 w/s)."*

---

## 5. Implementation Scope
1. **Backend (`kisholens/ml/similarity.py`)**:
   - Refactor `_generate_narrative_synthesis` to use dynamic editorial narrative synthesis.
   - Refactor `_compute_4pillar_breakdown` to produce clean, natural human editorial rationale without repetitive filler.
2. **Batch Precomputation (`scripts/recalculate_all_similarities.py`)**:
   - Recompute all 10,320 novels in `data/stats_cache.json` with the new editorial engine.
3. **Frontend Drawers (`library.astro`, `analyze.astro`, `global.css`)**:
   - Implement the zero-truncation Editorial Matrix layout with full text wrapping, visual chips, progress bars, and high-contrast light/dark themes.
4. **Automated Verification**:
   - Unit tests in `tests/ml/test_descriptive_similarity.py`.
   - Astro production build verification.
