# Design Specification: Accurate Literary Similarity & Style Matching Engine

**Date:** 2026-08-10  
**Author:** Antigravity Team  
**Status:** Proposed  
**Target Files:** `kisholens/ml/similarity.py`, `kisholens/api/main.py`, `tests/ml/test_similarity.py`

---

## 1. Executive Summary & Problem Statement

When analyzing prose excerpts in the KishoLens Analyzer (e.g., a Victorian detective mystery excerpt featuring Sherlock Holmes at 221B Baker Street), the "Similar Works & Stylistic Matches" section returned unrelated Japanese isekai/fantasy web novels (*Ihoujin, Dungeon ni Moguru*, *Reincarnated Princess Wishes to Avoid Death*, *Free Life Fantasy Online*) instead of canonical mystery classics (*The Adventures of Sherlock Holmes*, *The Memoirs of Sherlock Holmes*, etc.).

### Root Causes
1. **Unhydrated Vector Cache (`_novel_vector_cache`)**:
   `find_top_matches` in `kisholens/ml/similarity.py` checks an in-memory dictionary `_novel_vector_cache`. Because this cache was not hydrated from `data/stats_cache.json` on startup, all 10,320 database novels fell back to a neutral 8D vector `[0.5, 0.5, ...]`, flattening stylistic differentiation across the entire library.
2. **Coarse Database Genre Overlap**:
   `find_top_matches` matched against raw SQLite `novel.genre` strings rather than the 17-genre taxonomy classifications and confidence scores precomputed in `data/stats_cache.json`.
3. **Absence of Primary Genre Affinity**:
   When an excerpt has a strong dominant genre (e.g., `Mystery @ 95.1%`), generic web novels with broad `Action / Adventure` tags overwhelmed the ranking due to catalog size (~10,000 web novels vs ~300 classics).

---

## 2. Architecture & Design

```
+-------------------------------------------------------------------------+
|                          FastAPI Startup / API Call                     |
|                                     |                                   |
|                  _load_similarity_cache_from_disk()                     |
|                                     v                                   |
|                 Reads: data/stats_cache.json (10,320 works)              |
|                                     v                                   |
|               Populates: _novel_vector_cache (In-Memory)                |
|               - 8D Normalized Stylistic Radar Vector                    |
|               - Primary & Secondary Taxonomy Genres + Scores            |
|               - Territory & Author Metadata                             |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  find_top_matches(query_features, text)                 |
|                                                                         |
|  1. Extract Query 8D Vector (q_vec) & Primary Genre (q_primary)         |
|  2. Calculate 5-Factor Similarity for Candidates:                       |
|     - Factor 1 (30%): 8D Stylistic Radar Cosine + L1 Distance           |
|     - Factor 2 (35%): Genre Overlap + Primary Genre Affinity            |
|     - Factor 3 (20%): Semantic Concept Vector Cosine                    |
|     - Factor 4 (15%): Territory Semantic Similarity                     |
|  3. Sort by (Primary Genre Match, Composite Score Descending)           |
|  4. Return Top K Matches with Explanatory Breakdown                     |
+-------------------------------------------------------------------------+
```

---

## 3. Detailed Component Specifications

### 3.1 Cache Hydration (`kisholens/ml/similarity.py`)
Add `_init_cache_from_disk()` that automatically loads `data/stats_cache.json` on module import or first invocation:
```python
def _init_cache_from_disk():
    global _novel_vector_cache
    if _novel_vector_cache:
        return
    # Load data/stats_cache.json
    # For each novel ID:
    #   Extract 8D radar vector from normalized_radar
    #   Extract primary genre, top_genres, territory, title, author
    #   Store in _novel_vector_cache[novel_id]
```

### 3.2 8D Stylistic Vector Construction
Ensure the 8 canonical dimensions are consistently mapped:
1. `theme_explication_ratio`
2. `linearity_subversion_score`
3. `sensory_body_density`
4. `outside_world_engagement`
5. `narrative_feature_diversity`
6. `dialogue_ratio`
7. `ttr`
8. `temporal_shift_score`

### 3.3 Enhanced Multi-Factor Scoring
- **Style Similarity (30% weight)**:
  $$\text{StyleSim} = 0.5 \times \cos(\vec{q}, \vec{n}) + 0.5 \times \max(0, 1 - 4 \cdot \|\vec{q} - \vec{n}\|_1)$$
- **Genre Similarity & Primary Affinity (35% weight)**:
  - If query has a strong primary genre ($S_{\text{primary}} \ge 0.70$) and candidate has the same primary genre:
    $$\text{GenreSim} = 0.85 + 0.15 \times \text{Jaccard}(G_q, G_n)$$
  - Else if primary genre is shared in top 3:
    $$\text{GenreSim} = 0.60 + 0.25 \times \text{Jaccard}(G_q, G_n)$$
  - Else:
    $$\text{GenreSim} = 0.20 \times \text{Jaccard}(G_q, G_n)$$
- **Semantic Concept Embedding (20% weight)**:
  Sentence-transformer cosine similarity between query concept string and candidate concept embedding.
- **Territory Similarity (15% weight)**:
  Embedding cosine / exact match between `Classic Literature Territory` vs `Web Novel Territory`.

---

## 4. Verification & Success Criteria

1. **Empirical Match Quality**:
   - For the Sherlock Holmes Mystery excerpt, top matches **must** return Arthur Conan Doyle works (*The Adventures of Sherlock Holmes*, *The Memoirs of Sherlock Holmes*, etc.) with primary genre **Mystery** and composite similarity $> 75\%$.
2. **Speed & Latency**:
   - In-memory similarity matching across 10,320 cached novels must execute in $< 25\text{ms}$.
3. **Automated Tests**:
   - Write unit tests in `tests/ml/test_similarity.py` verifying cache hydration and top matches accuracy for Mystery, Romance, and Isekai samples.
   - Run full pytest test suite (100% passing).
