# Design Spec: Prose-Driven Multi-Signal Territory Classifier

## 1. Overview & Objective
Currently, KishoLens predicts a text's archetype (e.g., *Mystery*, *Isekai*, *Romance*) and assigns its Territory (*Classic Literature Territory* vs. *Web Novel Territory*) via static genre lookup. However, narrative craft and territory are fundamentally defined by **how the prose is structured and paced** rather than merely the subject matter.

This specification defines a **prose-driven territory classification engine** that evaluates syntax, sentence geometry, dialogue density, paragraph pacing, and semantic embedding vectors alongside a gentle genre prior.

---

## 2. Mathematical Scoring Model

The hybrid classifier computes a score for each territory $T \in \{\text{Classic Literature Territory}, \text{Web Novel Territory}\}$ using a linear combination of three normalized signals:

$$\text{Score}(T) = 0.60 \cdot S_{\text{stylistic}}(T) + 0.30 \cdot S_{\text{embedding}}(T) + 0.10 \cdot S_{\text{genre}}(T)$$

### Signal 1: Syntactic & Structural Stylistics ($60\%$)
Measures surface and grammatical features against empirical language baselines:
- **Average Sentence Length (`avg_sentence_len`)**:
  - Classic Literature: $\sim 25\text{--}45+$ words (English) / $45\text{--}75+$ chars (CJK).
  - Web Novel: $\sim 8\text{--}16$ words (English) / $18\text{--}35$ chars (CJK).
- **Dialogue Ratio (`dialogue_ratio`)**:
  - Classic Literature: $15\%\text{--}30\%$.
  - Web Novel: $40\%\text{--}65\%+$.
- **Paragraph Density (`avg_sentences_per_paragraph`)**:
  - Classic Literature: $4\text{--}8+$ sentences/paragraph.
  - Web Novel: $1\text{--}2.5$ sentences/paragraph (optimized for vertical scrolling).
- **Dependency Tree Depth (`dep_tree_depth`)**:
  - Classic Literature: higher syntactic nesting ($5\text{--}9$).
  - Web Novel: direct, action-oriented clauses ($2\text{--}4$).
- **Lexical Density (`ttr`)**:
  - Classic Literature: high type-token ratio with varied vocabulary.
  - Web Novel: accessible vocabulary with recurring character/stat terms.

### Signal 2: 384D Territory Corpus Embeddings ($30\%$)
Computes cosine similarity between the input text's `all-MiniLM-L6-v2` embedding and the precomputed territory centroids in `data/territory_centroids.npy`:
- $C_{\text{classic}}$: Derived from Project Gutenberg canonical literature.
- $C_{\text{web}}$: Derived from web fiction corpus (RoyalRoad / Syosetu / Webnovel).

### Signal 3: Genre Affinity Prior ($10\%$)
Provides a gentle prior derived from canonical `GENRE_TERRITORIES`:
- $1.0$ for matching territory, $0.0$ for non-matching territory.

---

## 3. Component Architecture

```
[User Passage / Request Text]
        │
        ├──► extract_features(text, lang) ──► 8D Stylistic Metrics & Pacing
        ├──► embed_texts(text)              ──► 384D Embedding Vector
        └──► analyze_prose(text)            ──► Primary Genre & Top Tropes
                     │
                     ▼
           [ classify_territory() ]
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
     Stylistic   Embedding     Genre
     Distance     Cosine       Prior
       (60%)       (30%)       (10%)
         └───────────┬───────────┘
                     ▼
       [ Composite Territory Scores ]
         - territory: "Classic Literature Territory" | "Web Novel Territory"
         - territory_confidence: 0.50 – 1.00
         - territory_scores: [...]
         - territory_breakdown: {...}
```

---

## 4. API & Data Contract

### Updated `/api/analyze` Response Payload:
```json
{
  "status": "success",
  "archetype": {
    "archetype": "Mystery",
    "confidence": 0.95,
    "description": "Classification: Classic Literature Territory. Stylistically and semantically matched.",
    "territory": "Classic Literature Territory",
    "territory_confidence": 0.88,
    "top_genres": [...],
    "top_territories": [
      { "territory": "Classic Literature Territory", "score": 0.88 },
      { "territory": "Web Novel Territory", "score": 0.12 }
    ]
  },
  "semantic": {
    "genre": "Mystery",
    "genre_confidence": 0.95,
    "territory": "Classic Literature Territory",
    "territory_confidence": 0.88,
    "territory_scores": [...],
    "territory_breakdown": {
      "stylistic": { "Classic Literature Territory": 0.85, "Web Novel Territory": 0.15 },
      "embedding": { "Classic Literature Territory": 0.90, "Web Novel Territory": 0.10 },
      "genre_prior": { "Classic Literature Territory": 1.0, "Web Novel Territory": 0.0 }
    }
  }
}
```

---

## 5. Verification & Test Plan

1. **Ornate Mystery (Sherlock Holmes)**:
   - Complex sentences + low dialogue $\rightarrow$ **`Classic Literature Territory`** ($> 85\%$).
2. **Snappy Modern Web Mystery**:
   - Short sentences, 1-sentence paragraphs, 60% dialogue $\rightarrow$ **`Web Novel Territory`** ($> 65\%$).
3. **Standard LitRPG / Isekai**:
   - System prompts, rapid dialogue $\rightarrow$ **`Web Novel Territory`** ($> 85\%$).
4. **Literary High Fantasy (Tolkien-style exposition)**:
   - Dense worldbuilding, multi-clause syntax $\rightarrow$ **`Classic Literature Territory`** ($> 75\%$).
5. **Full Regression Suite**:
   - PyTest test suite remains 100% passing across all 70+ tests.
