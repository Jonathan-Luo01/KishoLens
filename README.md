# KishoLens

A computational stylometry and narrative similarity platform for web fiction and classical literature. KishoLens analyzes sentence structure, dialogue cadence, vocabulary richness, and structural pacing across 10,000+ works in English, Japanese, and Chinese.

[Live Demo](https://kisholens.pages.dev)

---

## Core Capabilities

### 1. Multi-Lingual Prose Stylometry (37 Metrics)
Custom NLP tokenization pipelines extract 37 forensic literary dimensions across English (`spaCy`, `NLTK VADER`), Japanese (`SudachiPy`, `Oseti`), and Chinese (`spaCy`):
- **Structure & Volume**: Word/character counts, sentence count, average sentence length, sentences per paragraph, syntactic dependency tree depth.
- **Prose & Style**: Dialogue ratio, Type-Token Ratio (TTR / lexical richness), adjective/verb/particle distribution ratios, Kanji density.
- **Theme & Emotion**: Compound sentiment polarity, thematic explicitness ratio, visceral/somatic sensory density, outside-world engagement score.
- **Pacing & Narrative**: Linearity subversion score, temporal shift frequency, entity/subplot diversity, paragraph length variance.

### 2. Explainable Similarity Engine ("Doppelgängers")
Identifies stylistic and thematic twin works across the 10,320-novel database in **< 2ms**:
- **5-Factor Similarity Model**: Combines 384D semantic embeddings (`all-MiniLM-L6-v2`), 8D stylometric fingerprint vectors, parent genre overlap (Jaccard), fine-grained tag overlap, and territory semantic similarity.
- **NumPy Matrix Vectorization**: Full-corpus candidate screening vectorized with cosine dot products in memory.
- **4-Pillar Narrative Alignment**: Side-by-side comparative breakdown across *Premise Catalyst*, *Setting Atmosphere*, *Conflict Stakes*, and *Prose Cadence*.
- **Editorial Synthesis**: Dynamically generates concise editorial commentary explaining why works align.
- **Forensic Delta Table**: Interactive drawer comparing input prose metrics directly against matched works with visual alignment bars.

### 3. Interactive Visualization Suite
- **8-Axis Archetype Radar**: HTML5 Canvas radar chart with interactive spoke tooltips, spoken metric explanations, and baseline toggles (*Web Novel* vs *Classic Literature*).
- **Rhythmic Pacing Barcodes**: Visual chapter/paragraph density barcode displaying pacing tempo, scene velocity, and dialogue frequency variations.
- **4-Phase Kishōtenketsu Sentiment Arcs**: Dynamic emotional polarity curves tracking narrative progression across **Ki** (起 / Introduction), **Shō** (承 / Development), **Ten** (転 / Twist), and **Ketsu** (結 / Resolution).
- **Metric Cards Carousel**: Grouped by category with percentile progress meters and benchmark chips (`+X% vs Avg`).

### 4. Interactive Prose Analyzer (`/analyze`)
- Paste or type custom text in any supported language for instant NLP extraction, archetype prediction, and similarity matching.
- Preset literary excerpt loader for testing classic literature, translated light novels, wuxia, and modern web serials.

### 5. Library Explorer (`/library`)
- Search and browse 10,320 indexed works across Royal Road, Syosetu, Project Gutenberg, ScribbleHub, and CNNovel.
- Filter by Territory (*Web Novel* vs *Classic Literature*) and multi-tag inclusion/exclusion checkboxes.
- State preservation across client router navigation via `sessionStorage`.

---

## Architecture & Tech Stack

- **Frontend**: Astro 5, TypeScript, Vanilla CSS design tokens, HTML5 Canvas API (hosted on Cloudflare Pages)
- **Backend API**: FastAPI, Uvicorn, Python 3.11+ (deployed on Google Cloud Run)
- **NLP & Stylometry**: spaCy (`en_core_web_sm`, `zh_core_web_sm`), NLTK (VADER), SudachiPy, Oseti
- **Embeddings & ML**: Sentence-Transformers (`all-MiniLM-L6-v2`), NumPy, SciPy
- **Data Storage**: SQLite (`novel_stats.sqlite`), pre-computed JSON metadata, Cloudflare R2 object storage
- **Package Management**: Astral `uv` (Python), `npm` (Node.js)

---

## Local Setup

### Requirements
- Python 3.10+ (with `uv` recommended)
- Node.js 18+

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Jonathan-Luo01/KishoLens.git
cd KishoLens

# Install Python dependencies and NLP models
uv sync --extra nlp
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download zh_core_web_sm

# Install frontend dependencies
npm --prefix frontend install
```

### 2. Running Locally

Start both the backend and frontend concurrently:

```bash
npm run dev
```

Or run them individually:

```bash
# Backend (FastAPI on http://localhost:8000)
uv run uvicorn kisholens.api.main:app --reload --port 8000

# Frontend (Astro on http://localhost:4321)
npm --prefix frontend run dev
```

---

## API Reference

The FastAPI service exposes the following core endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status and loaded database diagnostics |
| `GET` | `/api/novels` | List indexed novels with genre, source, and territory filters |
| `GET` | `/api/novels/{id}/stats` | Full 37-metric profile, radar vector, pacing array, and top matches |
| `GET` | `/api/novels/{id}/arc` | 4-act Kishōtenketsu sentiment trajectory and quantile ranges |
| `POST` | `/api/analyze` | Real-time analysis of custom input prose |
| `GET` | `/api/db/stats` | Aggregated dataset counts and source distributions |
| `POST` | `/api/pipeline/ingest` | Trigger background scraping & ETL pipeline for new novels |
| `GET` | `/api/pipeline/jobs/{id}` | Ingestion job progress and completion status |

### Example: Analyze Custom Prose

```bash
curl -X POST "http://localhost:8000/api/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "The wind howled across the stone fortress as Duke Jeffrey unsheathed his blade...",
       "lang": "en"
     }'
```

---

## Testing

```bash
# Run backend test suite
uv run pytest

# Test frontend build
npm --prefix frontend run build
```

---

## Project Structure

```
KishoLens/
├── kisholens/              # Python backend package
│   ├── api/                # FastAPI application and route handlers
│   ├── ml/                 # Stylometric features, embeddings, similarity engine
│   ├── pipeline/           # Ingestion, scrapers, and ETL scripts
│   └── storage/            # Cloudflare R2 backup and storage utilities
├── frontend/               # Astro frontend application
│   ├── src/pages/          # Library explorer, prose analyzer, visualizer
│   └── src/styles/         # Design tokens, theme styling (Dark / Light)
├── data/                   # Precomputed SQLite databases and metadata
│   ├── novel_stats.sqlite  # Compact precalculated stats for all 10,320 novels
│   ├── novels_metadata.json# Index of titles, authors, genres, and chapter counts
│   └── vector_cache.json   # 8D stylometric fingerprint vectors
├── scripts/                # Ingestion and similarity recomputation scripts
└── tests/                  # Backend pytest test suite
```

---

## License & Dataset Notes

Raw scraped text from web platforms is subject to copyright by respective authors and is excluded from the repository. All statistical metrics, Gutenberg public domain data, and embeddings are provided for non-commercial research and educational use.
