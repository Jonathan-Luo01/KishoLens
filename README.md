# KishoLens

> Stylistic pacing and prose archetype analysis for web novels and light fiction.

KishoLens is a dashboard for analyzing the structural and stylistic DNA of web novels — sentence pacing, vocabulary density, dialogue rhythm, and prose archetypes — across various sources of modern light/web novels and public domain corpora.

## Stack

| Layer | Technology |
|---|---|
| **Frontend** | Astro + React islands, Vanilla CSS |
| **API** | FastAPI + Uvicorn |
| **NLP / ML** | spaCy, SudachiPy (Japanese), NLTK (English), HanLP (Chinese) |
| **HTML Parsing** | lxml + BeautifulSoup4 |
| **HTTP Client** | aiohttp (async) |
| **Data layer** | SQLModel + SQLite, Hugging Face Datasets (streaming) |
| **Python tooling** | Astral uv |
| **Node tooling** | npm + concurrently |

## Project Structure

```
KishoLens/
├── kisholens/              # All Python source
│   ├── api/                # FastAPI server  →  uv run uvicorn kisholens.api.main:app --reload
│   ├── ml/                 # NLP feature extraction
│   └── pipeline/           # Scrapers + normalizers
├── frontend/               # Astro + React dashboard
│   └── src/
│       ├── pages/          # Astro pages & routing
│       ├── components/     # React islands (charts, viz)
│       └── styles/         # CSS tokens & global styles
├── data/                   # Local only — never committed
│   ├── raw_cache/          # Cached HTML from scrapers
│   └── kisholens.db        # SQLite (metadata + metrics)
├── pyproject.toml          # Single uv package
└── package.json            # npm dev runner
```

## Setup

### Prerequisites

- Python ≥ 3.10 (3.13 recommended for NLP extras)
- Node.js ≥ 22.12.0
- [Astral uv](https://docs.astral.sh/uv/getting-started/installation/)

### Install

```bash
# Python deps (core)
uv sync

# NLP extras — requires Python ≤ 3.13 (spaCy wheels)
uv sync --extra nlp

# Node deps (frontend + dev runner)
npm install
npm --prefix frontend install
```

## Development

```bash
# Start API + frontend concurrently
npm run dev

# Or individually
npm run dev:backend    # FastAPI on http://localhost:8000
npm run dev:frontend   # Astro on http://localhost:4321
npm run dev:pipeline   # Run ingestion pipeline
npm run dev:ml         # Run NLP analyzer
```

Health check: `curl http://localhost:8000/health`

## Dataset Policy

Raw chapter text from modern web fiction platforms is **copyrighted by their authors** and is **never committed to this repository**.

| Data | Status |
|---|---|
| Raw scraped chapter text | 🔒 Local only (`data/raw_cache/`) |
| SQLite database | 🔒 Local only (`data/kisholens.db`) |
| Computed prose metrics | ✅ Shareable (HuggingFace / Kaggle) |
| Novel metadata | ✅ Shareable |
| Public domain text (Aozora / Gutenberg) | ✅ Shareable |

## Sources

| Source | Language | Access method |
|---|---|---|
| Shousetsuka ni Narou (Syosetu) | Japanese | Official JSON API + HTML scraping |
| Royal Road | English | HTML scraping (rate-limited) |
| Aozora Bunko | Japanese | Public domain archive |
| Project Gutenberg | English | Public domain archive |

## Architecture

```
Scraper (pipeline/)  →  Normalizer  →  SQLite
                                           ↓
                              NLP Analyzer (ml/)
                                           ↓
                               FastAPI (api/)  →  Astro Dashboard
```
