# Design Document: KishoLens Monorepo Architecture

This design document outlines the directory structure, dependencies, dataset policy, and component responsibilities for KishoLens—a stylistic pacing and prose archetype dashboard for analyzing structural metrics of web novels and light fiction.

---

## 1. Directory Structure

KishoLens is organized as a dual-ecosystem monorepo containing Node.js (frontend) and Python (backend, ML, data pipeline) services.

```text
KishoLens/
├── pyproject.toml              # Root UV Workspace config
├── uv.lock                     # Shared Python lockfile
├── package.json                # Root Node.js task runner configuration
├── data/                       # Local database & raw cache (gitignored raw text)
│   ├── raw_cache/              # Cached HTML files from scrapers
│   └── kisholens.db            # SQLite database for metadata/processed metrics
│
├── frontend/                   # Astro + React project
│   ├── src/
│   │   ├── components/         # Astro components & React Islands
│   │   │   └── react/          # Interactive charting widgets (D3.js / Chart.js)
│   │   └── pages/              # Astro pages/routing
│   ├── package.json
│   └── astro.config.mjs
│
├── backend/                    # FastAPI web server
│   ├── pyproject.toml
│   └── src/
│       ├── main.py             # Entrypoint
│       └── routes/             # REST endpoints
│
├── ml/                         # NLP and feature extraction
│   ├── pyproject.toml
│   └── src/
│       ├── analyzer.py         # Prose & structural metric calculators
│       └── tokenizers/         # Language-specific segmentation wrappers
│
└── data_pipeline/              # Scrapers and ingestion normalizers
    ├── pyproject.toml
    └── src/
        ├── scraper.py          # Crawler & parser logic
        └── normalizer.py       # HTML stripper & Ruby (Furigana) extractor
```

---

## 2. Ingestion & Dataset Policy

To comply with copyright laws and terms of service, the data is partitioned as follows:

1.  **Publicly Shareable Dataset (Hugging Face / Kaggle):**
    *   **Metadata:** Title, Author, Genre, Word Count, Chapter count, Release date.
    *   **Prose Metrics:** Computed numerical indices per chapter (e.g., lexical diversity ratio, sentence length standard deviation, readability grades, sentence counts, dialogue percentage).
    *   **Public Domain Text:** Out-of-copyright novels (from Project Gutenberg or Aozora Bunko).
2.  **Private Local Storage:**
    *   **Raw Chapter Text:** The actual prose scraped from modern web fiction sites (Syosetu, Royal Road) is kept locally on the developer's machine and is **never** checked into Git or shared in the public dataset.

---

## 3. Python Service Configuration (Astral `uv`)

### Root `pyproject.toml`
Configures the Astral `uv` toolchain to manage all three Python directories under a shared workspace:
```toml
[tool.uv]
workspace = { members = ["backend", "ml", "data_pipeline"] }
```

### Services Dependency Breakdown
*   **`data_pipeline`:** `httpx` (async requests), `selectolax` (HTML parser), `pydantic` (schemas), `sqlmodel` (SQLite integration).
*   **`ml`:** `spacy` (tokenization), `sudachipy` & `sudachidict-core` (Japanese tokenization), `nltk` (readability scores), `sqlmodel`.
*   **`backend`:** `fastapi` (API), `uvicorn` (server), `sqlmodel`.

---

## 4. Frontend Configuration (Astro + React)

*   **Integration:** Astro serves as the main framework for layout and static pages. Interactive charts are rendered as React components using the island architecture (`client:visible`).
*   **Visualizations:** D3.js or Chart.js inside React to render interactive pace-diagrams (e.g., dynamic sentence length variation scatter plots, vocabulary density distributions).
*   **Styling:** Scoped Vanilla CSS in Astro files.
