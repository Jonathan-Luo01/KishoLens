# KishoLens

> **Stylistic Pacing & Prose Archetype Analytics Engine for Web Fiction & Classical Literature**

KishoLens is a NLP and ML analytics engine designed for dissecting the structural, syntactic, and emotional DNA of fiction — sentence pacing, vocabulary diversity (TTR), dialogue rhythm, multi-lingual dependency tree depth, and 17 canonical prose archetypes across **2,800+ web novels and classical literature works**.

---

## 🌟 Key Features

- **📖 17 Canonical Prose Archetypes**: Zero-shot semantic genre classification powered by `sentence-transformers` (`all-MiniLM-L6-v2`), global mean centroid subtraction, and calibrated sigmoid confidence scoring.
- **🏮 Kishōtenketsu 4-Act Sentiment Arcs**: Dynamic emotional polarity curve tracking across **Ki (起/Introduction)**, **Shō (承/Development)**, **Ten (転/Twist)**, and **Ketsu (結/Resolution)**.
- **🌐 Tri-Language NLP Pipeline**: Native syntactic dependency tree parsing, POS tagging, and lexical density analysis for **English (spaCy)**, **Japanese (SudachiPy / Oseti)**, and **Chinese (spaCy / HanLP)**.
- **📊 Interactive Visual Dashboard**:
  - 8-Dimensional Archetype Radar Canvas with interactive spoke tooltips & baseline comparison (`Web Novel` vs `Classic Literature`).
  - Rhythmic Pacing Barcodes displaying paragraph word-density variations.
  - Categorized Metric Cards (Structure, Prose & Style, Theme & Emotion, Pacing) featuring percentile progress meters and benchmark chips (`+X% vs Avg`).
- **🔍 Multi-Faceted Doppelgänger Search**: Cosine + L1 multi-vector similarity search identifying top stylistic twin novels across the 2,800+ novel database in **< 1 ms**.

---

## 🛠️ Stack & Technology

| Layer | Technology |
|---|---|
| **Frontend** | Astro static/SSR, Vanilla CSS design tokens, HTML5 Canvas & SVG |
| **API Backend** | FastAPI, Uvicorn, Async context manager lifecycle |
| **Tri-Language NLP** | `spaCy` (`en_core_web_sm`, `zh_core_web_sm`), `NLTK VADER`, `SudachiPy`, `Oseti` |
| **Machine Learning** | `sentence-transformers` (`all-MiniLM-L6-v2`), `numpy`, `scipy` |
| **Database & Cache** | SQLModel (SQLite), pre-computed JSON disk vector caches (`vector_cache.json`) |
| **Package Managers** | Astral `uv` (Python), `npm` (Node.js) |

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python**: $\ge 3.10$ (Python 3.13 or 3.14 supported)
- **Node.js**: $\ge 18.0.0$
- **Astral `uv`**: Installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### 1. Clone & Install Dependencies

```bash
# 1. Clone the repository
git clone https://github.com/Jonathan-Luo01/KishoLens.git
cd KishoLens

# 2. Install Python dependencies (with NLP extras)
uv sync --extra nlp

# 3. Download required spaCy models for English and Chinese NLP
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download zh_core_web_sm

# 4. Install Node dependencies for frontend
npm install
npm --prefix frontend install
```

### 2. Run Local Development Servers

Run both the FastAPI backend and Astro frontend concurrently with a single command:

```bash
npm run dev
```

Or launch each service individually in separate terminals:

```bash
# Terminal 1: Start FastAPI Backend (http://localhost:8000)
uv run uvicorn kisholens.api.main:app --reload --port 8000

# Terminal 2: Start Astro Frontend (http://localhost:4321)
npm --prefix frontend run dev
```

Visit **`http://localhost:4321`** in your browser to launch the KishoLens dashboard!

---

## 📡 REST API Documentation

The FastAPI backend exposes the following key endpoints on `http://localhost:8000`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health check and NLP model status |
| `POST` | `/api/analyze` | Real-time analysis of pasted prose (returns 20+ features, archetype, 4-act arc, pacing barcode, and doppelgänger matches) |
| `GET` | `/api/novels` | List database novels with search query, genre, and territory filters |
| `GET` | `/api/novels/{id}/stats` | Pre-computed aggregated statistics & benchmark percentages for a specific novel |
| `GET` | `/api/novels/{id}/arc` | 4-act Kishōtenketsu sentiment arc array for a specific novel |

### Sample API Request (`POST /api/analyze`)

```bash
curl -X POST "http://localhost:8000/api/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Inspector Holmes knelt by the corpse, examining the faint scent of bitter almonds clinging to the victim lips.", "lang": "auto"}'
```

---

## 🧪 Testing & Verification

Run the full test suite to verify NLP feature extraction, semantic matchers, and API endpoints:

```bash
# Run Python backend unit test suite (37 tests)
uv run pytest tests/

# Verify Astro static build compilation
npm --prefix frontend run build
```

---

## 📁 Project Structure

```
KishoLens/
├── kisholens/              # Python Backend Source
│   ├── api/                # FastAPI application & REST routes (`main.py`)
│   ├── ml/                 # NLP feature extraction, sentiment arcs, semantic match
│   └── pipeline/           # Dataset ingestion pipelines & scrapers
├── frontend/               # Astro Frontend Dashboard
│   └── src/
│       ├── pages/          # Astro pages (`index.astro`, `analyze.astro`, `library.astro`)
│       └── styles/         # Global CSS tokens & themes (`global.css`)
├── data/                   # Database & pre-computed disk caches
│   ├── kisholens.db        # SQLite database (2,800+ novels)
│   ├── stats_cache.json    # Pre-computed novel statistics cache
│   └── vector_cache.json   # Pre-computed 8D feature vector cache
├── tests/                  # PyTest suite (`test_api_semantic.py`, `test_similarity.py`, etc.)
├── pyproject.toml          # Astral uv dependency manifest
└── package.json            # Node.js dev scripts
```

---

## 🛡️ License & Dataset Policy

Raw chapter texts from modern web fiction platforms are copyrighted by their respective authors and are **never committed to this repository**. All included database metrics, public domain Gutenberg texts, and model centroids are free for academic and non-commercial research use.
