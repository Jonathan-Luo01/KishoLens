# Design Document: ETL Pipeline Prototype Notebook

This design document outlines the prototype notebook designed to ingest, clean, and store Hugging Face datasets (specifically Japanese/English parallel web fiction) for the KishoLens platform.

---

## 1. Goal and Overview

The goal is to create a general-purpose ETL pipeline in a Jupyter Notebook (`notebooks/etl_pipeline_prototype.ipynb`). The pipeline:
1.  **Extracts** parallel text datasets from the Hugging Face Hub (streaming mode).
2.  **Transforms** (cleans) the text by stripping fan-translation artifacts, normalizing characters, and removing HTML tags.
3.  **Loads** the cleaned corpus into a local SQLite database (`data/kisholens.db`).
4.  **Previews** basic NLP feature extraction for downstream ML tasks.

---

## 2. Directory Layout & Pipeline Components

```text
KishoLens/
├── notebooks/
│   └── etl_pipeline_prototype.ipynb   # Main prototype notebook
├── data/
│   └── kisholens.db                  # Output SQLite DB
└── docs/
    └── superpowers/specs/
        └── 2026-07-11-etl-pipeline-prototype-design.md
```

### ETL Stage Definitions

#### A. Extract (Hugging Face Datasets)
*   Uses `datasets.load_dataset` in `streaming=True` mode.
*   Configures chunk/batch generation to keep execution fast and memory-efficient.

#### B. Transform (Cleaning Pipeline)
*   **HTML Stripping:** Removes residual tags/entities using `BeautifulSoup` with `lxml`.
*   **Japanese Normalization:** Normalizes unicode, standardizes spacing, and separates or strips ruby/furigana text notations.
*   **English Normalization:** Cleans fan-translation metadata, translator notes (e.g. `[TL note: ...]`), editor credits, and inconsistent line endings.

#### C. Load (SQLModel)
*   Configures local SQLite database schema.
*   Maps raw rows to clean `Novel` and `Chapter` records.

#### D. Feature Extraction Preview
*   Computes simple text statistics (sentence count, dialogue-to-prose ratio, word lengths) as a precursor to the main `ml` module.
