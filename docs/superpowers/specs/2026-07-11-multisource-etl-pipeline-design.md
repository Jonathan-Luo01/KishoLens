# Design Document: Multi-Source ETL Ingestion Pipeline

This document designs the extension of the KishoLens prototype ETL pipeline notebook to support multiple Hugging Face datasets:
1.  `NilanE/ParallelFiction-Ja_En-100k` (Japanese/English Parallel Fiction)
2.  `botp/RyokoAI_ScribbleHub17K` (English Creative Writing)
3.  `OmniAICreator/RoyalRoad-1.61M` (English Web Fiction)

---

## 1. Goal

Generalize `notebooks/etl_pipeline_prototype.ipynb` to support a configuration-driven registry pattern. The pipeline will ingest a configurable slice of chapters from any of these three datasets, clean the text dynamically using source-specific schemas, and store the output in the local SQLite database (`data/kisholens.db`).

---

## 2. Config-Driven Registry Mapping

A central registry dictionary maps the Hugging Face dataset identifier to the parser rules:

### A. ParallelFiction-Ja_En-100k
- **Source Platform:** Syosetu Fan Translation (Parallel)
- **Series Title:** `meta["general"]["series_title_eng"]`
- **Author:** `meta["syosetu"]["writer"]`
- **Text cleaning:** `BeautifulSoup` + custom regex for ruby tags and multiline English translator notes.

### B. RyokoAI_ScribbleHub17K
- **Source Platform:** Scribble Hub (English)
- **Series Title:** `meta["title"]`
- **Author:** `meta["author"]`
- **Text cleaning:** HTML stripping and English translator notes regex removal.

### C. RoyalRoad-1.61M
- **Source Platform:** Royal Road (English)
- **Series Title:** `title` (flat column)
- **Author:** `author` (flat column)
- **Chapter Number:** `chapter_id` (float representation converted to int)
- **Chapter Title:** `chapter_title`
- **Text cleaning:** HTML stripping and English translator notes regex removal.

---

## 3. Database Ingestion

For monolingual English datasets, the `text_ja` field in the `Chapter` table is populated as an empty string (`""`), preserving database integrity without modifying the relational schema.
