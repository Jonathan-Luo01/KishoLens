# Multi-Source ETL Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify `notebooks/etl_pipeline_prototype.ipynb` to support a configuration-driven dataset registry to ingest and clean three different Hugging Face datasets (ParallelFiction, ScribbleHub, and Royal Road) into the SQLite database.

**Tech Stack:** Python, Jupyter, Hugging Face `datasets`, SQLModel/SQLite.

---

### Task 1: Update etl_pipeline_prototype.ipynb for Multi-Source Ingestion

**Files:**
- Modify: `notebooks/etl_pipeline_prototype.ipynb`

- [ ] **Step 1: Define Dataset Registries and Helpers**
  Update the code cells to define the dataset configuration mapping:
  - Add helper `parse_parallel_fiction(item, idx)` which encapsulates the existing ParallelFiction chapter title splitting, ja cleaning, and en cleaning logic.
  - Define a mapping dictionary `DATASET_REGISTRY` containing parsing schemas for:
    - `"NilanE/ParallelFiction-Ja_En-100k"`
    - `"botp/RyokoAI_ScribbleHub17K"`
    - `"OmniAICreator/RoyalRoad-1.61M"`
  
- [ ] **Step 2: Update run_etl Function**
  Modify `run_etl` signature to accept `dataset_name: str` and `num_records: int`.
  Inside `run_etl`:
  - Retrieve the registry configuration for the passed `dataset_name`.
  - Stream the corresponding dataset using `load_dataset`.
  - Ingest 20 records using the `extractor` function, resolve cache/deduplication, and insert.
  
- [ ] **Step 3: Add Multi-Source Testing Cells**
  Update the orchestrator execution cell to run the ETL pipeline for all three datasets sequentially (e.g. ingesting 5-10 records for each to confirm correctness).

- [ ] **Step 4: Commit updated notebook**
  Run:
  ```bash
  git add notebooks/etl_pipeline_prototype.ipynb
  git commit -m "feat: extend ETL pipeline prototype to support ScribbleHub and Royal Road datasets"
  ```

---

### Task 2: Validate Multi-Source Ingestion

**Files:**
- Create: `notebooks/test_multisource_etl.py` (temporary test script)

- [ ] **Step 1: Write validation script**
  Create `notebooks/test_multisource_etl.py` that extracts notebook code cells and executes them.

- [ ] **Step 2: Execute validation**
  Run the validation script:
  `$env:PATH += ";$env:APPDATA\Python\Python314\Scripts"; uv run python notebooks/test_multisource_etl.py`
  Expected: Ingests records from all three datasets and inserts them into `data/kisholens.db`. Prints novels and chapters in database from multiple sources.

- [ ] **Step 3: Clean up test script**
  Remove `notebooks/test_multisource_etl.py` so it is not committed.
