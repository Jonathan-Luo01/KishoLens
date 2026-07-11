# ETL Pipeline Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Jupyter Notebook (`notebooks/etl_pipeline_prototype.ipynb`) that defines and implements a streamed ETL pipeline to ingest parallel fiction text from Hugging Face into a local SQLite database, with text cleaning and NLP feature extraction preview.

**Architecture:** 
- **Extract:** Streams dataset from Hugging Face (`datasets` library).
- **Transform:** Applies regex and BeautifulSoup-based cleaning for JP and EN text.
- **Load:** Commits clean text to SQLite tables via SQLModel.
- **Preview:** Performs base NLP token/character counts as feature stubs.

**Tech Stack:** Python, Jupyter, Hugging Face `datasets` library, SQLModel/SQLAlchemy, BeautifulSoup4/lxml.

## Global Constraints
- Python version >= 3.10
- All code modifications require verification.

---

### Task 1: Add Datasets Dependency & Sync

**Files:**
- Modify: `pyproject.toml`
- Run: `uv sync`

- [ ] **Step 1: Add `datasets` to pyproject.toml**
  Add `"datasets>=2.12.0"` to the `dependencies` list in `pyproject.toml` (right under `nltk`).
  
- [ ] **Step 2: Sync python environment**
  Run: `$env:PATH += ";$env:APPDATA\Python\Python314\Scripts"; uv sync`
  Expected: UV resolves and locks the new dependencies.

- [ ] **Step 3: Commit dependency changes**
  Run:
  ```bash
  git add pyproject.toml uv.lock
  git commit -m "chore: add datasets dependency for Hugging Face ingestion"
  ```

---

### Task 2: Create etl_pipeline_prototype.ipynb

**Files:**
- Create: `notebooks/etl_pipeline_prototype.ipynb`

- [ ] **Step 1: Create notebooks/ directory**
  Create the folder `notebooks/` if it does not exist.

- [ ] **Step 2: Write etl_pipeline_prototype.ipynb**
  Write the notebook content as a valid JSON file.
  The notebook will define:
  1. `datasets.load_dataset("NilanE/ParallelFiction-Ja_En-100k", split="train", streaming=True)`
  2. SQLModel tables:
     - `Novel` (id, title, author, source)
     - `Chapter` (id, novel_id, chapter_number, title, text_ja, text_en)
  3. Text cleaning methods:
     - `clean_html(text)`: BeautifulSoup parser.
     - `clean_japanese(text)`: Removes ruby tags (`｜` and `《 》`).
     - `clean_english(text)`: Strips translator/editor notes matching regex `\[TL note:.*?\]` or `[T/N: ...]`.
  4. ETL Orchestration:
     - Downloads the first 20 records (for speed and testing).
     - Cleans them.
     - Saves to `data/kisholens.db`.
  5. Baseline NLP Feature Extraction:
     - Computes token counts, sentence counts, punctuation density, and dialogue ratio.

- [ ] **Step 3: Commit the new notebook**
  Run:
  ```bash
  git add notebooks/etl_pipeline_prototype.ipynb
  git commit -m "feat: add ETL pipeline prototype notebook for Hugging Face ingestion"
  ```

---

### Task 3: Validate Notebook Execution

**Files:**
- Create: `notebooks/test_etl_notebook.py` (temporary test script)

- [ ] **Step 1: Create a test script to execute the notebook**
  Write a script `notebooks/test_etl_notebook.py` that extracts the Python code blocks from the notebook and executes them to verify the ETL runs successfully and loads data into SQLite.

- [ ] **Step 2: Run the validation script**
  Run: `$env:PATH += ";$env:APPDATA\Python\Python314\Scripts"; uv run python notebooks/test_etl_notebook.py`
  Expected output: Ingested and cleaned records successfully, printed database novel list.

- [ ] **Step 3: Clean up test script**
  Remove `notebooks/test_etl_notebook.py` so we don't commit test garbage.
