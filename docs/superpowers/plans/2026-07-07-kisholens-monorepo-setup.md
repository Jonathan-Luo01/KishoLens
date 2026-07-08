# KishoLens Monorepo Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the KishoLens monorepo containing Node (Astro + React) frontend and three UV-managed Python packages (`backend`, `ml`, `data_pipeline`).

**Architecture:** Use a dual-workspace structure. The root level defines the Python `uv` workspace to link all Python packages with a shared `uv.lock`. Node/npm manages the frontend in a dedicated workspace directory. A root-level `package.json` configures development launchers.

**Tech Stack:** Astral `uv` (Python tooling), Node/npm (Astro + React), FastAPI (backend).

## Global Constraints
- Node version >= 18
- Python version >= 3.10
- Astral `uv` version >= 0.1.0
- All code modifications require tests or verification commands.

---

### Task 1: Root Workspace Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `package.json`

**Interfaces:**
- Produces: The core workspace definition and run commands for npm and uv.

- [ ] **Step 1: Create the root pyproject.toml for UV workspace**
  Write this file at the workspace root (`C:\Users\kingj\Documents\KishoLens\pyproject.toml`):
  ```toml
  [tool.uv]
  workspace = { members = ["backend", "ml", "data_pipeline"] }
  ```

- [ ] **Step 2: Create the root package.json for runner scripting**
  Write this file at the workspace root (`C:\Users\kingj\Documents\KishoLens\package.json`):
  ```json
  {
    "name": "kisholens-monorepo",
    "version": "0.1.0",
    "private": true,
    "scripts": {
      "dev:frontend": "npm --prefix frontend run dev",
      "dev:backend": "uv run --package backend uvicorn src.main:app --reload",
      "dev:pipeline": "uv run --package data_pipeline python -m src.main",
      "dev": "concurrently \"npm:dev:frontend\" \"npm:dev:backend\""
    },
    "devDependencies": {
      "concurrently": "^8.2.2"
    }
  }
  ```

- [ ] **Step 3: Run root validation**
  Run: `npm install`
  Expected: Installs `concurrently` successfully.

- [ ] **Step 4: Commit root configuration**
  Run:
  ```bash
  git add pyproject.toml package.json package-lock.json
  git commit -m "chore: scaffold root monorepo configuration files"
  ```

---

### Task 2: Scaffold backend Service

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/src/main.py`

- [ ] **Step 1: Create backend/pyproject.toml**
  Write the file:
  ```toml
  [project]
  name = "backend"
  version = "0.1.0"
  description = "FastAPI backend for KishoLens"
  readme = "README.md"
  requires-python = ">=3.10"
  dependencies = [
      "fastapi>=0.100.0",
      "uvicorn>=0.22.0",
      "sqlmodel>=0.0.8",
  ]
  ```

- [ ] **Step 2: Create backend main server script**
  Create the folder `backend/src/` and write `backend/src/main.py`:
  ```python
  from fastapi import FastAPI

  app = FastAPI(title="KishoLens API")

  @app.get("/health")
  def health_check():
      return {"status": "ok", "service": "backend"}
  ```

- [ ] **Step 3: Run uv sync validation**
  Run: `uv sync`
  Expected: UV successfully resolves and locks all backend dependencies in the shared workspace lockfile.

- [ ] **Step 4: Test backend endpoint**
  Start server test: `uv run --package backend uvicorn src.main:app --port 8000` (run in background/test it using curl)
  Check: `curl http://127.0.0.1:8000/health`
  Expected: `{"status":"ok","service":"backend"}`

- [ ] **Step 5: Commit backend**
  Run:
  ```bash
  git add backend/pyproject.toml backend/src/main.py uv.lock
  git commit -m "feat: scaffold backend project structure with FastAPI"
  ```

---

### Task 3: Scaffold data_pipeline Service

**Files:**
- Create: `data_pipeline/pyproject.toml`
- Create: `data_pipeline/src/main.py`

- [ ] **Step 1: Create data_pipeline/pyproject.toml**
  Write the file:
  ```toml
  [project]
  name = "data_pipeline"
  version = "0.1.0"
  description = "Ingestion and Scraper Pipeline for KishoLens"
  readme = "README.md"
  requires-python = ">=3.10"
  dependencies = [
      "httpx>=0.24.0",
      "selectolax>=0.3.12",
      "pydantic>=2.0",
      "sqlmodel>=0.0.8",
  ]
  ```

- [ ] **Step 2: Create data_pipeline main execution script**
  Create the folder `data_pipeline/src/` and write `data_pipeline/src/main.py`:
  ```python
  def main():
      print("KishoLens Ingestion Pipeline Initialized.")

  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 3: Run uv sync validation**
  Run: `uv sync`
  Expected: UV resolves and locks data_pipeline dependencies.

- [ ] **Step 4: Test run pipeline**
  Run: `uv run --package data_pipeline python -m src.main`
  Expected output: `KishoLens Ingestion Pipeline Initialized.`

- [ ] **Step 5: Commit data_pipeline**
  Run:
  ```bash
  git add data_pipeline/pyproject.toml data_pipeline/src/main.py uv.lock
  git commit -m "feat: scaffold data_pipeline project structure"
  ```

---

### Task 4: Scaffold ml Service

**Files:**
- Create: `ml/pyproject.toml`
- Create: `ml/src/main.py`

- [ ] **Step 1: Create ml/pyproject.toml**
  Write the file:
  ```toml
  [project]
  name = "ml"
  version = "0.1.0"
  description = "ML and NLP metric analyzer for KishoLens"
  readme = "README.md"
  requires-python = ">=3.10"
  dependencies = [
      "spacy>=3.5.0",
      "sudachipy>=0.6.7",
      "sudachidict-core>=20230110",
      "nltk>=3.8.1",
      "sqlmodel>=0.0.8",
  ]
  ```

- [ ] **Step 2: Create ml main analyzer script**
  Create the folder `ml/src/` and write `ml/src/main.py`:
  ```python
  def main():
      print("KishoLens ML/NLP Engine Initialized.")

  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 3: Run uv sync validation**
  Run: `uv sync`
  Expected: UV successfully locks all ml dependencies.

- [ ] **Step 4: Test run ML package**
  Run: `uv run --package ml python -m src.main`
  Expected output: `KishoLens ML/NLP Engine Initialized.`

- [ ] **Step 5: Commit ML**
  Run:
  ```bash
  git add ml/pyproject.toml ml/src/main.py uv.lock
  git commit -m "feat: scaffold ml project structure"
  ```

---

### Task 5: Scaffold Astro + React Frontend

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/astro.config.mjs`
- Create: `frontend/src/pages/index.astro`

- [ ] **Step 1: Initialize Astro framework in frontend/**
  We run `npm create astro@latest` in non-interactive mode.
  Run: `npx -y create-astro@latest frontend --template minimal --install --no-git --typescript strict`
  Expected: A complete minimal Astro scaffold is generated in `frontend/`.

- [ ] **Step 2: Add React integration to Astro**
  Inside the `frontend/` directory, run the Astro CLI to add React:
  Run: `npx --yes astro add react` (when prompted to write configurations, approve automatically or verify configuration)
  *Note: Make sure this is run inside C:\Users\kingj\Documents\KishoLens\frontend*

- [ ] **Step 3: Create index.astro page**
  Overwrite `frontend/src/pages/index.astro` with a basic UI layout:
  ```html
  ---
  ---
  <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width" />
      <title>KishoLens - Prose Archetype Analysis</title>
      <style>
        body {
          background-color: #0b0f19;
          color: #f3f4f6;
          font-family: system-ui, sans-serif;
          margin: 0;
          padding: 2rem;
        }
        h1 {
          color: #60a5fa;
        }
      </style>
    </head>
    <body>
      <h1>KishoLens Dashboard</h1>
      <p>Analyze structural metrics of web novels and light fiction.</p>
    </body>
  </html>
  ```

- [ ] **Step 4: Test the frontend Dev server**
  Run: `npm --prefix frontend run dev -- --port 3000` (run and test endpoint)
  Check: `curl http://localhost:3000/`
  Expected: Returns HTML containing `<h1>KishoLens Dashboard</h1>`

- [ ] **Step 5: Commit Frontend**
  Run:
  ```bash
  git add frontend/
  git commit -m "feat: scaffold frontend project structure with Astro + React integration"
  ```
