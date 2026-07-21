# pynvml Deprecation Warning Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cleanly override the PyPI package `pynvml` version resolution in `uv` to use `nvidia-ml-py`'s `pynvml` module, completely silencing the deprecation `FutureWarning`.

**Architecture:** Use `uv`'s `override-dependencies` in `pyproject.toml` to block the installation of the deprecated wrapper package `pynvml` on all platforms except a dummy one. At the same time, ensure `nvidia-ml-py` is installed in the virtual environment to serve as the runtime provider for the `pynvml` module.

**Tech Stack:** Python 3.14, `uv` package manager.

## Global Constraints
*   Do not commit changes to Git without explicit directive.
*   Preserve all existing imports and functionality for spacy, hanlp, and nltk.

---

### Task 1: Override pynvml dependency in pyproject.toml

**Files:**
- Modify: `pyproject.toml`
- Test: `scratch/test_pynvml_warning.py`

**Interfaces:**
- Consumes: None
- Produces: Bypassed `pynvml` PyPI package installation, clean import from `nvidia-ml-py`

- [ ] **Step 1: Write the verification test**
  Create a temporary scratch script `/Users/jonathan/Documents/KishoLens/scratch/test_pynvml_warning.py` containing:
  ```python
  import warnings
  import sys

  # Verify that we can import the pynvml module
  try:
      with warnings.catch_warnings(record=True) as w:
          warnings.simplefilter("always")
          import pynvml
          print("Successfully imported pynvml module.")
          
          # Check if any FutureWarning from pynvml was captured
          pynvml_warnings = [
              warn for warn in w 
              if issubclass(warn.category, FutureWarning) and "pynvml" in str(warn.message)
          ]
          if len(pynvml_warnings) > 0:
              print(f"Warning found: {pynvml_warnings[0].message}")
              sys.exit(1)
          else:
              print("No pynvml deprecation warnings detected.")
              sys.exit(0)
  except ImportError as e:
      print(f"ImportError: {e}")
      sys.exit(2)
  ```

- [ ] **Step 2: Run test to verify it fails (warning is present)**
  Run: `node run-venv.js python scratch/test_pynvml_warning.py`
  Expected: Outputs `Warning found: The pynvml package is deprecated. Please install nvidia-ml-py instead.` and exits with code 1.

- [ ] **Step 3: Modify pyproject.toml to override pynvml**
  Add the following block to the end of `/Users/jonathan/Documents/KishoLens/pyproject.toml`:
  ```toml
  [tool.uv]
  override-dependencies = [
      "pynvml; sys_platform == 'never'"
  ]
  ```

- [ ] **Step 4: Sync python dependencies**
  Apply the dependency override to the virtual environment by running:
  `uv sync --extra nlp`

- [ ] **Step 5: Run test to verify it passes (warning is gone, import still works)**
  Run: `node run-venv.js python scratch/test_pynvml_warning.py`
  Expected: Outputs `Successfully imported pynvml module.` and `No pynvml deprecation warnings detected.` and exits with code 0.

- [ ] **Step 6: Remove the scratch verification script**
  Delete `/Users/jonathan/Documents/KishoLens/scratch/test_pynvml_warning.py`.
