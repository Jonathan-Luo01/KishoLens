# Spec: pynvml Deprecation Warning Fix

## Overview
PyTorch and HanLP trigger a `FutureWarning` deprecation warning when importing the `pynvml` package on macOS. This spec outlines the design for overriding `pynvml` package resolution in `uv` to use `nvidia-ml-py` exclusively, which suppresses the warning while maintaining runtime module compatibility.

---

## 1. Objectives & Requirements
*   **Silence warnings**: Ensure that `import pynvml` (triggered by PyTorch and HanLP) does not raise a `FutureWarning`.
*   **Runtime compatibility**: The `pynvml` module must remain importable and functional.
*   **Clean resolution**: Use `uv` configuration to resolve the package transitive dependency cleanly.

---

## 2. Design Details

### 2.1 UV Configuration Override
We will add the following resolution override in `pyproject.toml`:

```toml
[tool.uv]
override-dependencies = [
    "pynvml; sys_platform == 'never'"
]
```

This tells the `uv` resolver that the package `pynvml` should only resolve to itself on platform `'never'` (effectively disabling its installation).

### 2.2 Dependency Structure
*   `nvidia-ml-py` remains installed via the `[project.optional-dependencies]` `nlp` section.
*   `nvidia-ml-py` places `pynvml.py` directly under `site-packages`.
*   PyTorch/HanLP's code `import pynvml` resolves successfully to `site-packages/pynvml.py` provided by `nvidia-ml-py`.

---

## 3. Implementation Plan
1. Add `[tool.uv]` table and `override-dependencies` in `pyproject.toml`.
2. Run `uv sync --extra nlp` to apply the override and rebuild `uv.lock` and `.venv`.
3. Verify that `pynvml` redirects are removed and that imports still succeed without warnings.
