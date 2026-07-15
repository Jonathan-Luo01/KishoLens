# Custom Text Analysis Design Spec

This document details the design and architecture for adding a custom text analysis feature to KishoLens, enabling users to paste writing samples (chapters) and view stylistic analysis.

## 1. Objectives & Requirements
- Provide a dedicated, highly aesthetic page at `/analyze` with a split-pane layout.
- Accept writing paste inputs of arbitrary length (support English, Japanese, and Chinese).
- Allow users to explicitly choose the language or choose "Auto-detect".
- Compute and display stylistic metrics (dialogue ratio, vocabulary richness, sentence complexity) and match the prose to a style archetype.
- Maintain premium look and feel with responsive grids, micro-animations, and glassmorphism.

## 2. Backend Design
We will introduce a new POST endpoint in `kisholens/api/main.py` at `/api/analyze`.

### Request Schema
```python
from pydantic import BaseModel

class AnalysisRequest(BaseModel):
    text: str
    lang: str = "auto"  # "auto", "en", "ja", "zh"
    title: str = "Untitled"
```

### Language Detection
If the request specifies `lang == "auto"`, the backend detects the language using a simple character-range pattern match:
```python
import re

def detect_language(text: str) -> str:
    # 1. Japanese Hiragana/Katakana characters
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
        return "ja"
    # 2. Chinese (Han) characters
    if re.search(r'[\u4e00-\u9fff]', text):
        return "zh"
    # 3. Default to English
    return "en"
```

### Feature Extraction and Archetype Matching
Once the language is resolved:
1. Extract features using the language-specific extractor from `kisholens.ml.features`:
   - `en` $\rightarrow$ `extract_english_features(text)`
   - `ja` $\rightarrow$ `extract_japanese_features(text)`
   - `zh` $\rightarrow$ `extract_chinese_features(text)`
2. Structure the returned features with a language prefix (e.g. `{f"{lang}_{k}": v for k, v in features.items()}`) to align with `match_archetype` signature.
3. Call `match_archetype(aggregated_features)` to identify the stylistic archetype.
4. Return:
```json
{
  "status": "success",
  "detected_lang": "en",
  "features": { ... },
  "archetype": {
    "archetype": "Descriptive/Slow-paced",
    "confidence": 0.85,
    "description": "Slow, highly descriptive exposition..."
  }
}
```

## 3. Frontend UI Design
We will add a new page [analyze.astro](file:///Users/jonathan/Documents/KishoLens/frontend/src/pages/analyze.astro).

### Layout & Theme
- Glassmorphic panels using transparent backgrounds `rgba(255, 255, 255, 0.02)` and border colors `var(--color-border)`.
- Responsive double-column layout (stacks vertically on smaller screens).
- Responsive sidebar navigation to return to the home page or library.

### Left Pane: Text Input
- Optional Title Input.
- Dropdown language selector.
- Large `textarea` with character and word count tracking.
- CTA Button: "Analyze Prose" with pulsing gradient style and active spinner during load.

### Right Pane: Dynamic States
- **Empty State**: Centered text inviting users to paste content.
- **Loading State**: Shimmering/pulsing skeleton frames.
- **Success State**:
  - Main Archetype banner with calculated match percentage.
  - Interactive grid cards containing descriptive indicators:
    - Sentence Count & Average length.
    - Dialogue Ratio bar indicator.
    - Vocabulary richness / diversity gauge (Type-Token Ratio).
    - Part-of-speech ratios (adjective-to-verb, pronoun density, etc.).

## 4. Verification Plan
- **Backend API**: Send a curl request to `/api/analyze` with custom text in all 3 supported languages, verifying successful response and proper language detection.
- **Frontend App**: Navigate to `/analyze`, paste custom text, trigger analysis, and verify correct rendering of the success metrics and archetype.
