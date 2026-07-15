# Baseline Comparison Feature Design Spec

This spec details the integration of baseline metrics (Classic Literature vs Web Novels) in the custom text analysis feature of KishoLens.

## 1. Objectives
- Enable users to compare their custom prose style metrics directly against aggregated averages for Classic Literature (Project Gutenberg) and Web Novels (RoyalRoad, ScribbleHub, etc.).
- Maintain real-time performance and prevent redundant expensive NLP calculations by caching the database averages.
- Provide clean, highly readable indicators on the frontend `/analyze` dashboard under the Vocabulary Richness, Dialogue Ratio, and Average Sentence Length metric cards.

## 2. Backend Design (`kisholens/api/main.py`)
- We will cache the calculated baseline averages inside a module-level variable `_cached_baselines`.
- A helper `get_baseline_stats()` will compute the averages once from all chapters in the database:
  - If a chapter's source novel is `"gutenberg"`, it is aggregated into classic lit.
  - Otherwise (sources `"royalroad"`, `"scribblehub"`, `"syosetu"`, `"cnnovel"`), it is aggregated into web novel metrics.
- In case of failure or empty database, standard pre-computed fallbacks will be used:
  - Gutenberg: `ttr=0.173`, `dialogue_ratio=0.182`, `avg_sentence_len=12.464`
  - Web Novel: `ttr=0.306`, `dialogue_ratio=0.235`, `avg_sentence_len=12.699`
- The `POST /api/analyze` payload will return the `baselines` dictionary containing `"gutenberg"` and `"webnovel"` averages for `ttr`, `dialogue_ratio`, and `avg_sentence_len`.

## 3. Frontend Design (`frontend/src/pages/analyze.astro`)
- Add a CSS class `.baseline-comparison` styling comparative labels under the progress bars in metric cards.
- Update the HTML for the three target cards:
  - **Vocabulary Richness**: Add labels `#ttr-gutenberg-val` and `#ttr-webnovel-val`.
  - **Dialogue Ratio**: Add labels `#dialogue-gutenberg-val` and `#dialogue-webnovel-val`.
  - **Avg Sentence Length**: Add labels `#sent-gutenberg-val` and `#sent-webnovel-val`.
- Update `renderResults(data)` to format and output the baseline figures dynamically.

## 4. Verification Plan
- **Backend API**: Send a POST curl request to `/api/analyze` with sample text and verify the `baselines` block is returned with correct values.
- **Frontend Page**: Build and run, paste sample text, execute analysis, and verify that baseline labels display the computed values correctly.
