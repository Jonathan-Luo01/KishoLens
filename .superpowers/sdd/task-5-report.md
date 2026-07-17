# Task 5 Report: npm Script + Build Centroids Smoke Test

## Steps Taken

### Step 1: Add npm script
- Added `"dev:build-centroids": "node run-venv.js python -m kisholens.ml.build_centroids"` to `package.json` under `"scripts"`.

### Step 2: Verify CLI help
- Ran `npm run dev:build-centroids -- --help` and verified the output options.

### Step 3: Run centroid build smoke test
- Ran `npm run dev:build-centroids -- --samples 5 --data-dir data` and verified that the offline builder runs successfully, downloads the model, pulls streaming records from HF, and falls back gracefully to zero vectors for failed topic/dataset retrieves.

### Step 4: Verify built files
- Verified that `data/genre_centroids.npy` has shape `(9, 384)` and metadata is stored in `data/genre_centroids_meta.json`.

### Step 5: Test API
- Started the FastAPI server and queried `POST /api/analyze` using `curl`.
- Verified that `"semantic"` key is returned and contains correctly structured scores sorted descending:
  - Genre: `Isekai`
  - Territory: `Web Novel Territory`
  - Top 3 matches present and matching the expected scores list.
