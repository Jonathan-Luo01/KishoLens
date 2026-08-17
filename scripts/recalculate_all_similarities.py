#!/usr/bin/env python3
"""
scripts/recalculate_all_similarities.py
Batch-recomputes top 5 similar works for all 10,320 novels in data/stats_cache.json
using the story-dominant (85% Story / 15% Style) similarity engine and rich semantic embeddings.
"""

import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kisholens.ml.similarity import (
    extract_feature_vector,
    _extract_metric_values,
    _compute_metric_comparisons,
    _compute_match_badges,
    _extract_genre_list,
    _infer_query_anatomy,
    _generate_narrative_synthesis,
    _compute_4pillar_breakdown,
    _extract_shared_tropes,
    SIMILARITY_MODEL_VERSION
)
from kisholens.ml.embeddings import get_transformer_model


def build_concept_text(item: dict) -> str:
    title = (item.get("title") or "").strip()
    tax = item.get("taxonomy") if isinstance(item.get("taxonomy"), dict) else {}
    inciting = tax.get("inciting_event", {}).get("primary", "") if isinstance(tax.get("inciting_event"), dict) else ""
    world = tax.get("world_setting", {}).get("primary", "") if isinstance(tax.get("world_setting"), dict) else ""
    plot = tax.get("narrative_plot", {}).get("primary", "") if isinstance(tax.get("narrative_plot"), dict) else ""
    genre = (item.get("primary_genre") or item.get("genre") or "").strip()
    tags = (item.get("tags") or "").strip()
    territory = (item.get("territory") or "").strip()

    parts = []
    if title and title != "Unknown Title":
        parts.append(title)
    if world:
        parts.append(f"Setting: {world}")
    elif genre:
        parts.append(f"Genre: {genre}")
    if inciting:
        parts.append(f"Catalyst: {inciting}")
    if plot:
        parts.append(f"Plot: {plot}")
    if tags:
        parts.append(f"Tropes: {tags}")
    if territory:
        parts.append(f"Tradition: {territory}")

    return ". ".join(parts) if parts else "fiction"


def main():
    cache_path = PROJECT_ROOT / "data" / "stats_cache.json"
    if not cache_path.exists():
        print(f"Error: {cache_path} does not exist.")
        sys.exit(1)

    print("Loading data/stats_cache.json...")
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    novel_ids = []
    novel_items = []

    for k, item in data.items():
        if k.startswith("_") or not isinstance(item, dict):
            continue
        try:
            nid = int(k)
            novel_ids.append(nid)
            novel_items.append(item)
        except (ValueError, TypeError):
            continue

    num_novels = len(novel_ids)
    print(f"Loaded {num_novels} novels from disk cache.")

    # 1. Prepare concept texts
    print("Building concept texts for batch transformer embedding...")
    concept_texts = [build_concept_text(item) for item in novel_items]

    # 2. Batch encode all 10,320 concept texts
    print("Encoding concept texts using SentenceTransformer (all-MiniLM-L6-v2)...")
    t0 = time.time()
    model = get_transformer_model()
    concept_matrix = model.encode(
        concept_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    t_emb = time.time() - t0
    print(f"Batch embedding finished in {t_emb:.2f}s (matrix shape: {concept_matrix.shape}).")

    # 3. Extract 8D style radar vectors
    print("Extracting 8D style vectors...")
    style_vectors = np.array([extract_feature_vector(item) for item in novel_items], dtype=np.float32)
    style_norms = np.linalg.norm(style_vectors, axis=1, keepdims=True)
    style_norms[style_norms == 0] = 1.0
    style_norm_matrix = style_vectors / style_norms

    # 4. Extract genres, tags, territories, and titles
    print("Extracting genres, tags, and metadata...")
    all_genres = []
    all_primary_genres = []
    all_tags = []
    all_territories = []
    all_titles = []
    all_authors = []
    all_metrics = []
    all_anatomies = []

    for i, item in enumerate(novel_items):
        g_names = (
            _extract_genre_list(item.get("top_genres"))
            + _extract_genre_list(item.get("primary_genre"))
            + _extract_genre_list(item.get("genre"))
        )
        all_genres.append(set(g.lower().strip() for g in g_names if g and g.strip()))
        p_genre = (item.get("primary_genre") or item.get("genre") or "").lower().strip()
        all_primary_genres.append(p_genre)

        raw_tags = item.get("tags") or ""
        all_tags.append(set(t.strip().lower() for t in raw_tags.split(",") if t.strip()))
        all_territories.append((item.get("territory") or "Unknown").strip())
        all_titles.append(item.get("title") or "Unknown Title")
        all_authors.append(item.get("author") or "Unknown Author")
        all_metrics.append(_extract_metric_values(item, style_vectors[i]))
        all_anatomies.append(_infer_query_anatomy(None, item.get("semantic"), item))

    # 5. Build Fast Vectorized Primary Genre & Territory Match Matrices
    print("Building vectorized genre and territory affinity matrices...")
    # Primary genre exact match matrix (N, N)
    pgenre_ids = np.array([hash(p) % 1000000 if p else 0 for p in all_primary_genres], dtype=np.int64)
    pgenre_match = (pgenre_ids[:, None] == pgenre_ids[None, :]) & (pgenre_ids[:, None] != 0) # (N, N)

    # Territory exact match matrix (N, N)
    terr_ids = np.array([hash(t.lower()) % 1000000 if t != "Unknown" else 0 for t in all_territories], dtype=np.int64)
    terr_match = (terr_ids[:, None] == terr_ids[None, :]) & (terr_ids[:, None] != 0) # (N, N)

    # 6. Compute Pairwise Similarities in Vectorized Chunks
    print("Computing pairwise story-dominant similarities across all 10,320 novels in vectorized chunks...")
    t1 = time.time()
    chunk_size = 1000

    for start_idx in range(0, num_novels, chunk_size):
        end_idx = min(start_idx + chunk_size, num_novels)
        C = end_idx - start_idx

        q_concepts = concept_matrix[start_idx:end_idx] # (C, 384)
        q_styles = style_norm_matrix[start_idx:end_idx] # (C, 8)
        q_raw_styles = style_vectors[start_idx:end_idx] # (C, 8)

        # 1. Semantic Cosine Matrix (C, N)
        sem_chunk = np.clip(np.dot(q_concepts, concept_matrix.T), 0.0, 1.0) # (C, N)

        # 2. Style Cosine Matrix (C, N)
        cos_style_chunk = np.clip(np.dot(q_styles, style_norm_matrix.T), 0.0, 1.0) # (C, N)

        # 3. Style L1 Difference (C, N)
        # |q - cand| across 8 dimensions
        l1_diff_chunk = np.mean(np.abs(q_raw_styles[:, None, :] - style_vectors[None, :, :]), axis=2) # (C, N)
        style_sim_chunk = np.clip(0.5 * cos_style_chunk + 0.5 * np.maximum(0.0, 1.0 - 4.0 * l1_diff_chunk), 0.0, 1.0)

        # 4. Genre Sim Chunk (C, N)
        pgenre_chunk = pgenre_match[start_idx:end_idx] # (C, N)
        genre_sim_chunk = np.where(pgenre_chunk, 0.90, 0.40).astype(np.float32)

        # 5. Territory Sim Chunk (C, N)
        terr_chunk = terr_match[start_idx:end_idx] # (C, N)
        terr_sim_chunk = np.where(terr_chunk, 1.0, 0.50).astype(np.float32)

        # 6. Story Similarity Chunk (C, N)
        # 45% Semantic + 35% Genre + 15% Tags (approximated via genre) + 5% Territory
        tag_sim_chunk = genre_sim_chunk * 0.80
        story_sim_chunk = np.clip(
            0.45 * sem_chunk + 0.35 * genre_sim_chunk + 0.15 * tag_sim_chunk + 0.05 * terr_sim_chunk,
            0.0,
            1.0
        )

        # 7. Composite Score Chunk (C, N): 85% Story + 15% Style
        comp_chunk = np.clip(0.85 * story_sim_chunk + 0.15 * style_sim_chunk, 0.01, 0.99)

        # Zero out self-matches
        for local_i in range(C):
            global_i = start_idx + local_i
            comp_chunk[local_i, global_i] = -1.0

        # Find top 6 candidate indices per row via argpartition
        top_k_indices = np.argpartition(-comp_chunk, 6, axis=1)[:, :6]

        # Format Top 5 Matches for each query novel
        for local_i in range(C):
            global_i = start_idx + local_i
            qid = novel_ids[global_i]
            q_pgenre = all_primary_genres[global_i]
            q_m = all_metrics[global_i]

            row_top_indices = top_k_indices[local_i]
            # Sort the 6 candidates by exact composite score descending
            sorted_indices = sorted(row_top_indices, key=lambda idx: comp_chunk[local_i, idx], reverse=True)[:5]

            formatted_matches = []
            for cand_j in sorted_indices:
                cid = novel_ids[cand_j]
                c_item = novel_items[cand_j]
                c_score = round(float(comp_chunk[local_i, cand_j]), 4)
                st_sim = float(story_sim_chunk[local_i, cand_j])
                sty_sim = float(style_sim_chunk[local_i, cand_j])
                s_sim = float(sem_chunk[local_i, cand_j])
                g_sim = float(genre_sim_chunk[local_i, cand_j])
                t_sim = float(tag_sim_chunk[local_i, cand_j])
                terr_sim = float(terr_sim_chunk[local_i, cand_j])

                c_meta = {
                    "id": cid,
                    "title": all_titles[cand_j],
                    "author": all_authors[cand_j],
                    "genre": c_item.get("genre", ""),
                    "primary_genre": c_item.get("primary_genre") or c_item.get("genre") or "Fiction",
                    "territory": all_territories[cand_j],
                    "tags": c_item.get("tags", ""),
                    "top_genres": c_item.get("top_genres", []),
                    "taxonomy": c_item.get("taxonomy") or {}
                }

                # Reasons
                story_reasons = []
                style_reasons = []

                cand_primary = (c_meta["primary_genre"]).lower().strip()
                if q_pgenre and cand_primary and q_pgenre == cand_primary:
                    cand_pclean = c_item.get("primary_genre") or c_item.get("genre")
                    story_reasons.append(f"Matching primary archetype: {cand_pclean}")
                elif q_pgenre and q_pgenre in all_genres[cand_j]:
                    cand_pclean = novel_items[global_i].get("primary_genre") or novel_items[global_i].get("genre")
                    story_reasons.append(f"Shared genre: {cand_pclean}")
                elif g_sim >= 0.60:
                    story_reasons.append("Strong genre overlap")

                if s_sim >= 0.80:
                    story_reasons.append("Closely aligned plot premise & themes")
                elif s_sim >= 0.65:
                    story_reasons.append("Thematic narrative overlap")

                if t_sim >= 0.60:
                    story_reasons.append("Overlapping narrative tropes")

                if terr_sim >= 0.85 and c_meta["territory"] != "Unknown":
                    if "classic" in c_meta["territory"].lower():
                        story_reasons.append("Shared Classic Literature tradition")
                    elif "web" in c_meta["territory"].lower():
                        story_reasons.append("Shared Web Novel territory")

                if sty_sim >= 0.88:
                    style_reasons.append("Similar prose style & sentence structure")
                elif sty_sim >= 0.75:
                    style_reasons.append("Comparable sentence cadence")

                reasons = story_reasons.copy()
                if style_reasons:
                    reasons.extend(style_reasons)
                if not reasons:
                    reasons.append("Overall thematic and stylistic affinity")

                c_metrics = all_metrics[cand_j]
                metric_comparisons = _compute_metric_comparisons(q_m, c_metrics)
                match_badges = _compute_match_badges(
                    q_metrics=q_m,
                    c_metrics=c_metrics,
                    query_text=None,
                    query_semantic=novel_items[global_i].get("taxonomy"),
                    query_features=novel_items[global_i],
                    cand_meta=c_meta,
                    cand_primary_genre=c_meta["primary_genre"],
                    style_sim=sty_sim,
                    score=c_score
                )

                q_anat = all_anatomies[global_i]
                c_anat = all_anatomies[cand_j]
                narrative_synthesis = _generate_narrative_synthesis(q_anat, c_anat, st_sim, g_sim, is_user_input=False)
                pillars = _compute_4pillar_breakdown(q_anat, c_anat, q_m, c_metrics, st_sim, g_sim, sty_sim)
                shared_tropes = _extract_shared_tropes(q_anat, c_anat)
                narrative_reasoning = {
                    "narrative_synthesis": narrative_synthesis,
                    "pillars": pillars,
                    "shared_tropes": shared_tropes
                }

                formatted_matches.append({
                    "id": cid,
                    "title": all_titles[cand_j],
                    "author": all_authors[cand_j],
                    "genre": c_item.get("genre", ""),
                    "territory": all_territories[cand_j],
                    "similarity_score": c_score,
                    "story_similarity": int(round(st_sim * 100)),
                    "style_similarity": int(round(sty_sim * 100)),
                    "match_badges": match_badges,
                    "metric_comparisons": metric_comparisons,
                    "reasons": reasons,
                    "story_reasons": story_reasons,
                    "style_reasons": style_reasons,
                    "narrative_reasoning": narrative_reasoning,
                    "breakdown": {
                        "story": round(st_sim, 3),
                        "style": round(sty_sim, 3),
                        "semantic": round(s_sim, 3),
                        "genre": round(g_sim, 3),
                        "tags": round(t_sim, 3),
                        "territory": round(terr_sim, 3),
                    }
                })

            data[str(qid)]["top_matches"] = formatted_matches
            data[str(qid)]["similarity_version"] = SIMILARITY_MODEL_VERSION

        print(f"Processed {end_idx}/{num_novels} novels...")

    t_all = time.time() - t1
    print(f"Pairwise similarity recomputation finished in {t_all:.2f}s.")

    # 6. Save back to disk cache atomically
    print("Saving updated matches to data/stats_cache.json...")
    temp_path = cache_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    temp_path.replace(cache_path)
    print(f"Successfully updated all {num_novels} novels in {cache_path} with {SIMILARITY_MODEL_VERSION}!")


if __name__ == "__main__":
    main()
