"""
analyzer.py — Dual-Vector + Dynamic Semantic Concept Prose Analyzer for KishoLens.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np

from kisholens.ml.embeddings import generate_dual_vectors
from kisholens.ml.centroids import get_inciting_concept_vectors
from kisholens.ml.semantic_match import _load_with_cache, DEFAULT_DATA_DIR, scan_anchor_boosts


def analyze_prose(
    synopsis: Optional[str],
    ch1_text: str,
    ch10_text: Optional[str] = None,
    ch20_text: Optional[str] = None,
    title: Optional[str] = None,
    data_dir: str = DEFAULT_DATA_DIR,
) -> Dict[str, Any]:
    v_intro, v_sustained = generate_dual_vectors(synopsis, ch1_text, ch10_text, ch20_text)
    g_centroids, g_meta, t_centroids, t_meta = _load_with_cache(data_dir)

    if g_centroids is None or g_meta is None:
        return {}

    genres = g_meta["genres"]
    g_norms = np.linalg.norm(g_centroids, axis=1, keepdims=True)
    g_safe_norms = np.where(g_norms == 0, 1.0, g_norms)
    norm_g_centroids = g_centroids / g_safe_norms

    g_mean = norm_g_centroids.mean(axis=0, keepdims=True)
    g_sub = norm_g_centroids - g_mean

    sustained_sims = np.dot(v_sustained, g_sub.T)
    sustained_sims = sustained_sims - np.mean(sustained_sims)

    intro_sims = np.dot(v_intro, g_sub.T)
    intro_sims = intro_sims - np.mean(intro_sims)

    full_scan_text = f"{title or ''} {synopsis or ''} {ch1_text}".strip()
    anchor_boosts = scan_anchor_boosts(full_scan_text)

    def _calibrate(s: float, k: float = 5.5) -> float:
        return float(1.0 / (1.0 + np.exp(-k * s)))

    sustained_scores: Dict[str, float] = {}
    for i, gname in enumerate(genres):
        raw = float(sustained_sims[i])
        boosted = raw + anchor_boosts.get(gname, 0.0)
        sustained_scores[gname] = round(_calibrate(boosted), 4)

    concept_vecs = get_inciting_concept_vectors()
    inciting_results = []

    for concept_name, concept_vec in concept_vecs.items():
        if "Isekai" in concept_name:
            target_g = "Isekai"
        elif "System" in concept_name:
            target_g = "Progression Fantasy"
        else:
            target_g = "Cultivation"

        idx = genres.index(target_g) if target_g in genres else 0
        s_base = _calibrate(float(intro_sims[idx]) + anchor_boosts.get(target_g, 0.0))

        s_concept = float(np.dot(v_intro, concept_vec))

        if s_concept > 0.20:
            dynamic_boost = min(0.25, s_concept * 0.50)
            final_score = round(min(0.99, s_base + dynamic_boost), 4)
        else:
            final_score = round(s_base, 4)

        inciting_results.append((concept_name, final_score))

    inciting_results.sort(key=lambda x: x[1], reverse=True)
    best_inciting_name, best_inciting_score = inciting_results[0]

    if best_inciting_score < 0.55:
        inciting_payload = None
    else:
        inciting_payload = {"primary": best_inciting_name, "score": best_inciting_score}

    sorted_sustained = sorted(sustained_scores.items(), key=lambda x: x[1], reverse=True)
    world_gname, world_score = sorted_sustained[0]
    plot_gname, plot_score = sorted_sustained[1] if len(sorted_sustained) > 1 else (world_gname, world_score)

    display_parts = []
    if inciting_payload:
        display_parts.append(inciting_payload["primary"])
    display_parts.append(world_gname)
    display_parts.append(f"({plot_gname})")
    display_label = " ".join(display_parts)

    return {
        "inciting_event": inciting_payload,
        "world_setting": {"primary": world_gname, "score": world_score},
        "narrative_plot": {"primary": plot_gname, "score": plot_score},
        "display_label": display_label,
    }
