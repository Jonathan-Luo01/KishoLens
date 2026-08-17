"""
analyzer.py — Dual-Vector + Dynamic Semantic Concept Prose Analyzer for KishoLens.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np

from kisholens.ml.embeddings import generate_dual_vectors
from kisholens.ml.centroids import get_inciting_concept_vectors
from kisholens.ml.semantic_match import _load_with_cache, DEFAULT_DATA_DIR
from kisholens.pipeline.taxonomy import evaluate_epic_cultivation_guardrail, scan_anchor_boosts


def analyze_prose(
    synopsis: Optional[str],
    ch1_text: str,
    ch10_text: Optional[str] = None,
    ch20_text: Optional[str] = None,
    title: Optional[str] = None,
    data_dir: str = DEFAULT_DATA_DIR,
    use_regex_boost: bool = True,
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

    full_scan_text = f"{title or ''} {synopsis or ''} {ch1_text} {ch10_text or ''} {ch20_text or ''}".strip()
    anchor_boosts = scan_anchor_boosts(full_scan_text) if use_regex_boost else {}
    guardrail = evaluate_epic_cultivation_guardrail(full_scan_text) if use_regex_boost else {}

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
        
        # Compute Base Score and Concept Density Score
        raw_intro = float(intro_sims[idx]) + anchor_boosts.get(target_g, 0.0)
        s_base = _calibrate(raw_intro)
        s_concept = float(np.dot(v_intro, concept_vec))

        # Capped inciting event multiplier: require BOTH strong genre centroid
        # alignment (raw_intro > 0.10) AND high concept density (s_concept > 0.30)
        # to prevent standard travel/departure scenes (Moby Dick, Odyssey) from
        # triggering false Isekai/Cultivation detection.
        if raw_intro > 0.10 and s_concept > 0.30:
            dynamic_boost = min(0.15, s_concept * 0.35)
            final_score = round(min(0.99, s_base + dynamic_boost), 4)
        else:
            final_score = round(s_base, 4)

        inciting_results.append((concept_name, final_score))

    inciting_results.sort(key=lambda x: x[1], reverse=True)
    best_inciting_name, best_inciting_score = inciting_results[0]

    # Fallback threshold check (< 0.65) to avoid false positive setup events on standard non-setup novels
    if best_inciting_score < 0.65:
        inciting_payload = None
    else:
        inciting_payload = {"primary": best_inciting_name, "score": best_inciting_score}

    # If Inciting Event is confirmed (e.g. Isekai & Regression), integrate into primary genre predictions
    if inciting_payload:
        inciting_g = "Isekai" if "Isekai" in inciting_payload["primary"] else ("Progression Fantasy" if "System" in inciting_payload["primary"] else "Cultivation")
        sustained_scores[inciting_g] = round(min(0.99, max(0.01, max(sustained_scores.get(inciting_g, 0.0), inciting_payload["score"] + 0.10))), 4)

    # Apply guardrail penalties/boosts for Scenario A (Pure Classical Epic) and Scenario B (Hybrid Cultivation)
    if guardrail.get("scenario") in ("A", "B"):
        inciting_payload = None

    if guardrail.get("scenario") == "A":
        if "Cultivation" in sustained_scores:
            sustained_scores["Cultivation"] = round(max(0.01, sustained_scores["Cultivation"] - 0.40), 4)
        if "Historical" in sustained_scores:
            sustained_scores["Historical"] = round(min(0.99, sustained_scores["Historical"] + 0.20), 4)

    sorted_sustained = [(g, round(min(0.99, max(0.01, s)), 4)) for g, s in sorted(sustained_scores.items(), key=lambda x: x[1], reverse=True)]
    world_gname, world_score = sorted_sustained[0]
    plot_gname, plot_score = sorted_sustained[1] if len(sorted_sustained) > 1 else (world_gname, world_score)

    if guardrail.get("scenario") == "B":
        display_plot_gname = "Cultivation"
        plot_gname = "Historical / Military"
        plot_score = sustained_scores.get("Historical / Military", sustained_scores.get("Historical", plot_score))
    else:
        display_plot_gname = plot_gname

    display_parts = []
    if inciting_payload:
        display_parts.append(inciting_payload["primary"])
    display_parts.append(world_gname)
    if guardrail.get("display_tag"):
        display_parts.append(display_plot_gname)
        display_parts.append(guardrail["display_tag"])
    else:
        display_parts.append(f"({plot_gname})")
    display_label = " ".join(display_parts)

    genre_scores = [{"genre": gname, "score": score, "raw_score": score} for gname, score in sorted_sustained]

    return {
        "inciting_event": inciting_payload,
        "world_setting": {"primary": world_gname, "score": world_score},
        "narrative_plot": {"primary": plot_gname, "score": plot_score},
        "display_label": display_label,
        "genre_scores": genre_scores,
    }

