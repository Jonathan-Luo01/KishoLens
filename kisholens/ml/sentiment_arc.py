import math
import re
import numpy as np
from kisholens.ml.features import compute_sentence_sentiment_normalized

def compute_kishotenketsu_quantile_arc(all_sentences: list[str], lang: str, num_quantiles: int = 4, gain: float = 2.0) -> dict:
    """
    Computes the 4-act Kishōtenketsu sentiment arc across 4 quantiles (Ki, Shō, Ten, Ketsu).
    Utilizes sentiment density weighting (RMS including neutrals) and graceful asymptotic curving (numpy.tanh)
    to span the full -1.0 to 1.0 spectrum without artificial clamping.
    
    Returns a dict with:
      - 'acts': 4 Kishōtenketsu act objects (Ki, Shō, Ten, Ketsu) with sentiment scores [-1.0, 1.0]
      - 'quantiles': list of 4 float sentiment scores [-1.0, 1.0]
      - 'raw_quantiles': raw density-weighted scores before tanh curve
    """
    act_defs = [
        ("Ki",    "Introduction"),
        ("Shō",   "Development"),
        ("Ten",   "Twist"),
        ("Ketsu", "Resolution"),
    ]

    if not all_sentences:
        empty_quantiles = [0.0] * num_quantiles
        acts = [
            {"act": act_defs[i][0], "label": act_defs[i][1], "sentiment": 0.0, "sentence_range": [0, 0]}
            for i in range(min(num_quantiles, len(act_defs)))
        ]
        return {
            "acts": acts,
            "quantiles": empty_quantiles,
            "raw_quantiles": empty_quantiles,
        }

    scored_sentences = [
        compute_sentence_sentiment_normalized(s, lang)
        for s in all_sentences if s and s.strip()
    ]

    if not scored_sentences:
        empty_quantiles = [0.0] * num_quantiles
        acts = [
            {"act": act_defs[i][0], "label": act_defs[i][1], "sentiment": 0.0, "sentence_range": [0, 0]}
            for i in range(min(num_quantiles, len(act_defs)))
        ]
        return {
            "acts": acts,
            "quantiles": empty_quantiles,
            "raw_quantiles": empty_quantiles,
        }

    n = len(scored_sentences)
    quantile_scores = []
    raw_scores = []
    acts = []

    for k in range(num_quantiles):
        start = int(k * n / num_quantiles)
        end = int((k + 1) * n / num_quantiles)
        if start >= end:
            end = min(n, start + 1)

        chunk = scored_sentences[start:end]
        if not chunk:
            quantile_scores.append(0.0)
            raw_scores.append(0.0)
            if k < len(act_defs):
                acts.append({
                    "act": act_defs[k][0],
                    "label": act_defs[k][1],
                    "sentiment": 0.0,
                    "sentence_range": [start, max(start, end - 1)],
                })
            continue

        arr = np.array(chunk, dtype=float)
        # 1. Raw mean of the entire quantile including neutrals
        raw_mean = float(np.mean(arr))
        
        # 2. Root Mean Squared (RMS) over the entire quantile including neutrals
        rms = float(np.sqrt(np.mean(arr ** 2)))
        
        # 3. Sentiment density ratio (proportion of non-neutral sentences)
        density = float(np.mean(np.abs(arr) > 0.05))

        # Sentiment density weighting: directional polarity * RMS magnitude * density factor
        sign = 1.0 if raw_mean >= 0 else -1.0
        magnitude = math.sqrt(abs(raw_mean) * (rms + 1e-6))
        raw_weighted = sign * magnitude * (1.0 + 0.5 * density)
        raw_scores.append(round(raw_weighted, 4))

        # Graceful asymptotic scaling via np.tanh()
        curved_score = float(np.tanh(gain * raw_weighted))
        final_score = round(curved_score, 4)
        quantile_scores.append(final_score)

        if k < len(act_defs):
            acts.append({
                "act": act_defs[k][0],
                "label": act_defs[k][1],
                "sentiment": final_score,
                "sentence_range": [start, max(start, end - 1)],
            })

    return {
        "acts": acts,
        "quantiles": quantile_scores,
        "raw_quantiles": raw_scores,
    }
