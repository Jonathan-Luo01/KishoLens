"""
embeddings.py — Dual-scope vector generation for KishoLens.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

_model = None


def get_transformer_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _model


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0 or np.isnan(norm):
        return vec
    return (vec / norm).astype(np.float32)


def embed_single_text(text: str) -> np.ndarray:
    if not text or not text.strip():
        return np.zeros(384, dtype=np.float32)
    model = get_transformer_model()
    vec = model.encode(text.strip(), convert_to_numpy=True)
    return _normalize(vec.astype(np.float32))


def generate_dual_vectors(
    synopsis: Optional[str],
    ch1_text: str,
    ch10_text: Optional[str] = None,
    ch20_text: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    v_syn = embed_single_text(synopsis) if synopsis and synopsis.strip() else None
    v_ch1 = embed_single_text(ch1_text) if ch1_text and ch1_text.strip() else np.zeros(384, dtype=np.float32)
    v_ch10 = embed_single_text(ch10_text) if ch10_text and ch10_text.strip() else v_ch1
    v_ch20 = embed_single_text(ch20_text) if ch20_text and ch20_text.strip() else v_ch10

    if v_syn is not None:
        v_intro = 0.60 * v_syn + 0.40 * v_ch1
        v_sustained = 0.10 * v_syn + 0.10 * v_ch1 + 0.40 * v_ch10 + 0.40 * v_ch20
    else:
        v_intro = 1.0 * v_ch1
        v_sustained = 0.20 * v_ch1 + 0.40 * v_ch10 + 0.40 * v_ch20

    return _normalize(v_intro), _normalize(v_sustained)
