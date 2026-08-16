"""
embeddings.py — Dual-scope vector generation for KishoLens.
"""

from __future__ import annotations
from typing import Optional, Tuple
import threading
from typing import Optional, Tuple, Dict, Any
import numpy as np

_model_cache: Dict[str, Any] = {}
_model_lock = threading.Lock()


def get_transformer_model(model_name: str = "all-MiniLM-L6-v2"):
    """Thread-safe singleton getter for SentenceTransformer models on CPU."""
    if model_name not in _model_cache:
        with _model_lock:
            if model_name not in _model_cache:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name, device="cpu")
                model.eval()
                _model_cache[model_name] = model
    return _model_cache[model_name]


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
