"""
embeddings.py — Dual-scope vector generation for KishoLens with Hugging Face Inference API support.
"""

from __future__ import annotations
import os
import threading
from typing import Optional, Tuple, Dict, Any
import numpy as np

_model_cache: Dict[str, Any] = {}
_model_lock = threading.Lock()

HF_INFERENCE_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"


def get_transformer_model(model_name: str = "all-MiniLM-L6-v2"):
    """Thread-safe singleton getter for SentenceTransformer models on CPU."""
    if model_name not in _model_cache:
        with _model_lock:
            if model_name not in _model_cache:
                import torch
                torch.set_num_threads(1)
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


def _embed_via_huggingface(text: str, token: str) -> Optional[np.ndarray]:
    """Call Hugging Face Serverless Inference API for feature extraction."""
    try:
        import httpx
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "inputs": text.strip()[:2000],
            "options": {"wait_for_model": True}
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(HF_INFERENCE_URL, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list):
                        vec = np.array(data[0], dtype=np.float32)
                    else:
                        vec = np.array(data, dtype=np.float32)
                    if vec.shape == (384,):
                        return _normalize(vec)
            else:
                print(f"[WARN] HF Inference returned status {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        print(f"[WARN] Hugging Face Inference API failed: {e}. Falling back to local SentenceTransformer.")
    return None


def embed_single_text(text: str) -> np.ndarray:
    if not text or not text.strip():
        return np.zeros(384, dtype=np.float32)
    
    # 1. Check if Hugging Face API token is provided in environment
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        hf_vec = _embed_via_huggingface(text, hf_token)
        if hf_vec is not None:
            return hf_vec

    # 2. Fall back to local SentenceTransformer model
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
