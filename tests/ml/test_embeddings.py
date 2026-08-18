from unittest.mock import MagicMock
import numpy as np
import pytest
from kisholens.ml.embeddings import embed_single_text, generate_dual_vectors


def test_embed_single_text():
    text = "The protagonist woke up in an unfamiliar fantasy world."
    vec = embed_single_text(text)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert pytest.approx(np.linalg.norm(vec), abs=1e-3) == 1.0


def test_generate_dual_vectors_with_synopsis():
    synopsis = "A high school student dies in a truck accident and wakes up as an imperial prince."
    ch1 = "I opened my eyes to a grand canopy bed inside the Thirteenth Prince Palace."
    ch10 = "I was swinging my sword in the yard and having tea with the court knight."
    ch20 = "The demon sword awakened as the palace guards watched in awe."

    v_intro, v_sustained = generate_dual_vectors(synopsis, ch1, ch10, ch20)
    assert v_intro.shape == (384,)
    assert v_sustained.shape == (384,)
    assert pytest.approx(np.linalg.norm(v_intro), abs=1e-3) == 1.0
    assert pytest.approx(np.linalg.norm(v_sustained), abs=1e-3) == 1.0


def test_generate_dual_vectors_without_synopsis():
    ch1 = "I opened my eyes to a grand canopy bed inside the Thirteenth Prince Palace."
    ch10 = "I was swinging my sword in the yard."
    ch20 = "The demon sword awakened."

    v_intro, v_sustained = generate_dual_vectors(None, ch1, ch10, ch20)
    assert v_intro.shape == (384,)
    assert v_sustained.shape == (384,)


def test_embed_via_huggingface_mock(monkeypatch):
    import os
    from unittest.mock import MagicMock
    import httpx
    from kisholens.ml import embeddings

    mock_vec = [0.1] * 384
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_vec

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp

    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "hf_test_mock_token")
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: mock_client)

    vec = embeddings.embed_single_text("Testing Hugging Face API call")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert pytest.approx(np.linalg.norm(vec), abs=1e-3) == 1.0


def test_embed_via_huggingface_fallback_on_error(monkeypatch):
    import httpx
    from kisholens.ml import embeddings

    mock_resp = MagicMock()
    mock_resp.status_code = 500

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp

    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "hf_test_mock_token")
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: mock_client)

    # When HF fails with 500, it should seamlessly fall back to local SentenceTransformer
    vec = embeddings.embed_single_text("Testing fallback to local model")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert pytest.approx(np.linalg.norm(vec), abs=1e-3) == 1.0

