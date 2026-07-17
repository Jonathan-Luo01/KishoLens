import numpy as np
import json
import os
import tempfile
import pytest

from kisholens.ml.semantic_match import match_semantic


def _write_dummy_centroids(tmpdir: str):
    """Write minimal genre centroids for testing (normalized random vectors)."""
    from kisholens.ml.build_centroids import GENRE_TAG_MAP, GENRE_TERRITORIES

    genres = list(GENRE_TAG_MAP.keys())
    # Use deterministic vectors so tests are reproducible
    rng = np.random.default_rng(42)
    centroids = rng.random((len(genres), 384)).astype(np.float32)
    # Normalize rows so cosine similarity is well-defined
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / norms

    meta = {
        "genres": genres,
        "territories": [GENRE_TERRITORIES[g] for g in genres],
        "samples_used": {g: 10 for g in genres},
    }
    np.save(os.path.join(tmpdir, "genre_centroids.npy"), centroids)
    with open(os.path.join(tmpdir, "genre_centroids_meta.json"), "w") as f:
        json.dump(meta, f)
    return genres, centroids, meta


def test_match_semantic_returns_none_without_centroids():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = match_semantic("Some text here.", data_dir=tmpdir)
    assert result is None


def test_match_semantic_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        genres, _, _ = _write_dummy_centroids(tmpdir)
        result = match_semantic(
            "The young hero reincarnated into another world and gained a system.",
            data_dir=tmpdir,
        )
    assert result is not None
    assert "genre" in result
    assert "territory" in result
    assert "confidence" in result
    assert "scores" in result
    assert isinstance(result["scores"], list)
    assert len(result["scores"]) == len(genres)


def test_match_semantic_scores_sorted_descending():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("Magic cultivation and martial arts in ancient China.", data_dir=tmpdir)
    scores = [s["score"] for s in result["scores"]]
    assert scores == sorted(scores, reverse=True)


def test_match_semantic_scores_in_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("A detective investigates a murder in London.", data_dir=tmpdir)
    for s in result["scores"]:
        assert -1.0 <= s["score"] <= 1.0


def test_match_semantic_top_genre_matches_confidence():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("Leveling up with game stats and a system prompt.", data_dir=tmpdir)
    assert result["genre"] == result["scores"][0]["genre"]
    assert abs(result["confidence"] - result["scores"][0]["score"]) < 1e-6


def test_match_semantic_scores_have_all_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("Philosophy and existence.", data_dir=tmpdir)
    for s in result["scores"]:
        assert "genre" in s
        assert "territory" in s
        assert "score" in s


def test_match_semantic_territory_correct():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("Victorian gothic mystery in London fog.", data_dir=tmpdir)
    # Each score entry's territory must match the known territory for that genre
    from kisholens.ml.build_centroids import GENRE_TERRITORIES
    for s in result["scores"]:
        assert s["territory"] == GENRE_TERRITORIES[s["genre"]]
