import numpy as np
import json
import os
import tempfile
import pytest

from kisholens.ml.semantic_match import match_semantic


def _write_dummy_centroids(tmpdir: str):
    """Write minimal genre and territory centroids for testing."""
    from kisholens.ml.build_centroids import GENRE_TAG_MAP

    genres = list(GENRE_TAG_MAP.keys())
    territories = ["Web Novel Territory", "Traditional Fiction Territory", "Classic Literature Territory"]
    
    rng = np.random.default_rng(42)
    
    # Genre centroids
    g_centroids = rng.random((len(genres), 384)).astype(np.float32)
    g_norms = np.linalg.norm(g_centroids, axis=1, keepdims=True)
    g_centroids = g_centroids / g_norms
    g_meta = {
        "genres": genres,
        "samples_used": {g: 10 for g in genres},
    }
    np.save(os.path.join(tmpdir, "genre_centroids.npy"), g_centroids)
    with open(os.path.join(tmpdir, "genre_centroids_meta.json"), "w") as f:
        json.dump(g_meta, f)
        
    # Territory centroids
    t_centroids = rng.random((len(territories), 384)).astype(np.float32)
    t_norms = np.linalg.norm(t_centroids, axis=1, keepdims=True)
    t_centroids = t_centroids / t_norms
    t_meta = {
        "territories": territories,
        "samples_used": {t: 10 for t in territories},
    }
    np.save(os.path.join(tmpdir, "territory_centroids.npy"), t_centroids)
    with open(os.path.join(tmpdir, "territory_centroids_meta.json"), "w") as f:
        json.dump(t_meta, f)
        
    return genres, territories


def test_match_semantic_returns_none_without_centroids():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = match_semantic("Some text here.", data_dir=tmpdir)
    assert result is None


def test_match_semantic_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        genres, territories = _write_dummy_centroids(tmpdir)
        result = match_semantic(
            "The young hero reincarnated into another world and gained a system.",
            data_dir=tmpdir,
        )
    assert result is not None
    assert "genre" in result
    assert "genre_confidence" in result
    assert "genre_scores" in result
    assert "territory" in result
    assert "territory_confidence" in result
    assert "territory_scores" in result
    assert isinstance(result["genre_scores"], list)
    assert len(result["genre_scores"]) == len(genres)
    assert isinstance(result["territory_scores"], list)
    assert len(result["territory_scores"]) == len(territories)


def test_match_semantic_scores_sorted_descending():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("Magic cultivation and martial arts in ancient China.", data_dir=tmpdir)
    
    g_scores = [s["score"] for s in result["genre_scores"]]
    assert g_scores == sorted(g_scores, reverse=True)
    
    t_scores = [s["score"] for s in result["territory_scores"]]
    assert t_scores == sorted(t_scores, reverse=True)


def test_match_semantic_scores_in_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("A detective investigates a murder in London.", data_dir=tmpdir)
    
    for s in result["genre_scores"]:
        assert -1.0 <= s["score"] <= 1.0
    for s in result["territory_scores"]:
        assert -1.0 <= s["score"] <= 1.0


def test_match_semantic_top_genre_matches_confidence():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("Leveling up with game stats and a system prompt.", data_dir=tmpdir)
    
    assert result["genre"] == result["genre_scores"][0]["genre"]
    assert abs(result["genre_confidence"] - result["genre_scores"][0]["score"]) < 1e-6
    
    assert result["territory"] == result["territory_scores"][0]["territory"]
    assert abs(result["territory_confidence"] - result["territory_scores"][0]["score"]) < 1e-6


def test_match_semantic_scores_have_all_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_dummy_centroids(tmpdir)
        result = match_semantic("Philosophy and existence.", data_dir=tmpdir)
    
    for s in result["genre_scores"]:
        assert "genre" in s
        assert "score" in s
    for s in result["territory_scores"]:
        assert "territory" in s
        assert "score" in s
