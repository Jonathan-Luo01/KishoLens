"""
test_similarity.py — Unit tests for nearest neighbors vector search logic.
"""

from kisholens.ml.similarity import find_top_matches, _init_cache_from_disk, _novel_vector_cache


def test_cache_hydration_from_disk():
    _init_cache_from_disk()
    assert len(_novel_vector_cache) >= 10000
    assert 235 in _novel_vector_cache
    assert _novel_vector_cache[235]["title"] == "The Adventures of Sherlock Holmes"
    assert _novel_vector_cache[235]["vector"].shape == (8,)
    assert _novel_vector_cache[235]["primary_genre"] == "Mystery"



def test_find_top_matches_basic():
    dummy_features = {
        "en_theme_explication_ratio": 0.05,
        "en_linearity_subversion_score": 0.15,
        "en_sensory_body_density": 0.40,
        "en_outside_world_engagement": 1.20,
        "en_narrative_feature_diversity": 0.60,
        "en_dialogue_ratio": 0.35,
        "en_ttr": 0.50,
        "en_temporal_shift_score": 0.15,
    }

    matches = find_top_matches(dummy_features, query_text="Reincarnation into a fantasy game world with magic status screens.", top_k=5)
    
    assert isinstance(matches, list)
    assert len(matches) <= 5
    for match in matches:
        assert "id" in match
        assert "title" in match
        assert "author" in match
        assert "genre" in match
        assert "similarity_score" in match
        assert 0.0 <= match["similarity_score"] <= 1.0
        # Verify breakdown is present
        assert "breakdown" in match
        breakdown = match["breakdown"]
        assert "style" in breakdown
        assert "semantic" in breakdown
        assert "genre" in breakdown
        assert "tags" in breakdown
        assert "territory" in breakdown
        for factor_name, factor_val in breakdown.items():
            assert 0.0 <= factor_val <= 1.0, f"{factor_name} out of range: {factor_val}"


def test_find_top_matches_exclusion():
    dummy_features = {
        "en_dialogue_ratio": 0.40,
        "en_ttr": 0.50,
    }

    matches = find_top_matches(dummy_features, exclude_novel_id=1, top_k=5)
    
    assert isinstance(matches, list)
    for match in matches:
        assert match["id"] != 1
        assert "breakdown" in match
