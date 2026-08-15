"""
test_similarity.py — Unit tests for nearest neighbors vector search logic.
"""

from kisholens.ml.features import extract_english_features
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


def test_sherlock_holmes_mystery_matching():
    text = """The rain beat against the fog-stained windowpanes of 221B Baker Street as Inspector Lestrade threw open the heavy oak door. His coat was drenched, and his eyes burned with anxiety. "Holmes, you must come at once," he gasped, resting his hands upon the polished mahogany table. "Lord Harrington lies motionless in his study, the doors locked from within and a shattered crystal decanter resting beside his chair."

Sherlock Holmes did not rise immediately. He slowly lowered his pipe, allowing a dense ring of blue smoke to curl toward the ceiling before adjusting his magnifying lens. "A locked room, Lestrade? How delightfully elementary. And tell me, did you observe the faint scent of bitter almonds clinging to the victim's lips?" Lestrade blinked in astonishment. "Why, yes, Holmes—how could you possibly know?" Holmes turned to me with a faint smile. "A classic case of cyanide poisoning, Watson. Pack your bag; the hunt is afoot."""

    feat = extract_english_features(text)
    matches = find_top_matches(feat, query_text=text, top_k=5)

    assert len(matches) == 5
    matched_titles = [m["title"] for m in matches]
    has_sherlock = any("Sherlock" in t for t in matched_titles)
    assert has_sherlock, f"Expected Sherlock Holmes in top matches, got: {matched_titles}"

    top_match = matches[0]
    assert top_match["similarity_score"] >= 0.70
    assert top_match["breakdown"]["genre"] >= 0.80


def test_isekai_fantasy_matching():
    text = """I woke up in an unfamiliar stone chamber with a glowing blue interface hovering before my eyes. 
[System Initialized. Welcome, User. Status: Level 1 Reincarnated Adventurer.]
I grabbed my iron dagger and stepped out into the monster-infested dungeon."""
    feat = extract_english_features(text)
    matches = find_top_matches(feat, query_text=text, top_k=5)
    assert len(matches) == 5
    # Primary match should be Isekai / Fantasy / Web Novel
    top_match = matches[0]
    assert top_match["similarity_score"] >= 0.50


def test_romance_matching():
    text = """Elizabeth Bennett smiled gently across the drawing room, feeling her heart flutter as Mr. Darcy approached with quiet hesitation. His gaze was full of tender affection and unuttered promises of eternal love."""
    feat = extract_english_features(text)
    matches = find_top_matches(feat, query_text=text, top_k=5)
    assert len(matches) == 5
    top_match = matches[0]
    assert top_match["similarity_score"] >= 0.50


def test_match_reasons_generation():
    text = """The rain beat against the fog-stained windowpanes of 221B Baker Street as Inspector Lestrade threw open the heavy oak door. "Holmes, you must come at once," he gasped."""
    feat = extract_english_features(text)
    matches = find_top_matches(feat, query_text=text, top_k=5)

    assert len(matches) > 0
    for m in matches:
        assert "reasons" in m
        assert isinstance(m["reasons"], list)
        assert len(m["reasons"]) >= 1
        for reason in m["reasons"]:
            assert isinstance(reason, str)
            # Confirm no emojis or icon characters
            assert all(ord(c) < 128 or '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in reason), f"Found emoji or non-text char in reason: {reason}"


def test_granular_match_badges_and_comparisons():
    from kisholens.ml.similarity import find_top_matches
    q_feats = {
        "en_dialogue_ratio": 0.65,
        "en_avg_sentence_length": 11.2,
        "en_ttr": 0.46,
        "en_sensory_body_density": 0.70,
        "en_theme_explication_ratio": 2.8,
        "genre": "Fantasy, Isekai",
        "territory": "Web Novel"
    }
    matches = find_top_matches(q_feats, query_text="Hero summoned to another world", top_k=3)
    assert len(matches) > 0
    top = matches[0]
    assert "match_badges" in top
    assert isinstance(top["match_badges"], list)
    assert len(top["match_badges"]) >= 1
    assert "type" in top["match_badges"][0]
    assert "label" in top["match_badges"][0]
    assert "detail" in top["match_badges"][0]
    assert "tier" in top["match_badges"][0]
    assert "metric_comparisons" in top
    assert isinstance(top["metric_comparisons"], list)
    assert len(top["metric_comparisons"]) >= 1
    assert "metric" in top["metric_comparisons"][0]
    assert "query" in top["metric_comparisons"][0]
    assert "candidate" in top["metric_comparisons"][0]
    assert "match" in top["metric_comparisons"][0]




