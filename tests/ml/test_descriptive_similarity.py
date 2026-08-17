# tests/ml/test_descriptive_similarity.py
import pytest
from kisholens.ml.similarity import find_top_matches


def test_descriptive_similarity_for_database_novel():
    # Test matching for Noble Reincarnation (ID 1)
    matches = find_top_matches(target_novel_id=1, limit=3)
    assert len(matches) > 0
    top = matches[0]
    
    # 1. Verify narrative reasoning wrapper
    assert "narrative_reasoning" in top
    reasoning = top["narrative_reasoning"]
    
    # 2. Verify dynamic narrative synthesis paragraph
    assert "narrative_synthesis" in reasoning
    synthesis = reasoning["narrative_synthesis"]
    assert isinstance(synthesis, str)
    assert len(synthesis) > 30
    assert not synthesis.startswith("Unknown")
    
    # 3. Verify 4-Pillar Narrative Alignment Matrix schema
    assert "pillars" in reasoning
    pillars = reasoning["pillars"]
    
    required_pillars = ["catalyst", "setting", "conflict", "style_cadence"]
    for p_key in required_pillars:
        assert p_key in pillars, f"Missing pillar: {p_key}"
        pillar = pillars[p_key]
        assert "name" in pillar and isinstance(pillar["name"], str) and len(pillar["name"]) > 0
        assert "score" in pillar and isinstance(pillar["score"], (float, int))
        assert 0.0 <= pillar["score"] <= 1.0
        assert "query_val" in pillar and isinstance(pillar["query_val"], str) and len(pillar["query_val"]) > 0
        assert "cand_val" in pillar and isinstance(pillar["cand_val"], str) and len(pillar["cand_val"]) > 0
        assert "explanation" in pillar and isinstance(pillar["explanation"], str) and len(pillar["explanation"]) > 10

    # 4. Verify shared tropes
    assert "shared_tropes" in reasoning
    assert isinstance(reasoning["shared_tropes"], list)


def test_descriptive_similarity_for_raw_user_text():
    # Test dynamic inference for raw user input prose without metadata
    raw_text = """
    In a flash of blinding azure light, I opened my eyes in an ornate palace chamber.
    The grand duke stared down with cold calculation. "You have awakened, my son," he murmured.
    My previous life as an ordinary salaryman was gone; I was now the third prince in an empire teetering on civil war.
    """
    matches = find_top_matches(query_text=raw_text, limit=3)
    assert len(matches) > 0
    top = matches[0]
    
    assert "narrative_reasoning" in top
    reasoning = top["narrative_reasoning"]
    
    # Verify narrative synthesis addresses user input
    synthesis = reasoning["narrative_synthesis"]
    assert "Your" in synthesis or "narrative" in synthesis or "prose" in synthesis
    
    # Verify dynamic NLP inference extracted the reincarnation & palace keywords
    pillars = reasoning["pillars"]
    assert "Reincarnation" in pillars["catalyst"]["query_val"]
    assert "Court" in pillars["setting"]["query_val"] or "Imperial" in pillars["setting"]["query_val"] or "Aristocracy" in pillars["setting"]["query_val"]
    assert "War" in pillars["conflict"]["query_val"] or "Intrigue" in pillars["conflict"]["query_val"] or "Succession" in pillars["conflict"]["query_val"]
    
    # Check that style metrics are populated
    assert "w/s" in pillars["style_cadence"]["query_val"]
    assert "w/s" in pillars["style_cadence"]["cand_val"]
    
    # Check shared tropes extracted
    tropes = reasoning["shared_tropes"]
    assert isinstance(tropes, list)
