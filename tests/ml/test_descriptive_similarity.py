# tests/ml/test_descriptive_similarity.py
import pytest
from kisholens.ml.similarity import find_top_matches

def test_descriptive_similarity_for_database_novel():
    # Test matching for Noble Reincarnation (ID 1)
    matches = find_top_matches(target_novel_id=1, limit=3)
    assert len(matches) > 0
    top = matches[0]
    
    assert "narrative_reasoning" in top
    reasoning = top["narrative_reasoning"]
    assert "narrative_synthesis" in reasoning
    assert len(reasoning["narrative_synthesis"]) > 20
    
    assert "pillars" in reasoning
    pillars = reasoning["pillars"]
    assert "catalyst" in pillars
    assert "setting" in pillars
    assert "conflict" in pillars
    assert "style_cadence" in pillars
    
    assert "shared_tropes" in reasoning
    assert isinstance(reasoning["shared_tropes"], list)


def test_descriptive_similarity_for_raw_user_text():
    # Test matching for arbitrary user text without title or synopsis
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
    assert "narrative_synthesis" in reasoning
    assert "pillars" in reasoning
    pillars = reasoning["pillars"]
    assert pillars["catalyst"]["score"] > 0
    assert pillars["setting"]["score"] > 0
