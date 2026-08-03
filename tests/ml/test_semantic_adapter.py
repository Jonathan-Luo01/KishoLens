import pytest
from kisholens.ml.semantic_match import match_semantic

def test_match_semantic_adapter():
    text = "Reincarnated into another world as the 13th Imperial Prince with divine stats."
    res = match_semantic(text, title="Noble Reincarnation")
    assert res is not None
    assert "genre" in res
    assert "genre_scores" in res
    assert "taxonomy" in res
    assert res["taxonomy"]["world_setting"]["primary"] is not None
