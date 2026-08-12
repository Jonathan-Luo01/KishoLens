import pytest
from kisholens.pipeline.disambiguation import disambiguate_and_rank_genres

def test_horror_disambiguation_penalty():
    # Skeleton Necromancer LitRPG without fear atmosphere
    tags = "skeleton, necromancer, level up, status window, system notification, reincarnate"
    res = disambiguate_and_rank_genres(tags_str=tags, source="syosetu")
    
    assert "Horror" not in res["parent_genre_str"] or res["primary_genre"] != "Horror"
    assert any("Horror Penalty" in p for p in res["penalties_applied"])
    assert res["primary_genre"] in ["Fantasy", "Progression Fantasy", "Isekai"]

def test_horror_classic_preservation():
    # Classic Dracula from Gutenberg
    tags = "vampire, dread, terrifying, chilling, gothic fiction"
    res = disambiguate_and_rank_genres(
        tags_str=tags,
        source="gutenberg",
        territory="Classic Literature Territory",
        initial_genre="Horror, Classic Literature"
    )
    
    assert "Horror" in res["parent_genre_str"]
    assert len(res["penalties_applied"]) == 0
    assert res["is_classic"] is True

def test_scifi_disambiguation_penalty():
    # VRMMO system interface markers without pure Sci-Fi
    tags = "vrmmo, status window, level up, skill point, quest log"
    res = disambiguate_and_rank_genres(tags_str=tags, source="syosetu")
    
    assert "Sci-Fi" not in res["parent_genre_str"] or res["primary_genre"] != "Sci-Fi"
    assert any("Sci-Fi Penalty" in p for p in res["penalties_applied"])

def test_scifi_pure_preservation():
    # Cyberpunk spaceship interstellar
    tags = "cyberpunk, spaceship, interstellar, galactic empire, android"
    res = disambiguate_and_rank_genres(tags_str=tags, source="syosetu")
    
    assert res["primary_genre"] == "Sci-Fi" or "Sci-Fi" in res["parent_genre_str"]
    assert len(res["penalties_applied"]) == 0

def test_cultivation_prioritization_and_hierarchy():
    # Reincarnated Cultivator into Xianxia world
    tags = "reincarnate, another world, qi, dantian, golden core, sect, cultivator"
    res = disambiguate_and_rank_genres(tags_str=tags, source="syosetu")
    
    assert res["primary_genre"] == "Cultivation"
    assert "Isekai" in res["secondary_genres"] or "Isekai" in res["parent_genre_str"]
    assert any("Cultivation Boost" in b for b in res["boosts_applied"])

def test_zero_cultivation_prevention():
    # Generic magic/energy without cultivation markers
    tags = "magic, energy, sword and sorcery, elf"
    res = disambiguate_and_rank_genres(tags_str=tags, source="syosetu")
    
    assert "Cultivation" not in res["parent_genre_str"]
    assert res["scores"]["Cultivation"] == 0.0

def test_non_chinese_cultivation_penalty():
    # Non-Chinese web novel source with cultivation tags
    tags = "qi, dantian, cultivation, sect"
    res_en = disambiguate_and_rank_genres(tags_str=tags, source="syosetu")
    res_cn = disambiguate_and_rank_genres(tags_str=tags, source="cnnovel")
    
    assert any("Cultivation Non-Chinese Penalty" in p for p in res_en["penalties_applied"])
    assert res_cn["scores"]["Cultivation"] > res_en["scores"]["Cultivation"]
