import pytest
from kisholens.ml.analyzer import analyze_prose


def test_analyze_prose_isekai_novel():
    synopsis = "Reincarnated into another world as the 13th Imperial Prince with divine stats."
    ch1 = "I opened my eyes to a grand canopy bed inside the Thirteenth Prince Palace."
    ch10 = "I held the demon sword and relaxed in the imperial garden."
    ch20 = "The demon sword unleashed its power."

    res = analyze_prose(synopsis, ch1, ch10, ch20, title="Noble Reincarnation")
    assert "inciting_event" in res
    assert "world_setting" in res
    assert "narrative_plot" in res
    assert "display_label" in res

    inciting = res["inciting_event"]
    assert inciting is not None
    assert inciting["primary"] == "Isekai & Regression"
    assert inciting["score"] >= 0.70
    assert "Isekai & Regression" in res["display_label"]


def test_analyze_prose_fallback_threshold():
    ch1 = "The tea was served cold in the parlor as Mr. Bennett discussed the evening news."
    ch10 = "Lady Catherine walked through the garden complaining about the weather."
    ch20 = "They sat quietly by the fireplace in the drawing room."

    res = analyze_prose(None, ch1, ch10, ch20, title="Quiet Tea Room")
    assert res["inciting_event"] is None
    assert "world_setting" in res
    assert "narrative_plot" in res
    assert "display_label" in res
    # Should not start with inciting event if inciting_event is None
    assert not res["display_label"].startswith("Isekai")
    assert not res["display_label"].startswith("System")
    assert not res["display_label"].startswith("Cultivation")


def test_analyze_prose_no_centroids():
    res = analyze_prose("Synopsis", "Chapter 1", data_dir="/invalid/directory/path")
    assert res == {}


def test_analyze_prose_scenario_a_pure_classical_epic():
    title = "The Great Dynasty War"
    synopsis = "The emperor led his army and troops into battle against the rebellion."
    ch1 = "General Zhao commanded the cavalry and ordered a siege on the rebel stronghold."
    ch10 = "The imperial court debated military tactics while troops marched towards the border."
    ch20 = "Warlord Li declared war under the mandate of heaven."

    res = analyze_prose(synopsis, ch1, ch10, ch20, title=title)
    assert "(Military Epic)" in res["display_label"]


def test_analyze_prose_scenario_b_hybrid_cultivation():
    title = "The Cultivator General"
    synopsis = "An emperor and his army fought the rebellion using ancient martial arts."
    ch1 = "General Zhao gathered troops while channeling energy through his dantian and meridian."
    ch10 = "The general achieved a breakthrough during the siege."
    ch20 = "Spirit stone resources supplied the imperial army."

    res = analyze_prose(synopsis, ch1, ch10, ch20, title=title)
    assert "(Kingdom Building / Military)" in res["display_label"]
    assert res["narrative_plot"]["primary"] == "Historical / Military"

