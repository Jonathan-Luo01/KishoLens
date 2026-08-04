from kisholens.pipeline.taxonomy import (
    HARD_CULTIVATION_MARKERS,
    MILITARY_EPIC_MARKERS,
    scan_marker_counts,
    evaluate_epic_cultivation_guardrail,
)


def test_taxonomy_lexicons():
    assert len(HARD_CULTIVATION_MARKERS) > 0
    assert len(MILITARY_EPIC_MARKERS) > 0
    assert "dantian" in HARD_CULTIVATION_MARKERS
    assert "丹田" in HARD_CULTIVATION_MARKERS
    assert "army" in MILITARY_EPIC_MARKERS
    assert "将军" in MILITARY_EPIC_MARKERS


def test_scan_marker_counts():
    text_zh = "将军带领军队和诸侯，在经过关卡时遭到了叛乱。"
    counts = scan_marker_counts(text_zh)
    assert counts["military"] >= 3
    assert counts["cultivation"] == 0


def test_evaluate_epic_cultivation_guardrail_scenario_a():
    text = "The general led the emperor's troops and cavalry against the rebellion near the camp."
    res = evaluate_epic_cultivation_guardrail(text)
    assert res["scenario"] == "A"
    assert res["cultivation_penalty"] == -0.40
    assert res["historical_boost"] == 0.20
    assert res["display_tag"] == "(Military Epic)"


def test_evaluate_epic_cultivation_guardrail_scenario_b():
    text = "The emperor's general gathered qi in his dantian to achieve a breakthrough during the siege."
    res = evaluate_epic_cultivation_guardrail(text)
    assert res["scenario"] == "B"
    assert res["cultivation_penalty"] == 0.0
    assert res["historical_boost"] == 0.0
    assert res["secondary_plot"] == "Historical / Military"
    assert res["display_tag"] == "(Kingdom Building / Military)"


def test_evaluate_epic_cultivation_guardrail_scenario_none():
    text = "A quiet day in the city with no military or cultivation."
    res = evaluate_epic_cultivation_guardrail(text)
    assert res["scenario"] == "NONE"
    assert res["cultivation_penalty"] == 0.0
    assert res["historical_boost"] == 0.0
