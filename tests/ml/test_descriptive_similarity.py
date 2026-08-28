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

def test_editorial_natural_phrasing():
    from kisholens.ml.similarity import _generate_narrative_synthesis, _compute_4pillar_breakdown
    
    q_anat = {"catalyst": "Reincarnation", "setting": "Empire", "conflict": "War"}
    c_anat = {"catalyst": "Reincarnation", "setting": "Empire", "conflict": "War"}
    
    synth = _generate_narrative_synthesis(q_anat, c_anat, 0.9, 0.9, False)
    assert "thematic beats" not in synth.lower()
    assert "anchored by a" not in synth.lower()
    assert "socio-political hierarchy" not in synth.lower()
    assert "richly drawn backdrop" not in synth.lower()
    assert "factional friction and purposeful protagonist progression" not in synth.lower()
    
    q_m = {"dialogue_ratio": 0.5, "avg_sentence_len": 12.0}
    c_m = {"dialogue_ratio": 0.5, "avg_sentence_len": 12.0}
    
    pillars = _compute_4pillar_breakdown(q_anat, c_anat, q_m, c_m, 0.9, 0.9, 0.9)
    for p_key in ["catalyst", "setting", "conflict", "style_cadence"]:
        exp = pillars[p_key]["explanation"]
        assert "thematic beats" not in exp.lower()
        assert "socio-political hierarchy" not in exp.lower()
        assert "richly drawn backdrop" not in exp.lower()


def test_dynamic_story_anatomy_subgenres():
    from kisholens.ml.similarity import _infer_query_anatomy

    # 1. Cozy / Slice of Life Novel
    sol = _infer_query_anatomy("", {}, {"title": "Cozy Tavern Life", "synopsis": "After working as a chef, he opens a quiet tavern in a peaceful frontier town, cooking hearty stew and farming herbs.", "genre": "Slice of Life", "territory": "Web Novel Territory"})

    assert "Territorial Warfare" not in sol["conflict"]
    assert "Pastoral" in sol["setting"] or "Tavern" in sol["setting"] or "Frontier" in sol["setting"] or "Village" in sol["setting"]

    # 2. LitRPG / Dungeon Hunter
    hunter = _infer_query_anatomy("", {}, {"title": "Shadow Hunter Rebirth", "synopsis": "Awakening a mysterious status window in an S-Rank dungeon raid after dying to the boss monster.", "genre": "Action / Adventure", "tags": "System, Dungeon, Hunter"})
    assert "System" in hunter["catalyst"] or "Dungeon" in hunter["catalyst"] or "Status" in hunter["catalyst"]
    assert "Dungeon" in hunter["setting"] or "Urban" in hunter["setting"] or "Monster" in hunter["setting"]

    # 3. Otome Villainess
    villainess = _infer_query_anatomy("", {}, {"title": "The Villainess Desires a Quiet Life", "synopsis": "I regained memories of my past life right as the crown prince broke our engagement at the royal banquet. I must avoid execution!", "genre": "Romance", "tags": "Villainess, Otome"})
    assert "Villainess" in villainess["catalyst"] or "Otome" in villainess["catalyst"]
    assert "Ruin" in villainess["conflict"] or "Doom" in villainess["conflict"] or "Execution" in villainess["conflict"] or "Death Flag" in villainess["conflict"]

    # 4. Cultivation / Xianxia
    xianxia = _infer_query_anatomy("", {}, {"title": "Peerless Sword Dao", "synopsis": "A crippled disciple cleanses his broken dantian to ascend through ancient sect tribulations and master the flying sword.", "genre": "Fantasy", "tags": "Cultivation, Sect, Dao, Xianxia"})
    assert "Cultivation" in xianxia["catalyst"] or "Meridian" in xianxia["catalyst"] or "Martial" in xianxia["catalyst"]
    assert "Sect" in xianxia["setting"] or "Martial" in xianxia["setting"]
    assert "Sect" in xianxia["conflict"] or "Dao" in xianxia["conflict"] or "Ascension" in xianxia["conflict"]

    # 5. Locked-Room Mystery
    mystery = _infer_query_anatomy("", {}, {"title": "The Cyanide Decanter Mystery", "synopsis": "Inspector Lestrade investigates a locked-room murder where a lord was poisoned in his private library.", "genre": "Mystery", "tags": "Detective, Murder, Investigation"})
    assert "Mystery" in mystery["catalyst"] or "Investigation" in mystery["catalyst"] or "Murder" in mystery["catalyst"]
    assert "Mystery" in mystery["conflict"] or "Murder" in mystery["conflict"] or "Conspirator" in mystery["conflict"] or "Investigation" in mystery["conflict"] or "Crime" in mystery["conflict"]


def test_narrative_synthesis_and_pillar_diversity():
    from kisholens.ml.similarity import _generate_narrative_synthesis, _compute_4pillar_breakdown

    scenarios = [
        # (q_anat, c_anat, q_title, c_title)
        (
            {"catalyst": "Cozy Resettlement", "setting": "Pastoral Frontier Village & Cozy Tavern", "conflict": "Pastoral Slow-Life & Frontier Complications"},
            {"catalyst": "Cozy Resettlement", "setting": "Pastoral Frontier Village & Cozy Tavern", "conflict": "Pastoral Slow-Life & Frontier Complications"},
            "Tavern Life", "Herb Farmer"
        ),
        (
            {"catalyst": "Homicide Discovery & Forensic Investigation", "setting": "Victorian Manor & Fog-Bound Alleys", "conflict": "Deductive Investigation & Unmasking Conspirators"},
            {"catalyst": "Homicide Discovery & Forensic Investigation", "setting": "Victorian Manor & Fog-Bound Alleys", "conflict": "Deductive Investigation & Unmasking Conspirators"},
            "Sherlock Holmes", "The Poisoned Decanter"
        ),
        (
            {"catalyst": "Cultivation Initiation & Meridian Awakening", "setting": "Immortal Martial World & Wilderness Sects", "conflict": "Sect Hierarchies & Heavenly Dao Ascension"},
            {"catalyst": "Cultivation Initiation & Meridian Awakening", "setting": "Immortal Martial World & Wilderness Sects", "conflict": "Sect Hierarchies & Heavenly Dao Ascension"},
            "Sword Immortal", "Peerless Sect"
        ),
        (
            {"catalyst": "System Interface & Hunter Awakening", "setting": "Urban Fantasy & Labyrinthine Monster Gates", "conflict": "Climbing Monster Gates & High-Stakes Raids"},
            {"catalyst": "System Interface & Hunter Awakening", "setting": "Urban Fantasy & Labyrinthine Monster Gates", "conflict": "Climbing Monster Gates & High-Stakes Raids"},
            "Shadow Hunter", "Solo Gate Raid"
        ),
        (
            {"catalyst": "Villainess Fate Subversion Reincarnation", "setting": "Otome Aristocratic Empire & High Society", "conflict": "Subverting Execution & Dismantling Death Flags"},
            {"catalyst": "Villainess Fate Subversion Reincarnation", "setting": "Otome Aristocratic Empire & High Society", "conflict": "Subverting Execution & Dismantling Death Flags"},
            "Villainess Hour", "Death Flag Overturn"
        ),
        (
            {"catalyst": "Reincarnation into Imperial Nobility", "setting": "High Fantasy Imperial Court & Noble Salons", "conflict": "Imperial Succession & Concealing Overpowered Might"},
            {"catalyst": "Reincarnation into Imperial Nobility", "setting": "High Fantasy Imperial Court & Noble Salons", "conflict": "Imperial Succession & Concealing Overpowered Might"},
            "Noble Reincarnation", "Prince of the Realm"
        ),
    ]

    syntheses = set()
    for q_a, c_a, q_t, c_t in scenarios:
        synth = _generate_narrative_synthesis(q_a, c_a, 0.85, 0.85, False, q_t, c_t)
        assert len(synth) > 20
        assert synth not in syntheses, f"Duplicate synthesis generated: {synth}"
        syntheses.add(synth)

        # Check pillars
        q_m = {"dialogue_ratio": 0.6, "avg_sentence_len": 11.0}
        c_m = {"dialogue_ratio": 0.5, "avg_sentence_len": 13.0}
        pillars = _compute_4pillar_breakdown(q_a, c_a, q_m, c_m, 0.85, 0.85, 0.85, c_t)
        for p_k in ["catalyst", "setting", "conflict", "style_cadence"]:
            p = pillars[p_k]
            assert len(p["explanation"]) > 10
            assert "thematic beats" not in p["explanation"].lower()
            assert "richly drawn backdrop" not in p["explanation"].lower()


