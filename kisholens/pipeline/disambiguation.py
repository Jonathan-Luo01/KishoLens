import re
from typing import List, Dict, Any, Optional

# --- HORROR ---
MONSTER_ENTITY_MARKERS = [
    "skeleton", "demon lord", "demon king", "vampire", "ghost", "lich",
    "undead", "necromancer", "zombie", "monster", "creature", "fiend", "spectre"
]
FEAR_ATMOSPHERE_MARKERS = [
    "dread", "spine-chilling", "grotesque", "horrifying", "unspeakable horror",
    "terror", "chilling", "uncanny", "creeping fear", "mutilated", "macabre",
    "gruesome", "eldritch", "terrifying", "horror", "fear", "gothic"
]

# --- SCI-FI ---
PURE_SCIFI_MARKERS = [
    "spaceship", "cyberpunk", "interstellar", "alien", "galactic empire",
    "dystopia", "starship", "quantum", "nanite", "android", "warp drive"
]
SYSTEM_INTERFACE_MARKERS = [
    "status window", "level up", "stat point", "vrmmo", "system notification",
    "skill point", "dungeon", "quest log", "inventory slot"
]

# --- CULTIVATION ---
CULTIVATION_CORE_MARKERS = [
    "qi", "dantian", "cultivation", "cultivator", "cultivate", "dao", "sect",
    "immortal ascension", "immortal cultivation", "immortal emperor", "immortal monarch",
    "immortal cave", "tribulation", "meridian", "spiritual root", "alchemy furnace",
    "pill refining", "elixir refining", "dan refining", "sutra",
    "golden core", "foundation building", "jianghu", "courting death",
    "young master of the clan", "young master of the sect", "yin", "yang",
    "escort agency", "internal energy", "acupoint", "bottleneck", "impurities",
    "nascent soul", "ginseng", "spirit stone", "jade slip", "heavenly tribulation",
    "xianxia", "wuxia", "xuanhuan", "eastern fantasy"
]

# --- PROGRESSION & ISEKAI ---
PROGRESSION_ISEKAI_MARKERS = [
    "reincarnation", "reincarnated", "reincarnate", "transmigration",
    "transmigrated", "transmigrate", "another world", "other world",
    "different world", "summoned hero", "summoned", "truck-kun", "reborn",
    "isekai", "tensei"
]


def count_markers(markers: List[str], text: str) -> int:
    """Count exact word/phrase boundary occurrences of markers in text, supporting CJK characters."""
    if not text:
        return 0
    text_lower = text.lower()
    total = 0
    for marker in markers:
        if re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', marker):
            matches = len(re.findall(re.escape(marker), text_lower))
        else:
            matches = len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
        total += matches
    return total


def disambiguate_and_rank_genres(
    tags_str: str = "",
    text_sample: str = "",
    source: str = "",
    territory: str = "",
    initial_genre: str = ""
) -> Dict[str, Any]:
    """
    Disambiguation and penalty/boost engine for cross-genre trope overlaps.
    """
    is_classic = (
        source == "gutenberg" or
        "classic literature" in territory.lower() or
        "classic literature" in initial_genre.lower()
    )

    corpus = f"{tags_str} {text_sample}".strip()

    monster_count = count_markers(MONSTER_ENTITY_MARKERS, corpus)
    fear_count = count_markers(FEAR_ATMOSPHERE_MARKERS, corpus)
    pure_scifi_count = count_markers(PURE_SCIFI_MARKERS, corpus)
    system_count = count_markers(SYSTEM_INTERFACE_MARKERS, corpus)
    cultivation_count = count_markers(CULTIVATION_CORE_MARKERS, corpus)
    isekai_count = count_markers(PROGRESSION_ISEKAI_MARKERS, corpus)
    sys_isekai_count = system_count + isekai_count

    penalties = []
    boosts = []

    # Initial base scores derived from tags and keyword matching
    from kisholens.ml.build_centroids import COMMON_GENRES
    scores: Dict[str, float] = {g: 0.0 for g in COMMON_GENRES.keys()}
    tags_lower = [t.strip().lower() for t in (tags_str or "").split(",")]

    for g_name, kw_list in COMMON_GENRES.items():
        matched_tags = 0
        exact_canonical_match = False
        cjk_match_score = 0.0
        
        for kw in kw_list:
            is_cjk = bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', kw))
            if is_cjk:
                if kw in (tags_str or ""):
                    exact_canonical_match = True
                    cjk_match_score += 2.0
                elif kw in corpus:
                    cjk_match_score += 1.0
            else:
                for t in tags_lower:
                    if t == g_name.lower() or t == kw.lower():
                        exact_canonical_match = True
                    if re.search(r'\b' + re.escape(kw) + r'\b', t):
                        matched_tags += 1

        if exact_canonical_match:
            scores[g_name] += 1.5 + cjk_match_score
        elif matched_tags >= 3 or cjk_match_score > 0:
            scores[g_name] += float(matched_tags) + cjk_match_score

    # Preserve explicit initial genre if present (only if not forcing initial_genre parameter empty)
    if initial_genre:
        for g_name in COMMON_GENRES.keys():
            if g_name.lower() in initial_genre.lower():
                scores[g_name] += 1.5

    # Text-based Horror Boost (for classic/gothic horror works like Dracula and Frankenstein)
    if fear_count >= 2 or (monster_count >= 1 and fear_count >= 1):
        scores["Horror"] += 1.5
        boosts.append(f"Horror Text Boost (+1.50): {fear_count} fear markers & {monster_count} monster markers found in text.")

    # Skip trope penalties for classic literature
    if not is_classic:
        # --- RULE A: HORROR DISAMBIGUATION ---
        if monster_count > 0 and sys_isekai_count > 0:
            if fear_count <= sys_isekai_count:
                scores["Horror"] = max(0.0, scores["Horror"] - 0.30)
                penalties.append("Horror Penalty (-0.30): Monster entities found alongside system/isekai tropes without dominant fear atmosphere.")
                if scores["Fantasy"] == 0.0:
                    scores["Fantasy"] += 0.5
                if isekai_count > 0:
                    scores["Isekai"] += 0.5
                else:
                    scores["Progression Fantasy"] += 0.5

        # --- RULE B: SCI-FI DISAMBIGUATION ---
        if scores.get("Sci-Fi", 0.0) > 0.0 or "sci-fi" in corpus.lower():
            if system_count > 0 and pure_scifi_count < system_count:
                scores["Sci-Fi"] = max(0.0, scores["Sci-Fi"] - 0.35)
                penalties.append("Sci-Fi Penalty (-0.35): System interface / LitRPG markers outnumber pure Sci-Fi markers.")
                if "vrmmo" in corpus.lower() or "vrmmorpg" in corpus.lower():
                    scores["Progression Fantasy"] += 0.5
                else:
                    scores["Fantasy"] += 0.5

        # --- RULE C: CULTIVATION PRIORITIZATION & NON-CHINESE PENALTY ---
        is_chinese_novel = (
            source.lower() in ["cnnovel", "qidian", "faloo", "ciweimao"] or
            bool(re.search(r'[\u4e00-\u9fff]', corpus))
        )

        has_explicit_cultivation_tag = any(
            t in (tags_str or "").lower() for t in ["cultivation", "xianxia", "wuxia", "xuanhuan"]
        )

        if cultivation_count >= 3:
            scores["Cultivation"] += 1.50
            boosts.append(f"Cultivation Boost (+1.50): {cultivation_count} core cultivation markers found.")
            # Promote Cultivation to Primary Genre above generic Action / Fantasy
            scores["Cultivation"] = max(scores["Cultivation"], scores.get("Fantasy", 0.0) + 0.50, scores.get("Action / Adventure", 0.0) + 0.50)
            if isekai_count >= 3 or "isekai" in (tags_str or "").lower():
                # Hierarchy Rule: Cultivation as Parent Genre, Isekai as Secondary Subgenre
                scores["Isekai"] = max(scores.get("Isekai", 0.0), scores.get("Fantasy", 0.0) + 0.25)
        elif has_explicit_cultivation_tag:
            scores["Cultivation"] += 1.0
        else:
            # Zero out Cultivation for isolated occurrences without explicit tags or >= 3 markers
            scores["Cultivation"] = 0.0

        if not is_chinese_novel and scores.get("Cultivation", 0.0) > 0.0:
            scores["Cultivation"] = max(0.0, scores["Cultivation"] - 0.01)
            penalties.append("Cultivation Non-Chinese Penalty (-0.01): Small penalty applied for non-Chinese web novel source.")

        # --- ISEKAI PRIORITIZATION ---
        has_explicit_isekai_tag = any(
            t in (tags_str or "").lower() for t in ["isekai", "reincarnation", "transmigration", "tensei"]
        )
        if (isekai_count >= 3 or has_explicit_isekai_tag) and cultivation_count < 3:
            scores["Isekai"] += 1.0
            # Ensure Isekai is in top parent genres over generic Action / Adventure
            scores["Isekai"] = max(scores["Isekai"], scores.get("Action / Adventure", 0.0) + 0.25)
            boosts.append(f"Isekai Boost (+1.00): {isekai_count} isekai markers found.")

        # --- SLICE OF LIFE PRIORITIZATION ---
        has_explicit_sol_tag = any(
            t in (tags_str or "").lower() for t in [
                "slice of life", "slice-of-life", "slow life", "slow-life", "daily life",
                "everyday life", "cozy", "farming", "cooking", "school life"
            ]
        )
        if has_explicit_sol_tag:
            scores["Slice of Life"] += 1.0
            boosts.append("Slice of Life Boost (+1.00): Explicit Slice of Life / Slow Life tag found.")

    # Sort genres by score descending
    sorted_genres = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    active_genres = [g for g, s in sorted_genres if s > 0.0]

    if not active_genres:
        active_genres = ["Action / Adventure", "Fantasy"]

    # Hierarchy Rule for Cultivation & Isekai
    if not is_classic and cultivation_count >= 3:
        if "Cultivation" in active_genres:
            active_genres.remove("Cultivation")
            if "Isekai" in active_genres:
                active_genres.remove("Isekai")
                active_genres = ["Cultivation", "Isekai"] + active_genres
            else:
                active_genres = ["Cultivation"] + active_genres
    elif not is_classic and (isekai_count >= 3 or has_explicit_isekai_tag):
        if "Isekai" in active_genres:
            active_genres.remove("Isekai")
            if active_genres and active_genres[0] in ["Romance", "Comedy", "Progression Fantasy", "Slice of Life"]:
                active_genres = [active_genres[0], "Isekai"] + active_genres[1:]
            else:
                active_genres = ["Isekai"] + active_genres

    final_genres_str = ", ".join(active_genres[:3])

    return {
        "primary_genre": active_genres[0],
        "secondary_genres": active_genres[1:3],
        "parent_genre_str": final_genres_str,
        "scores": scores,
        "penalties_applied": penalties,
        "boosts_applied": boosts,
        "is_classic": is_classic
    }
