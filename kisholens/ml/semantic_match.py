"""
semantic_match.py — Live inference for semantic genre/territory matching.

Loads pre-built genre centroids from disk and matches a user's text
to the closest genre using cosine similarity over sentence embeddings.

Gracefully returns None if centroids have not been built yet.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from kisholens.ml.build_centroids import (
    embed_texts,
    load_centroids_from_disk,
)

# Module-level centroid cache: {data_dir: {"genre": (centroids, meta), "territory": (centroids, meta)}}
_centroid_cache: dict[str, dict[str, tuple[np.ndarray, dict]]] = {}

# Default centroid location (relative to project root)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)


def clear_centroid_cache() -> None:
    """Clear in-memory centroid cache to force reload from disk."""
    _centroid_cache.clear()


def _load_with_cache(data_dir: str) -> tuple[Optional[np.ndarray], Optional[dict], Optional[np.ndarray], Optional[dict]]:
    """
    Load centroids from disk, caching per data_dir so repeated calls within
    the same process do not re-read files.
    """
    if data_dir not in _centroid_cache:
        g_centroids, g_meta = load_centroids_from_disk("genre", data_dir)
        t_centroids, t_meta = load_centroids_from_disk("territory", data_dir)
        if g_centroids is not None and t_centroids is not None:
            _centroid_cache[data_dir] = {
                "genre": (g_centroids, g_meta),
                "territory": (t_centroids, t_meta)
            }
        else:
            return None, None, None, None
    entry = _centroid_cache[data_dir]
    return entry["genre"][0], entry["genre"][1], entry["territory"][0], entry["territory"][1]


import re

GENRE_TAXONOMY: dict[str, dict] = {
    "Fantasy": {
        "anchor_terms_en": ["archmage", "wizard", "sorcerer", "elf", "elven", "dragon", "magic", "spell", "spire", "goblin", "orc", "kingdom", "paladin", "high fantasy", "wyrm", "mana", "realm", "arcane", "fairy tale", "enchanted"],
        "keywords_en": ["mythology", "folklore", "legends", "epic fantasy", "sword and sorcery", "kingdom building", "academy", "cozy fantasy", "dungeon core"],
        "keywords_zh": ["奇幻", "玄幻", "魔法", "神话", "剑与魔法", "异界"],
        "keywords_ja": ["ファンタジー", "魔法", "魔王", "勇者", "ハイファンタジー", "ドラゴン", "エルフ", "異世界ファンタジー"]
    },
    "Isekai": {
        "anchor_terms_en": ["reincarnat", "transmigrat", "reborn", "truck-kun", "another world", "summoned", "otome", "villainess", "otherworld", "regressor", "second chance", "portal fantasy"],
        "keywords_en": ["summoned hero", "different world", "banished from the party", "returnee"],
        "keywords_zh": ["穿越", "重生", "转生", "穿书", "快穿", "异界", "异世", "魂穿", "恶役千金", "召回"],
        "keywords_ja": ["異世界", "転生", "転移", "悪役令嬢", "追放", "チート", "異世界召喚", "現世", "復讐", "再生", "やり直し"]
    },
    "Cultivation": {
        "anchor_terms_en": ["dantian", "qi", "tribulation", "immortal", "dao", "xianxia", "wuxia", "breakthrough", "cultivat", "pill refining", "spirit herb", "meridian", "golden core", "nascent soul"],
        "keywords_en": ["cultivator", "spiritual root", "alchemy furnace", "elixir refining", "dan refining", "foundation building", "jianghu", "courting death", "acupoint", "spirit stone", "jade slip"],
        "keywords_zh": ["修仙", "仙侠", "修真", "玄幻", "武侠", "功法", "宗门", "丹药", "金丹", "元婴", "道士", "灵石", "神通", "渡劫", "飞升", "筑基", "炼气", "江湖"],
        "keywords_ja": ["仙侠", "武侠", "仙人", "修行", "気", "丹田", "道"]
    },
    "Progression Fantasy": {
        "anchor_terms_en": ["status window", "stat point", "level up", "dungeon", "system notification", "litrpg", "stat screen", "leveling", "experience points", "exp", "class rank", "weak to strong", "level system", "accelerated growth", "vrmmo", "vrmmorpg", "divine stats"],
        "keywords_en": ["skill point", "quest log", "inventory slot", "system prompt", "dungeon crawler", "raid boss"],
        "keywords_zh": ["属性面板", "任务日志", "升级", "系统", "游戏异界", "玩家", "副本", "领地"],
        "keywords_ja": ["ステータス", "レベル", "ダンジョン", "スキル", "システム", "VRMMO", "ゲーマー", "ステータスウィンドウ", "レベルアップ"]
    },
    "Horror": {
        "anchor_terms_en": ["eldritch", "possession", "macabre", "nightmare", "coffin", "crypt", "ghoul", "zombie"],
        "fear_atmosphere_en": ["dread", "spine-chilling", "grotesque", "horrifying", "unspeakable horror", "terror", "uncanny", "creeping fear", "mutilated", "gruesome", "visceral horror", "suffocating panic"],
        "keywords_zh": ["恐怖", "惊悚", "灵异", "鬼怪", "凶宅", "阴尸", "诡异", "噩梦", "绝望", "悚然"],
        "keywords_ja": ["ホラー", "怪談", "呪い", "心霊", "恐怖", "畏怖", "怪异", "絶望", "狂気"],
        "neutral_entities": ["vampire", "werewolf", "skeleton", "ghost", "specter", "undead", "reanimated"]
    },
    "Romance": {
        "anchor_terms_en": ["romance", "otome", "harem", "love story", "contract marriage", "relationship", "rom-com", "romantic comedy", "reverse harem", "concubine", "fiancee", "fiancée", "bride", "wife", "lover", "heroine", "love", "romantic"],
        "keywords_zh": ["恋爱", "甜文", "虐恋", "总裁", "豪门", "婚约", "契约婚姻", "纯爱", "言情", "傲娇"],
        "keywords_ja": ["恋愛", "溺愛", "婚約", "幼馴染", "ツンデレ", "乙女ゲーム", "ラブコメ", "伯爵夫人", "契約婚"]
    },
    "Sci-Fi": {
        "anchor_terms_en": ["spaceship", "starship", "cybernetic", "android", "warp drive", "alien", "galaxy", "quantum", "cyberpunk", "teleport", "spacecraft", "interstellar", "sci-fi", "science fiction", "terraforming", "mecha"],
        "keywords_en": ["galactic empire", "dystopia", "nanite", "space opera", "future technology", "robot", "cyborg", "artificial intelligence"],
        "keywords_zh": ["科幻", "星际", "末世", "机甲", "赛博朋克", "太空", "战舰", "人工智能", "银河"],
        "keywords_ja": ["SF", "サイバーパンク", "メカ", "宇宙", "宇宙船", "ロボット", "ポストアポカリプス", "人工知能", "銀河"]
    },
    "Historical": {
        "anchor_terms_en": ["historical fiction", "victorian era", "regency era", "qing dynasty", "ming dynasty", "tang dynasty", "han dynasty", "song dynasty", "edo period", "sengoku", "french revolution", "american civil war", "ancient history", "medieval europe"],
        "keywords_zh": ["历史", "宫斗", "朝堂", "穿越历史", "架空历史", "种田历史", "权谋"],
        "keywords_ja": ["歴史", "時代劇", "江戸", "戦国", "大正", "昭和", "架空歴史", "宫廷"]
    },
    "Mystery": {
        "anchor_terms_en": ["inspector", "detective", "cyanide", "poison", "whodunit", "murder victim", "suspect", "forensic", "crime scene", "homicide", "clue", "case", "interrogat"],
        "keywords_en": ["police procedural", "criminology", "investigation", "thriller", "suspense", "noir", "sleuth", "locked-room mystery"],
        "keywords_zh": ["悬疑", "推理", "探案", "刑侦", "破案", "法医", "盗墓", "密室"],
        "keywords_ja": ["推理", "ミステリー", "探偵", "密室", "刑事", "サスペンス", "犯人"]
    },
    "Action / Adventure": {
        "anchor_terms_en": ["swordfight", "battlefield", "ambush", "expedition", "quest", "combat", "blade", "warrior", "erupted", "unleashed", "roar", "mercenary", "survival", "death game"],
        "keywords_en": ["martial arts", "hunting", "voyages", "travels", "pirates", "shipwreck"],
        "keywords_zh": ["动作", "冒险", "兵王", "特种兵", "战神", "雇佣兵", "爽文"],
        "keywords_ja": ["アクション", "バトル", "冒険", "海賊", "生き残り", "爽快"]
    },
    "Drama": {
        "anchor_terms_en": ["tragedy", "betrayal", "conflict", "family feud", "tearful", "sorrow", "heartbreak", "emotional", "social hierarchy"],
        "keywords_en": ["domestic drama", "social drama", "interpersonal relations", "psychological", "political intrigue"],
        "keywords_zh": ["戏剧", "家庭", "伦理", "情感", "权谋", "狗血", "背叛"],
        "keywords_ja": ["ドラマ", "内部抗争", "裏切り", "泥沼", "愛憎", "心理戦"]
    },
    "Comedy": {
        "anchor_terms_en": ["hilarious", "laugh", "absurd", "prank", "chuckle", "parody", "sarcastic", "slapstick", "funny protagonist", "misunderstandings"],
        "keywords_en": ["satire", "gag", "situational comedy"],
        "keywords_zh": ["搞笑", "沙雕", "爆笑", "幽默", "喜剧", "吐槽", "勘误"],
        "keywords_ja": ["コメディ", "ギャグ", "勘違い", "爆笑", "喜劇", "ギャグアニメ"]
    },
    "Slice of Life": {
        "anchor_terms_en": ["cozy", "cafe", "tea shop", "everyday", "peaceful", "neighborhood", "school life", "slow life", "farming", "pet raising", "cooking"],
        "keywords_en": ["workplace life", "daily life", "relaxing", "healing", "slow pacing"],
        "keywords_zh": ["日常", "温馨", "慢生活", "种田", "美食", "治愈", "轻松"],
        "keywords_ja": ["日常", "ほのぼの", "スローライフ", "追放されスローライフ", "料理", "育成", "学園", "治癒"]
    },
    "Supernatural": {
        "anchor_terms_en": ["paranormal", "spirit", "phantom", "curse", "exorcist", "occult", "urban fantasy", "modern magic", "exorcism", "werewolf", "vampire"],
        "keywords_en": ["spiritualism", "apparitions", "demonology", "hunters", "necromancy"],
        "keywords_zh": ["超自然", "通灵", "都市异能", "阴阳师", "驱魔", "捉鬼"],
        "keywords_ja": ["伝奇", "あやかし", "妖", "陰陽師", "祓い屋", "現代ファンタジー", "異能"]
    },
    "Poetry": {
        "anchor_terms_en": ["epic poem", "verse", "sonnet", "stanza", "poetic", "ballad", "canto", "lyric", "heav'n", "thou", "thy", "thee", "dost", "hath", "sing", "heavenly muse", "high song"],
        "keywords_zh": ["诗歌", "诗集", "词集", "赋", "韵文"],
        "keywords_ja": ["詩", "歌集", "俳句", "短歌", "叙情詩"]
    },
    "Philosophy": {
        "anchor_terms_en": ["philosophical", "philosophy", "theological", "ethics", "existence", "metaphysics", "moral philosophy", "virtue", "providence", "divine", "heavenly", "soul", "wisdom"],
        "keywords_zh": ["哲学", "伦理", "存在主义", "形而上学", "神学", "思想"],
        "keywords_ja": ["哲学", "倫理", "思想", "存在论", "形而上学"]
    },
    "Tragedy": {
        "anchor_terms_en": ["fatalism", "moral downfall", "melancholy", "suffering", "sad ending", "terminal illness", "angst", "grief", "mourning", "doom", "sacrifice"],
        "keywords_zh": ["悲剧", "虐文", "意难平", "绝症", "致命伤"],
        "keywords_ja": ["悲劇", "メリーバッドエンド", "ウツ人生", "鬱展開", "死別"]
    }
}

# Legacy ANCHOR_TERMS mapping for backwards compatibility
ANCHOR_TERMS: dict[str, list[str]] = {
    k: data.get("anchor_terms_en", []) for k, data in GENRE_TAXONOMY.items()
}


def detect_text_language(text: str) -> str:
    """Detect whether input text is Japanese (ja), Chinese (zh), or English (en)."""
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
        return "ja"
    elif re.search(r'[\u4e00-\u9fff]', text):
        return "zh"
    return "en"


def scan_anchor_boosts(text: str, lang: Optional[str] = None) -> dict[str, float]:
    """
    Scan text dynamically based on detected text language for macro-genre anchor terms.
    """
    if not lang:
        lang = detect_text_language(text)

    low_text = text.lower()
    boosts: dict[str, float] = {}

    # 1. Dynamic multi-lingual macro term scanning
    for genre_key, data in GENRE_TAXONOMY.items():
        patterns = list(data.get("anchor_terms_en", []))
        if lang == "zh":
            patterns.extend(data.get("keywords_zh", []))
        elif lang == "ja":
            patterns.extend(data.get("keywords_ja", []))
        else:
            patterns.extend(data.get("keywords_en", []))

        matches = set()
        for pat in patterns:
            if re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', pat):
                if pat.lower() in low_text:
                    matches.add(pat)
            else:
                clean_pat = pat.rstrip("*")
                if clean_pat.startswith(r"\b"):
                    regex = clean_pat
                else:
                    regex = r"\b" + re.escape(clean_pat) + r"\b"
                if re.search(regex, low_text):
                    matches.add(pat)
        if matches:
            boosts[genre_key] = min(0.35, 0.12 * len(matches))

    # 2. Horror Neutral Entity vs Atmospheric Fear Isolation
    horror_data = GENRE_TAXONOMY["Horror"]
    fear_atmosphere = horror_data.get("fear_atmosphere_en", []) + horror_data.get("keywords_zh", []) + horror_data.get("keywords_ja", [])
    neutral_entities = horror_data.get("neutral_entities", [])

    fear_matches = sum(1 for m in fear_atmosphere if (m in low_text if re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', m) else re.search(r'\b' + re.escape(m) + r'\b', low_text)))
    entity_matches = sum(1 for e in neutral_entities if re.search(r'\b' + re.escape(e) + r'\b', low_text))

    if fear_matches == 0:
        # Penalize Horror if zero atmospheric fear markers are found
        boosts["Horror"] = max(-0.45, boosts.get("Horror", 0.0) - 0.45)
        # If neutral entities exist without fear atmosphere, boost Supernatural instead of Horror
        if entity_matches >= 1:
            boosts["Supernatural"] = boosts.get("Supernatural", 0.0) + 0.25

    # 3. Fantasy / Isekai European nobility protection
    fantasy_isekai_markers = sum(1 for m in ["archmage", "wizard", "sorcerer", "dragon", "magic", "spell", "reincarnat", "transmigrat", "truck-kun", "another world", "system notification", "status window", "level up", "litrpg", "dungeon", "demon lord", "grand duke", "marquess", "king", "kingdom", "monarch"] if re.search(r'\b' + re.escape(m) + r'\b', low_text))
    if fantasy_isekai_markers >= 2:
        boosts["Historical"] = max(-0.35, boosts.get("Historical", 0.0) - 0.35)
        if any(w in low_text for w in ["reincarnat", "transmigrat", "truck-kun", "another world", "summoned", "otome"]):
            boosts["Isekai"] = boosts.get("Isekai", 0.0) + 0.30

    # 5. Archaic Verse Poetry Boost
    poetic_archaic = sum(1 for p in ["thou", "thy", "thee", "dost", "hath", "heav'n", "canto", "stanza", "verse", "poetic", "muses"] if re.search(r'\b' + re.escape(p) + r'\b', low_text))
    if poetic_archaic >= 3:
        boosts["Poetry"] = boosts.get("Poetry", 0.0) + 0.40
        boosts["Philosophy"] = boosts.get("Philosophy", 0.0) + 0.25

    return boosts


from kisholens.ml.analyzer import analyze_prose


def match_semantic(
    text: str,
    title: Optional[str] = None,
    synopsis: Optional[str] = None,
    model_name: str = "all-MiniLM-L6-v2",
    data_dir: str = DEFAULT_DATA_DIR,
) -> Optional[dict]:
    paragraphs = text.split("\n\n")
    ch1 = paragraphs[0] if paragraphs else text
    ch10 = paragraphs[len(paragraphs) // 2] if len(paragraphs) > 2 else None
    ch20 = paragraphs[-1] if len(paragraphs) > 2 else None

    taxonomy = analyze_prose(synopsis, ch1, ch10, ch20, title=title, data_dir=data_dir)
    if not taxonomy:
        return None

    world_primary = taxonomy["world_setting"]["primary"]
    world_score = taxonomy["world_setting"]["score"]

    return {
        "genre": world_primary,
        "genre_confidence": world_score,
        "territory": "Web Novel Territory",
        "territory_confidence": 0.95,
        "genre_scores": [{"genre": world_primary, "score": world_score, "raw_score": world_score}],
        "territory_scores": [{"territory": "Web Novel Territory", "score": 0.95, "raw_score": 0.95}],
        "taxonomy": taxonomy,
    }

