"""
taxonomy.py — Centralized taxonomy definitions, marker lexicons, and guardrail evaluation.
"""

import re
from typing import Dict, Any, List

HARD_CULTIVATION_MARKERS: List[str] = [
    "dantian", "meridian", "qi gathering", "foundation establishment", "nascent soul", 
    "pill refining", "spirit stone", "bottleneck", "breakthrough",
    "丹田", "经脉", "煉氣", "炼气", "築基", "筑基", "元嬰", "元婴", "靈石", "灵石", "突破", "瓶頸", "瓶颈"
]

MILITARY_EPIC_MARKERS: List[str] = [
    "army", "general", "emperor", "rebellion", "troops", "strategy", 
    "dynasty", "warlord", "mandate of heaven", "cavalry", "imperial court", 
    "marched", "camp", "siege",
    "将军", "將軍", "军队", "軍隊", "皇帝", "朝廷", "叛乱", "叛亂", "天下", "诸侯", "諸侯", 
    "兵马", "兵馬", "城池", "谋士", "謀士", "官军", "官軍", "大将", "大將"
]


def scan_marker_counts(text: str) -> Dict[str, int]:
    """
    Counts distinct matches for HARD_CULTIVATION_MARKERS and MILITARY_EPIC_MARKERS in text.
    Uses regex word-boundaries for English terms and substring containment for CJK characters.
    """
    low_text = text.lower()
    
    def _count_matches(markers: List[str]) -> int:
        matched = set()
        for pat in markers:
            if re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', pat):
                if pat in text:
                    matched.add(pat)
            else:
                regex = r'\b' + re.escape(pat) + r'\b'
                if re.search(regex, low_text):
                    matched.add(pat)
        return len(matched)

    return {
        "cultivation": _count_matches(HARD_CULTIVATION_MARKERS),
        "military": _count_matches(MILITARY_EPIC_MARKERS),
    }


def evaluate_epic_cultivation_guardrail(text: str) -> Dict[str, Any]:
    """
    Evaluates whether text is Scenario A (Pure Classical Epic) or Scenario B (Hybrid Cultivation).
    """
    counts = scan_marker_counts(text)
    mil_count = counts["military"]
    cult_count = counts["cultivation"]

    if mil_count >= 3 and cult_count == 0:
        return {
            "scenario": "A",
            "cultivation_penalty": -0.40,
            "historical_boost": 0.20,
            "display_tag": "(Military Epic)",
        }
    elif mil_count >= 3 and cult_count >= 2:
        return {
            "scenario": "B",
            "cultivation_penalty": 0.0,
            "historical_boost": 0.0,
            "secondary_plot": "Historical / Military",
            "display_tag": "(Kingdom Building / Military)",
        }
    
    return {
        "scenario": "NONE",
        "cultivation_penalty": 0.0,
        "historical_boost": 0.0,
    }
