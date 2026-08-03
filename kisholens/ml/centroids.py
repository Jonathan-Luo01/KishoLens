"""
centroids.py — Pure Concept Vectors and Book Centroid manager.
"""

from __future__ import annotations
from typing import Dict
import numpy as np
from kisholens.ml.embeddings import embed_single_text

INCITING_CONCEPTS: Dict[str, str] = {
    "Isekai & Regression": (
        "The protagonist dies and is reincarnated, opens their eyes and finds themselves in a fantasy/game/other world, "
        "transmigrated into a novel or game as a villainess or mob character, summoned to another world as a hero, "
        "or regresses back in time to their past life for a second chance at changing their fate."
    ),
    "System Initialization": (
        "A mysterious system interface suddenly appears before the protagonist's eyes, granting them a status window, "
        "levels, skills, and quests. The world undergoes an apocalyptic evolution or shifts into a game-like reality "
        "with dungeons and monsters."
    ),
    "Cultivation Awakening": (
        "The protagonist discovers a heaven-defying cheat artifact, awakens a supreme spiritual root, "
        "or repairs their crippled meridians to begin their journey on the path of cultivation, martial arts, and immortality."
    ),
}

_concept_vector_cache: Dict[str, np.ndarray] = {}


def get_concept_vector(concept_name: str) -> np.ndarray:
    if concept_name not in _concept_vector_cache:
        text = INCITING_CONCEPTS.get(concept_name, "")
        _concept_vector_cache[concept_name] = embed_single_text(text)
    return _concept_vector_cache[concept_name]


def get_inciting_concept_vectors() -> Dict[str, np.ndarray]:
    for name in INCITING_CONCEPTS:
        get_concept_vector(name)
    return _concept_vector_cache
