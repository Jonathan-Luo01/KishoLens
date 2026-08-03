import numpy as np
import pytest
from kisholens.ml.centroids import (
    INCITING_CONCEPTS,
    get_concept_vector,
    get_inciting_concept_vectors,
)


def test_inciting_concepts_dict():
    assert "Isekai & Regression" in INCITING_CONCEPTS
    assert "System Initialization" in INCITING_CONCEPTS
    assert "Cultivation Awakening" in INCITING_CONCEPTS
    assert len(INCITING_CONCEPTS) == 3


def test_get_concept_vector():
    vec = get_concept_vector("Isekai & Regression")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert pytest.approx(float(np.linalg.norm(vec)), abs=1e-3) == 1.0


def test_get_inciting_concept_vectors():
    vecs = get_inciting_concept_vectors()
    assert len(vecs) == 3
    assert "Isekai & Regression" in vecs
    assert vecs["Isekai & Regression"].shape == (384,)
    assert "System Initialization" in vecs
    assert "Cultivation Awakening" in vecs


def test_concept_vector_caching():
    vec1 = get_concept_vector("System Initialization")
    vec2 = get_concept_vector("System Initialization")
    assert vec1 is vec2
