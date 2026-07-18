import numpy as np
import os
import json
import tempfile
import pytest

from kisholens.ml.build_centroids import (
    consolidate_genre,
    embed_texts,
    compute_centroid,
    save_centroids,
    load_centroids_from_disk,
    GENRE_TAG_MAP,
    GENRE_TERRITORIES,
)


def test_consolidate_genre_known_litrpg():
    assert consolidate_genre(["litrpg", "adventure"]) == "LitRPG"


def test_consolidate_genre_known_system():
    assert consolidate_genre(["system", "action"]) == "LitRPG"


def test_consolidate_genre_known_isekai():
    assert consolidate_genre(["isekai", "fantasy"]) == "Isekai"


def test_consolidate_genre_known_cultivation():
    assert consolidate_genre(["cultivation", "action"]) == "Xianxia / Wuxia"


def test_consolidate_genre_no_match():
    assert consolidate_genre(["unknown-tag-xyz"]) is None


def test_consolidate_genre_known_mystery():
    assert consolidate_genre(["mystery"]) == "Mystery"


def test_consolidate_genre_empty():
    assert consolidate_genre([]) is None


def test_consolidate_genre_case_insensitive():
    assert consolidate_genre(["Isekai"]) == "Isekai"
    assert consolidate_genre(["LITRPG"]) == "LitRPG"


def test_genre_tag_map_has_all_genres():
    expected = {
        "LitRPG", "Isekai", "Xianxia / Wuxia", "Urban Romance",
        "Cozy Fantasy", "Slice of Life / Contemporary", "Villainess / Otome Game",
        "Kingdom Building / Strategy", "Monster Protagonist / Evolution",
        "Dungeon Core / Dungeon MC", "Urban Fantasy / Dungeons",
        "Harem", "Girls Love / Boys Love",
        "High Fantasy", "Hard Sci-Fi", "Modern Thriller",
        "Victorian Novel", "Philosophical Fiction",
        "Mystery", "Horror", "Romance", "Fantasy",
        "Sci-Fi", "Action / Adventure", "Comedy"
    }
    assert expected == set(GENRE_TAG_MAP.keys())


def test_genre_territories_covers_all_genres():
    for genre in GENRE_TAG_MAP:
        assert genre in GENRE_TERRITORIES, f"Missing territory for {genre}"


def test_embed_texts_shape():
    texts = ["The hero raised his sword.", "She cultivated qi in silence.", "A mystery unfolded."]
    embeddings = embed_texts(texts)
    assert embeddings.shape == (3, 384)
    assert embeddings.dtype == np.float32


def test_embed_texts_single():
    texts = ["Just one sentence."]
    embeddings = embed_texts(texts)
    assert embeddings.shape == (1, 384)


def test_embed_texts_truncation():
    # 2000-word text should not raise
    long_text = "word " * 2000
    embeddings = embed_texts([long_text])
    assert embeddings.shape == (1, 384)


def test_compute_centroid_of_identical_vectors():
    v = np.array([1.0, 0.0, 0.0] * 128, dtype=np.float32).reshape(1, 384)
    repeated = np.tile(v, (5, 1))
    centroid = compute_centroid(repeated)
    np.testing.assert_allclose(centroid, v[0], atol=1e-6)


def test_compute_centroid_shape():
    embeddings = np.random.rand(10, 384).astype(np.float32)
    centroid = compute_centroid(embeddings)
    assert centroid.shape == (384,)


def test_save_load_roundtrip():
    centroids = np.random.rand(3, 384).astype(np.float32)
    meta = {
        "genres": ["LitRPG", "Isekai", "High Fantasy"],
        "territories": ["Web Novel Territory", "Web Novel Territory", "Traditional Fiction Territory"],
        "samples_used": {"LitRPG": 10, "Isekai": 10, "High Fantasy": 10}
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        save_centroids(centroids, meta, filename_prefix="genre", data_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "genre_centroids.npy"))
        assert os.path.exists(os.path.join(tmpdir, "genre_centroids_meta.json"))
        loaded_centroids, loaded_meta = load_centroids_from_disk(filename_prefix="genre", data_dir=tmpdir)
        np.testing.assert_array_equal(centroids, loaded_centroids)
        assert loaded_meta["genres"] == meta["genres"]


def test_load_centroids_missing_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_centroids_from_disk(filename_prefix="genre", data_dir=tmpdir)
        assert result == (None, None)
