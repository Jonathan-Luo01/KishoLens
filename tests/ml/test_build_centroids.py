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
    get_representative_sample,
)


def test_consolidate_genre_known_litrpg():
    assert consolidate_genre(["litrpg", "adventure"]) == "Progression Fantasy"


def test_consolidate_genre_known_system():
    assert consolidate_genre(["system", "action"]) == "Progression Fantasy"


def test_consolidate_genre_known_isekai():
    assert consolidate_genre(["isekai", "fantasy"]) == "Isekai"


def test_consolidate_genre_known_cultivation():
    assert consolidate_genre(["cultivation", "action"]) == "Cultivation"


def test_consolidate_genre_no_match():
    assert consolidate_genre(["unknown-tag-xyz"]) is None


def test_consolidate_genre_known_mystery():
    assert consolidate_genre(["mystery"]) == "Mystery"


def test_consolidate_genre_empty():
    assert consolidate_genre([]) is None


def test_consolidate_genre_case_insensitive():
    assert consolidate_genre(["Isekai"]) == "Isekai"
    assert consolidate_genre(["LITRPG"]) == "Progression Fantasy"


def test_consolidate_genre_gutenberg_subjects():
    assert consolidate_genre(["Sea stories", "Pirates -- Fiction"]) == "Action / Adventure"
    assert consolidate_genre(["Humorous stories", "Satire"]) == "Comedy"
    assert consolidate_genre(["Fairy tales", "Mythology"]) == "Fantasy"
    assert consolidate_genre(["Horror tales", "Gothic fiction"]) == "Horror"
    assert consolidate_genre(["Historical fiction", "Regency Fiction"]) == "Historical"
    assert consolidate_genre(["Science fiction", "Time travel -- Fiction"]) == "Sci-Fi"
    assert consolidate_genre(["Philosophical fiction", "Ethics -- Fiction"]) == "Philosophy"
    assert consolidate_genre(["Detective and mystery stories"]) == "Mystery"
    assert consolidate_genre(["Tragedies", "Fatalism"]) == "Tragedy"
    assert consolidate_genre(["Supernatural -- Fiction", "Apparitions"]) == "Supernatural"
    assert consolidate_genre(["Poetry", "Sonnets"]) == "Poetry"
    assert consolidate_genre(["Romance fiction", "Love stories"]) == "Romance"


def test_genre_tag_map_has_all_genres():
    expected = {
        "Action / Adventure", "Comedy", "Drama", "Fantasy", "Horror",
        "Historical", "Sci-Fi", "Philosophy", "Mystery", "Tragedy",
        "Supernatural", "Poetry", "Romance", "Slice of Life",
        "Cultivation", "Isekai", "Progression Fantasy"
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
        "genres": ["Fantasy", "Isekai", "Cultivation"],
        "territories": ["Classic Literature Territory", "Web Novel Territory", "Web Novel Territory"],
        "samples_used": {"Fantasy": 10, "Isekai": 10, "Cultivation": 10}
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


def test_get_representative_sample_short_text():
    text = "word " * 500
    sample = get_representative_sample(text, target_words=10)
    assert sample == "word word word word word word word word word word"


def test_get_representative_sample_long_text_toc():
    # Long text (> 15000 chars) with table of contents/preface
    preface = "TOC line 1\nTOC line 2\nPreface line 3\n\n" * 100
    story = "This is the real story content that should be sampled cleanly.\n\n" * 300
    text = preface + story
    assert len(text) > 15000
    
    sample = get_representative_sample(text, target_words=30)
    # The first 20% is skipped, so it should start cleanly inside the story section
    assert "TOC" not in sample
    assert "Preface" not in sample
    assert "story" in sample
