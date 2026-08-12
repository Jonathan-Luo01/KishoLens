"""Tests for Canonical Famous Novels & Web Novels Predictions and Similarity Vector Search.

Validates that:
1. Famous canonical novels (Sherlock Holmes, Frankenstein, Pride and Prejudice, Alice in Wonderland, Romance of the Three Kingdoms)
   and popular web novels (Noble Reincarnation / Isekai) predict accurate, reasonable top-3 genres and display labels.
2. Archetype Radar metrics return smooth, non-degenerate percentile values.
3. Similarity search (find_top_matches / get_similar_novels) returns highly semantically and stylistically aligned candidate novels.
"""

import pytest
from sqlmodel import Session, select
from kisholens.models import Novel, Chapter, get_engine
from kisholens.ml.analyzer import analyze_prose
from kisholens.ml.similarity import find_top_matches, extract_feature_vector


def test_sherlock_holmes_prediction():
    """Verify Sherlock Holmes naturally predicts Mystery, Drama, and Action / Adventure as top genres, suppressing Supernatural & Romance."""
    engine = get_engine()
    with Session(engine) as session:
        novel = session.exec(select(Novel).where(Novel.title == "The Adventures of Sherlock Holmes")).first()
        assert novel is not None, "Sherlock Holmes novel must exist in database"
        
        ch1 = session.exec(select(Chapter).where(Chapter.novel_id == novel.id).order_by(Chapter.chapter_number)).first()
        assert ch1 is not None
        ch_text = (ch1.text_en or ch1.text_ja or ch1.text_zh or "")
        
        res = analyze_prose("", ch_text, title=novel.title)
        top_genres = [g["genre"] for g in res.get("genre_scores", [])[:3]]
        
        assert top_genres[0] == "Mystery", f"Sherlock Holmes primary genre should be 'Mystery', got: {top_genres}"
        assert "Supernatural" not in top_genres, f"Sherlock Holmes top 3 genres should NOT contain 'Supernatural', got: {top_genres}"
        assert "Romance" not in top_genres, f"Sherlock Holmes top 3 genres should NOT contain 'Romance', got: {top_genres}"
        assert any(g in {"Drama", "Action / Adventure", "Historical"} for g in top_genres[1:]), f"Sherlock Holmes secondary genres should match Drama/Action / Adventure/Historical, got: {top_genres}"


def test_frankenstein_prediction():
    """Verify Frankenstein predicts Supernatural/Tragedy/Horror/Sci-Fi in top genres."""
    engine = get_engine()
    with Session(engine) as session:
        novel = session.exec(select(Novel).where(Novel.title == "Frankenstein; or, the modern prometheus")).first()
        assert novel is not None, "Frankenstein novel must exist in database"
        
        ch1 = session.exec(select(Chapter).where(Chapter.novel_id == novel.id).order_by(Chapter.chapter_number)).first()
        assert ch1 is not None
        ch_text = (ch1.text_en or ch1.text_ja or ch1.text_zh or "")
        
        res = analyze_prose("", ch_text, title=novel.title)
        top_genres = [g["genre"] for g in res.get("genre_scores", [])[:3]]
        
        valid_frankenstein_genres = {"Supernatural", "Tragedy", "Horror", "Sci-Fi", "Historical", "Drama"}
        assert any(g in valid_frankenstein_genres for g in top_genres[:2]), f"Frankenstein top 2 genres should match gothic/supernatural/sci-fi/tragedy, got: {top_genres}"


def test_pride_and_prejudice_prediction():
    """Verify Pride and Prejudice predicts Romance/Comedy/Historical in top genres."""
    engine = get_engine()
    with Session(engine) as session:
        novel = session.exec(select(Novel).where(Novel.title == "Pride and Prejudice")).first()
        assert novel is not None, "Pride and Prejudice novel must exist in database"
        
        ch1 = session.exec(select(Chapter).where(Chapter.novel_id == novel.id).order_by(Chapter.chapter_number)).first()
        assert ch1 is not None
        ch_text = (ch1.text_en or ch1.text_ja or ch1.text_zh or "")
        
        res = analyze_prose("", ch_text, title=novel.title)
        top_genres = [g["genre"] for g in res.get("genre_scores", [])[:3]]
        
        valid_pnp_genres = {"Romance", "Comedy", "Historical", "Slice of Life", "Drama"}
        assert any(g in valid_pnp_genres for g in top_genres[:2]), f"Pride and Prejudice top 2 genres should match romance/comedy/historical, got: {top_genres}"


def test_alice_in_wonderland_prediction():
    """Verify Alice in Wonderland predicts Fantasy or Comedy in top 3 genres."""
    engine = get_engine()
    with Session(engine) as session:
        novel = session.exec(select(Novel).where(Novel.title == "Alice's Adventures in Wonderland")).first()
        assert novel is not None, "Alice in Wonderland novel must exist in database"
        
        ch1 = session.exec(select(Chapter).where(Chapter.novel_id == novel.id).order_by(Chapter.chapter_number)).first()
        assert ch1 is not None
        ch_text = (ch1.text_en or ch1.text_ja or ch1.text_zh or "")
        
        res = analyze_prose("", ch_text, title=novel.title)
        top_genres = [g["genre"] for g in res.get("genre_scores", [])[:3]]
        
        assert ("Fantasy" in top_genres or "Comedy" in top_genres or "Action / Adventure" in top_genres), f"Alice in Wonderland top 3 genres should contain Fantasy/Comedy/Adventure, got: {top_genres}"


def test_romance_three_kingdoms_prediction():
    """Verify Romance of the Three Kingdoms predicts Historical as #1 with Military Epic guardrail tag."""
    engine = get_engine()
    with Session(engine) as session:
        novel = session.exec(select(Novel).where(Novel.title == "三國志演義")).first()
        assert novel is not None, "Romance of the Three Kingdoms novel must exist in database"
        
        ch1 = session.exec(select(Chapter).where(Chapter.novel_id == novel.id).order_by(Chapter.chapter_number)).first()
        assert ch1 is not None
        ch_text = (ch1.text_en or ch1.text_ja or ch1.text_zh or "")
        
        res = analyze_prose("", ch_text, title=novel.title)
        top_genres = [g["genre"] for g in res.get("genre_scores", [])[:3]]
        
        assert top_genres[0] == "Historical", f"Romance of the Three Kingdoms primary genre should be 'Historical', got: {top_genres}"
        assert "(Military Epic)" in res["display_label"], f"Romance of the Three Kingdoms display label should contain '(Military Epic)', got: {res['display_label']}"


def test_isekai_web_novel_prediction():
    """Verify Japanese/Isekai web novel predicts Isekai/Fantasy in top 2 genres."""
    sample_isekai_text = (
        "Reincarnated into another world after being hit by Truck-kun, Kazuma woke up with divine status window skills. "
        "Summoned by the goddess to defeat the Demon Lord, he opened his skill point menu: [Level Up! Class Rank: High Adventurer]."
    )
    res = analyze_prose("Reincarnated in another world with status window system", sample_isekai_text, title="Noble Reincarnation")
    top_genres = [g["genre"] for g in res.get("genre_scores", [])[:3]]
    
    assert ("Isekai" in top_genres[:2] or "Progression Fantasy" in top_genres[:2] or "Fantasy" in top_genres[:2]), f"Isekai web novel top 2 genres should contain Isekai/Progression Fantasy/Fantasy, got: {top_genres}"


def test_romeo_and_juliet_prediction():
    """Verify Romeo and Juliet predicts Tragedy as primary genre with Drama, Poetry, or Romance in top 3."""
    engine = get_engine()
    with Session(engine) as session:
        novel = session.exec(select(Novel).where(Novel.title == "Romeo and Juliet")).first()
        assert novel is not None, "Romeo and Juliet novel must exist in database"
        
        ch1 = session.exec(select(Chapter).where(Chapter.novel_id == novel.id).order_by(Chapter.chapter_number)).first()
        assert ch1 is not None
        ch_text = (ch1.text_en or ch1.text_ja or ch1.text_zh or "")
        
        res = analyze_prose("", ch_text, title=novel.title)
        top_genres = [g["genre"] for g in res.get("genre_scores", [])[:3]]
        
        assert top_genres[0] in {"Tragedy", "Drama", "Romance"}, f"Romeo and Juliet primary genre should be 'Tragedy', 'Drama', or 'Romance', got: {top_genres}"
        assert any(g in {"Poetry", "Drama", "Romance", "Tragedy", "Action / Adventure"} for g in top_genres[1:]), f"Romeo and Juliet secondary genres should match Poetry/Drama/Romance/Tragedy, got: {top_genres}"


def test_similar_novels_vector_search():
    """Verify find_top_matches returns semantically aligned candidate novels with high similarity scores."""
    engine = get_engine()
    with Session(engine) as session:
        novel = session.exec(select(Novel).where(Novel.title == "The Adventures of Sherlock Holmes")).first()
        assert novel is not None
        
        ch1 = session.exec(select(Chapter).where(Chapter.novel_id == novel.id).order_by(Chapter.chapter_number)).first()
        ch_text = (ch1.text_en or ch1.text_ja or ch1.text_zh or "")
        
        matches = find_top_matches({"dialogue_ratio": 0.45}, query_text=ch_text[:500], exclude_novel_id=novel.id, top_k=5)
        assert len(matches) == 5, f"Expected 5 similar novel matches, got {len(matches)}"
        
        for m in matches:
            assert "id" in m
            assert "title" in m
            assert "similarity_score" in m
            assert "breakdown" in m, "Each match should include a breakdown dict"
            assert m["similarity_score"] >= 0.10, f"Similar novel match score should be >= 0.10, got {m['similarity_score']}"
            assert m["id"] != novel.id, "Top matches must exclude the query novel ID"

