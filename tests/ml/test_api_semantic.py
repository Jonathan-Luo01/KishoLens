from fastapi.testclient import TestClient
from kisholens.api.main import app


def test_api_analyze_without_centroids():
    # Let's mock match_semantic to return None to simulate missing centroids.
    import kisholens.api.main as api_main
    original_match_semantic = api_main.match_semantic
    api_main.match_semantic = lambda text, *args, **kwargs: None

    try:
        client = TestClient(app)
        response = client.post(
            "/api/analyze",
            json={"text": "The hero raised his sword against the demon king.", "lang": "en"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "archetype" in data
        assert "semantic" not in data  # Graceful degradation
    finally:
        api_main.match_semantic = original_match_semantic


def test_api_analyze_with_centroids():
    # Mock match_semantic to return a valid dummy response
    import kisholens.api.main as api_main
    original_match_semantic = api_main.match_semantic

    dummy_response = {
        "genre": "LitRPG",
        "genre_confidence": 0.95,
        "genre_scores": [
            {"genre": "LitRPG", "score": 0.95},
            {"genre": "Isekai", "score": 0.80}
        ],
        "territory": "Web Novel Territory",
        "territory_confidence": 0.90,
        "territory_scores": [
            {"territory": "Web Novel Territory", "score": 0.90}
        ]
    }
    api_main.match_semantic = lambda text, *args, **kwargs: dummy_response

    try:
        client = TestClient(app)
        response = client.post(
            "/api/analyze",
            json={"text": "Reincarnated into a game with stats.", "lang": "en"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "archetype" in data
        assert "semantic" in data
        assert data["semantic"]["genre"] == "LitRPG"
        assert data["semantic"]["genre_confidence"] == 0.95
        assert len(data["semantic"]["genre_scores"]) == 2
        assert data["semantic"]["territory"] == "Web Novel Territory"
        assert data["semantic"]["territory_confidence"] == 0.90
    finally:
        api_main.match_semantic = original_match_semantic
