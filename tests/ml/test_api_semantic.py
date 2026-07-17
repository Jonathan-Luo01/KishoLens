from fastapi.testclient import TestClient
from kisholens.api.main import app


def test_api_analyze_without_centroids():
    # Let's mock match_semantic to return None to simulate missing centroids.
    import kisholens.api.main as api_main
    original_match_semantic = api_main.match_semantic
    api_main.match_semantic = lambda text: None

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
        "territory": "Web Novel Territory",
        "confidence": 0.95,
        "scores": [
            {"genre": "LitRPG", "territory": "Web Novel Territory", "score": 0.95},
            {"genre": "Isekai", "territory": "Web Novel Territory", "score": 0.80}
        ]
    }
    api_main.match_semantic = lambda text: dummy_response

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
        assert data["semantic"]["confidence"] == 0.95
        assert len(data["semantic"]["scores"]) == 2
    finally:
        api_main.match_semantic = original_match_semantic
