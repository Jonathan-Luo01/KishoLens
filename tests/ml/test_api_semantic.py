from fastapi.testclient import TestClient
import tempfile
import os
import json
import numpy as np

from kisholens.api.main import app
from kisholens.ml.semantic_match import DEFAULT_DATA_DIR


def test_api_analyze_without_centroids():
    # Force a temporary directory where centroids do not exist
    with tempfile.TemporaryDirectory() as tmpdir:
        # We override DEFAULT_DATA_DIR / mock loading by patch or just using the api client
        # In api/main.py, match_semantic is called with default arguments (which points to DEFAULT_DATA_DIR).
        # We can temporarily rename the real data folder or mock the function.
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
