import sys
from fastapi.testclient import TestClient

sys.path.append("/Users/jonathan/Documents/KishoLens")
from kisholens.api.main import app

client = TestClient(app)

def test_analyze_baselines_returned():
    response = client.post("/api/analyze", json={
        "text": "This is a simple sentence to test whether the baselines are returned by the API.",
        "lang": "en",
        "title": "Baseline Test"
    })
    assert response.status_code == 200
    data = response.json()
    assert "baselines" in data
    baselines = data["baselines"]
    assert "gutenberg" in baselines
    assert "webnovel" in baselines
    
    # Assert specific metrics are present
    for source in ("gutenberg", "webnovel"):
        metrics = baselines[source]
        assert "ttr" in metrics
        assert "dialogue_ratio" in metrics
        assert "avg_sentence_len" in metrics
        assert isinstance(metrics["ttr"], float)
        assert isinstance(metrics["dialogue_ratio"], float)
        assert isinstance(metrics["avg_sentence_len"], float)

if __name__ == "__main__":
    try:
        test_analyze_baselines_returned()
        print("ALL BASELINE API TESTS PASSED!")
    except AssertionError as e:
        import traceback
        traceback.print_exc()
        print("Baseline test verification failed:", e)
        sys.exit(1)
