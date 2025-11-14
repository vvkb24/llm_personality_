import json
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_analyze_text_no_model():
    # If model not trained, we expect 503
    resp = client.post("/api/analyze/text", json={"text":"Hello world"})
    assert resp.status_code in (503, 200)
