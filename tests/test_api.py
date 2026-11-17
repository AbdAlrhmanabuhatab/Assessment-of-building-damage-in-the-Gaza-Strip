import os
import pytest
from fastapi.testclient import TestClient
from src.api.server import create_app

client = TestClient(create_app())

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "model" in data

def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 3
    assert any(s["category"] == "Repair" for s in samples)

def test_predict_endpoint_valid():
    sample_path = "assets/sample_repair.jpg"
    if not os.path.exists(sample_path):
        sample_path = "static/samples/repair_bank.jpg"

    with open(sample_path, "rb") as f:
        response = client.post(
            "/api/predict",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "gradcam_overlay_base64" in data
    assert data["prediction"] in ("Safe", "Repair", "Demolish")

def test_predict_endpoint_invalid_file():
    response = client.post(
        "/api/predict",
        files={"file": ("note.txt", b"plain text", "text/plain")}
    )
    assert response.status_code == 400

def test_predict_endpoint_empty():
    response = client.post(
        "/api/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
