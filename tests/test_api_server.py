"""Integration tests for FastAPI backend."""

import pytest
from fastapi.testclient import TestClient

from src.api.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_status(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["problem_id"] == "26053"
    assert data["memory_optimization"]["reduction_pct"] >= 98.0


def test_api_scenarios(client):
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert "scenarios" in data
    assert len(data["scenarios"]) >= 3


def test_api_sweep(client):
    response = client.get("/api/sweep?scenario=military_recon&frame=0")
    assert response.status_code == 200
    data = response.json()
    assert "timings_ms" in data
    assert "dynamic_objects" in data
    assert "elevation_heatmap" in data
    assert "terrain_class_map" in data
    assert len(data["elevation_heatmap"]) == 100
    assert len(data["terrain_class_map"]) == 100


def test_api_process_custom_points(client):
    # Send 10 custom points
    custom_pts = [
        [0.0, 0.0, 0.0, 100.0],
        [1.0, 1.0, 0.0, 120.0],
        [5.0, 5.0, 1.2, 200.0],
        [-2.0, 3.0, 0.15, 80.0],
    ]
    response = client.post("/api/process_points", json={"points": custom_pts, "timestamp": 0.1})
    assert response.status_code == 200
    data = response.json()
    assert data["total_points"] == 4


def test_api_training_metrics(client):
    response = client.get("/api/training_metrics")
    assert response.status_code == 200
    data = response.json()
    assert "mean_val_accuracy" in data or "status" in data


def test_api_rover_control(client):
    response = client.post("/api/rover_control", json={"steering_deg": 15.0, "speed_mps": 2.5})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["rover"]["steering_deg"] == 15.0
