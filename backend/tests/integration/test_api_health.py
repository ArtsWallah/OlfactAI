"""Integration tests for health endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestHealthAPI:
    """Integration test suite for application health checks."""

    def test_root_health_endpoint(self, client: TestClient):
        """Verify GET /health returns 200 OK and valid health payload."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["app_name"] == "OlfactAI"
        assert data["cpu_only"] is True
        assert "rdkit" in data["components"]
        assert "torch" in data["components"]

    def test_v1_health_endpoint(self, client: TestClient):
        """Verify GET /api/v1/health returns matching payload."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["cpu_only"] is True

