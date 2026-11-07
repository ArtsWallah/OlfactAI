"""Integration tests for chemistry API endpoints and error handling."""

import pytest
from fastapi.testclient import TestClient


class TestChemistryAPI:
    """Integration test suite for chemistry endpoints."""

    def test_analyze_valid_smiles(self, client: TestClient, sample_molecules):
        """Verify successful molecular analysis for aspirin."""
        aspirin = sample_molecules["aspirin"]
        payload = {
            "smiles": aspirin["smiles"],
            "include_fingerprint": False,
        }

        response = client.post("/api/v1/chemistry/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["is_valid"] is True
        assert data["canonical_smiles"] == aspirin["canonical"]
        assert "descriptors" in data
        assert "fingerprint" in data

        # Verify descriptors
        desc = data["descriptors"]
        assert desc["molecular_formula"] == "C9H8O4"
        assert abs(desc["molecular_weight"] - 180.16) < 0.1
        assert desc["ring_count"] == 1
        assert desc["h_bond_donors"] == 1
        assert desc["h_bond_acceptors"] == 3
        assert desc["rotatable_bonds"] == 2

        # Verify fingerprint metadata (dense vector omitted by default)
        fp = data["fingerprint"]
        assert fp["radius"] == 2
        assert fp["nbits"] == 2048
        assert fp["on_bits_count"] > 0
        assert fp["bit_vector"] is None

    def test_analyze_with_dense_fingerprint(self, client: TestClient):
        """Verify request with include_fingerprint=True returns full binary array."""
        payload = {
            "smiles": "CCO",
            "include_fingerprint": True,
            "morgan_radius": 2,
            "morgan_nbits": 2048,
        }

        response = client.post("/api/v1/chemistry/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()

        fp = data["fingerprint"]
        assert fp["bit_vector"] is not None
        assert len(fp["bit_vector"]) == 2048
        assert all(b in (0, 1) for b in fp["bit_vector"])

    def test_analyze_invalid_smiles_returns_structured_error(self, client: TestClient):
        """Verify invalid SMILES returns 422 with standardized error response."""
        payload = {
            "smiles": "Invalid_Chemical_SMILES_123",
            "include_fingerprint": False,
        }

        response = client.post("/api/v1/chemistry/analyze", json=payload)
        assert response.status_code == 422
        data = response.json()

        assert data["success"] is False
        assert "error" in data
        assert data["error"]["code"] == "INVALID_SMILES"
        assert "Invalid" in data["error"]["message"] or "rejected" in data["error"]["message"]

    def test_analyze_empty_smiles_validation_error(self, client: TestClient):
        """Verify empty SMILES triggers Pydantic validation error."""
        payload = {"smiles": ""}
        response = client.post("/api/v1/chemistry/analyze", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_validate_endpoint_valid_smiles(self, client: TestClient):
        """Verify POST /api/v1/chemistry/validate with valid structure."""
        payload = {"smiles": "c1ccccc1"}
        response = client.post("/api/v1/chemistry/validate", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["is_valid"] is True
        assert data["canonical_smiles"] == "c1ccccc1"
        assert data["error_details"] is None

    def test_validate_endpoint_invalid_smiles(self, client: TestClient):
        """Verify POST /api/v1/chemistry/validate with unparseable structure."""
        payload = {"smiles": "C1CC"}  # Unclosed ring
        response = client.post("/api/v1/chemistry/validate", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["is_valid"] is False
        assert data["canonical_smiles"] is None
        assert data["error_details"] is not None
