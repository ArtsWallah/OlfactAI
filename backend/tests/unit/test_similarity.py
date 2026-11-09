"""Unit tests for molecular Tanimoto similarity."""

import pytest
from backend.app.similarity.service import MolecularSimilarityService


class TestMolecularSimilarity:
    """Test suite for Tanimoto fingerprint similarity."""

    def test_tanimoto_identity(self):
        """Verify identical molecules have Tanimoto similarity of 1.0."""
        service = MolecularSimilarityService()
        sim = service.calculate_tanimoto("CCO", "OCC")
        assert abs(sim - 1.0) < 1e-4

    def test_tanimoto_different_molecules(self):
        """Verify different molecules have similarity < 1.0."""
        service = MolecularSimilarityService()
        sim = service.calculate_tanimoto("CCO", "c1ccccc1")
        assert 0.0 <= sim < 1.0

    def test_rank_similar_molecules(self):
        """Verify candidate ranking orders highest similarity first."""
        service = MolecularSimilarityService()
        pool = ["CCO", "CCCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
        ranked = service.rank_similar_molecules("CCO", pool, top_k=3)
        assert len(ranked) == 3
        # First entry should be identical molecule
        assert ranked[0][0] == "CCO"
        assert abs(ranked[0][1] - 1.0) < 1e-4

    def test_rank_similar_molecules_invalid_k(self):
        """Verify invalid top_k raises ValueError."""
        service = MolecularSimilarityService()
        with pytest.raises(ValueError, match="top_k must be positive"):
            service.rank_similar_molecules("CCO", ["CCO", "CCCO"], top_k=0)

    def test_service_search_corpus(self):
        """Verify service delegates search_corpus to underlying engine."""
        service = MolecularSimilarityService()
        res = service.search_corpus("MOL_00001", top_k=3)
        assert len(res.matches) == 3
        assert service.engine.is_loaded


