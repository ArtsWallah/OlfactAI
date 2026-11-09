"""Unit tests for Phase 5 Molecular Similarity Engine."""

# pyrefly: ignore[missing-import]
import pytest
from backend.app.core.exceptions import InvalidSMILESError
from backend.app.similarity.engine import MolecularSimilarityEngine
from backend.app.similarity.models import SimilaritySearchResult


@pytest.fixture(scope="module")
def similarity_engine():
    """Load the similarity engine over the preprocessed dataset once for test efficiency."""
    return MolecularSimilarityEngine(auto_load=True)


class TestMolecularSimilarityEngine:
    """Test suite for high-performance Tanimoto search and ranking."""

    def test_engine_initialization(self, similarity_engine):
        """Verify engine successfully loads and caches 4,983 molecules."""
        assert similarity_engine.is_loaded
        assert similarity_engine.corpus_size == 4983
        assert len(similarity_engine.fingerprints) == 4983
        assert len(similarity_engine.active_labels) == 4983
        assert len(similarity_engine.descriptors_table) == 4983

    def test_fingerprint_configuration(self, similarity_engine):
        """Verify Morgan fingerprint configuration matches Phase 2 specifications."""
        assert similarity_engine.radius == 2
        assert similarity_engine.nbits == 2048
        assert similarity_engine.use_chirality is True

        # Test fingerprint on chiral molecule
        fp_r = similarity_engine.compute_fingerprint("C[C@H](O)CC")
        fp_s = similarity_engine.compute_fingerprint("C[C@@H](O)CC")
        assert fp_r.GetNumBits() == 2048
        assert fp_s.GetNumBits() == 2048
        # Because chirality is enabled, enantiomers should have non-identical fingerprints
        assert fp_r != fp_s

    def test_search_by_dataset_id(self, similarity_engine):
        """Verify search using an existing dataset molecule ID."""
        res = similarity_engine.search(query="MOL_00001", top_k=5, exclude_self=True)
        assert isinstance(res, SimilaritySearchResult)
        assert res.query_type == "dataset_id"
        assert res.query_molecule_id == "MOL_00001"
        assert len(res.matches) == 5

        # Self-match exclusion check
        match_ids = [m.molecule_id for m in res.matches]
        assert "MOL_00001" not in match_ids

        # Ranks must be 1 to 5 monotonically
        assert [m.rank for m in res.matches] == [1, 2, 3, 4, 5]

        # Top matches must have descending similarities
        sims = [m.tanimoto_similarity for m in res.matches]
        assert sims == sorted(sims, reverse=True)

    def test_search_by_external_smiles(self, similarity_engine):
        """Verify search using an external SMILES string."""
        vanillin_smiles = "O=Cc1ccc(O)c(OC)c1"
        res = similarity_engine.search(query=vanillin_smiles, top_k=10, exclude_self=True)
        assert res.query_type == "smiles"
        assert len(res.matches) > 0
        assert res.matches[0].tanimoto_similarity <= 1.0

        for match in res.matches:
            assert match.molecule_id.startswith("MOL_")
            assert len(match.canonical_smiles) > 0
            assert "molecular_weight" in match.physicochemical_descriptors

    def test_identity_based_self_match_exclusion(self, similarity_engine):
        """Verify self-match exclusion uses molecular identity (ID, canonical SMILES, InChIKey)."""
        # Query using the exact canonical SMILES of MOL_00001
        smi_00001 = similarity_engine.canonical_smiles[0]
        id_00001 = similarity_engine.molecule_ids[0]

        # With exclude_self=True
        res_excluded = similarity_engine.search(query=smi_00001, top_k=5, exclude_self=True)
        match_ids = [m.molecule_id for m in res_excluded.matches]
        match_smis = [m.canonical_smiles for m in res_excluded.matches]
        assert id_00001 not in match_ids
        assert smi_00001 not in match_smis

        # With exclude_self=False, top result must be the identical molecule with similarity 1.0
        res_included = similarity_engine.search(query=smi_00001, top_k=5, exclude_self=False)
        assert res_included.matches[0].molecule_id == id_00001
        assert res_included.matches[0].tanimoto_similarity == 1.0

    def test_threshold_filtering(self, similarity_engine):
        """Verify minimum similarity threshold cuts off lower matches."""
        res_low = similarity_engine.search(query="MOL_00002", top_k=50, min_similarity=0.0)
        res_high = similarity_engine.search(query="MOL_00002", top_k=50, min_similarity=0.40)

        assert len(res_high.matches) <= len(res_low.matches)
        for m in res_high.matches:
            assert m.tanimoto_similarity >= 0.40

    def test_deterministic_ranking(self, similarity_engine):
        """Verify tie-breaking produces deterministic and reproducible rankings."""
        res1 = similarity_engine.search(query="c1ccccc1", top_k=20)
        res2 = similarity_engine.search(query="c1ccccc1", top_k=20)

        ids1 = [(m.molecule_id, m.tanimoto_similarity) for m in res1.matches]
        ids2 = [(m.molecule_id, m.tanimoto_similarity) for m in res2.matches]
        assert ids1 == ids2

    def test_invalid_smiles_handling(self, similarity_engine):
        """Verify invalid SMILES raises InvalidSMILESError."""
        with pytest.raises(InvalidSMILESError):
            similarity_engine.search("Invalid!SMILES###")

    def test_boundary_and_validation_errors(self, similarity_engine):
        """Verify invalid parameters raise appropriate ValueErrors or KeyErrors."""
        with pytest.raises(KeyError, match="not found in the indexed dataset corpus"):
            similarity_engine.search("MOL_99999")

        with pytest.raises(ValueError, match="cannot be empty"):
            similarity_engine.search("   ")

        with pytest.raises(ValueError, match="top_k must be positive"):
            similarity_engine.search("CCO", top_k=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            similarity_engine.search("CCO", top_k=-5)

        with pytest.raises(ValueError, match="min_similarity must be within"):
            similarity_engine.search("CCO", min_similarity=-0.1)

        with pytest.raises(ValueError, match="min_similarity must be within"):
            similarity_engine.search("CCO", min_similarity=1.5)

    def test_latency_benchmarking(self, similarity_engine):
        """Verify benchmarking breaks down latency stages properly."""
        bench = similarity_engine.benchmark_search_latency(query="MOL_00010", top_k=10, n_repeats=5)
        assert bench.corpus_size == 4983
        assert bench.n_repeats == 5
        assert bench.fp_generation_ms >= 0.0
        assert bench.bulk_tanimoto_ms >= 0.0
        assert bench.ranking_packaging_ms >= 0.0
        assert bench.end_to_end_ms >= 0.0
        # Bulk Tanimoto plus ranking must be non-zero
        assert bench.bulk_tanimoto_ms + bench.ranking_packaging_ms > 0.0

