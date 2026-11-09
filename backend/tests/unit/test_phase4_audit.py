"""Unit and regression tests for Phase 4 audit guards and verification checks."""

import json
from pathlib import Path
import numpy as np
import pytest
from rdkit import Chem
import torch

from backend.app.core.exceptions import InvalidSMILESError
from backend.app.ml.evaluation.calibration import (
    TemperatureScaler,
    compute_brier_score,
    compute_expected_calibration_error,
)
from backend.app.ml.evaluation.metrics import MultiLabelEvaluator
from backend.app.ml.evaluation.thresholds import apply_thresholds
from backend.app.ml.evaluation.uncertainty import (
    compute_decision_margin_uncertainty,
    compute_predictive_entropy,
)
from backend.app.ml.features.graph import (
    ALLOWED_ATOMS,
    ALLOWED_BOND_STEREO,
    ALLOWED_BOND_TYPES,
    ALLOWED_CHARGES,
    ALLOWED_CHIRALITIES,
    ALLOWED_DEGREES,
    ALLOWED_HYBRIDIZATIONS,
    ALLOWED_RING_SIZES,
    ALLOWED_TOTAL_HS,
    MolecularGraph,
    MolecularGraphBatch,
    MolecularGraphFeaturizer,
)
from backend.app.ml.models.gnn import MolecularGNNNetwork, MolecularGNNSmellModel


class TestPhase4AuditGuards:
    """Audit verification tests for Phase 4 GNN and calibration integrity."""

    @pytest.fixture
    def featurizer(self):
        return MolecularGraphFeaturizer()

    def test_feature_dimensions_exactness(self, featurizer):
        """Verify exact dimensional breakdown of atom (50) and bond (12) features."""
        # 13 + 7 + 6 + 6 + 1 + 6 + 1 + 6 + 4 = 50
        atom_expected = (
            len(ALLOWED_ATOMS) + 1
            + len(ALLOWED_DEGREES) + 1
            + len(ALLOWED_CHARGES) + 1
            + len(ALLOWED_HYBRIDIZATIONS) + 1
            + 1  # aromaticity
            + len(ALLOWED_TOTAL_HS) + 1
            + 1  # is_in_ring
            + len(ALLOWED_RING_SIZES)
            + len(ALLOWED_CHIRALITIES) + 1
        )
        assert atom_expected == 50
        assert featurizer.ATOM_FEATURE_DIM == 50

        # 5 + 1 + 1 + 5 = 12
        bond_expected = (
            len(ALLOWED_BOND_TYPES) + 1
            + 1  # conjugated
            + 1  # is_in_ring
            + len(ALLOWED_BOND_STEREO) + 1
        )
        assert bond_expected == 12
        assert featurizer.BOND_FEATURE_DIM == 12

    def test_bidirectional_edge_feature_symmetry(self, featurizer):
        """Verify that every undirected chemical bond produces symmetric directed pairs with identical features."""
        graph = featurizer("CCO")  # 2 bonds -> 4 directed edges
        assert graph.num_bonds == 4

        edge_dict = {}
        for k in range(graph.edge_index.shape[1]):
            u = int(graph.edge_index[0, k])
            v = int(graph.edge_index[1, k])
            edge_dict[(u, v)] = graph.bond_features[k]

        for (u, v), feat in edge_dict.items():
            assert (v, u) in edge_dict, f"Reverse edge ({v}, {u}) missing"
            assert torch.equal(feat, edge_dict[(v, u)]), f"Asymmetric bond features for ({u}, {v})"

    def test_disjoint_union_node_offsets(self, featurizer):
        """Verify that disjoint union collation offsets edge indices and maps batch_idx correctly."""
        g1 = featurizer("C")        # 1 atom, 1 self-loop edge
        g2 = featurizer("CC")       # 2 atoms, 2 directed edges
        g3 = featurizer("c1ccccc1") # 6 atoms, 12 directed edges

        batch = featurizer.collate_graphs([g1, g2, g3])

        assert batch.num_graphs == 3
        assert batch.atom_features.shape[0] == 1 + 2 + 6
        assert batch.batch_idx.tolist() == [0, 1, 1, 2, 2, 2, 2, 2, 2]

        # Verify g3 edge indices are within range [3, 8]
        g3_start_edge = g1.num_bonds + g2.num_bonds
        g3_edges = batch.edge_index[:, g3_start_edge:]
        assert g3_edges.min().item() >= 3
        assert g3_edges.max().item() <= 8

    def test_zero_support_evaluator_safety(self):
        """Verify that evaluator handles zero-support test labels safely without division-by-zero crashes."""
        evaluator = MultiLabelEvaluator()
        # Synthetic test set where label 2 has 0 positive ground truth examples
        y_true = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ])
        y_pred = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ])
        y_prob = np.array([
            [0.9, 0.1, 0.1],
            [0.1, 0.8, 0.2],
            [0.8, 0.4, 0.1],
            [0.1, 0.2, 0.05],
        ])

        results = evaluator.evaluate(
            y_true, y_pred, y_prob, target_labels=["label_0", "label_1", "zero_label"]
        )
        # Should evaluate without exception
        assert results["evaluable_labels_count"] == 2
        assert results["total_labels_count"] == 3
        assert 0.0 <= results["f1_macro"] <= 1.0

    def test_temperature_scaler_strictly_positive(self):
        """Verify that temperature scaler enforces T > 0 and transforms properly."""
        scaler = TemperatureScaler(lr=0.05, max_iter=20)
        val_logits = np.random.randn(50, 10).astype(np.float32)
        val_y = (np.random.rand(50, 10) > 0.8).astype(int)

        scaler.fit(val_logits, val_y)
        assert scaler.is_fitted
        assert scaler.temperature > 0.0

        probs = scaler.transform(val_logits)
        assert probs.shape == val_logits.shape
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_predictive_entropy_boundary_behavior(self):
        """Verify predictive entropy is 0 for confident predictions and maximal for 0.5."""
        p_certain = np.array([[0.0, 1.0], [1.0, 0.0]])
        res_certain = compute_predictive_entropy(p_certain)
        assert np.isclose(res_certain["mean_dataset_entropy"], 0.0, atol=1e-3)

        p_uncertain = np.array([[0.5, 0.5], [0.5, 0.5]])
        res_uncertain = compute_predictive_entropy(p_uncertain)
        assert np.isclose(res_uncertain["mean_dataset_entropy"], 1.0, atol=1e-3)
