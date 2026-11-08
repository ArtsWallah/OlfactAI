"""Unit tests for Phase 4 Molecular Graph Featurizer and Graph Neural Network (GNN)."""

from pathlib import Path
import numpy as np
import pytest
import torch

from backend.app.core.exceptions import InvalidSMILESError
from backend.app.ml.features.graph import (
    MolecularGraph,
    MolecularGraphBatch,
    MolecularGraphFeaturizer,
)
from backend.app.ml.models.gnn import (
    MolecularGNNNetwork,
    MolecularGNNSmellModel,
)


class TestPhase4MolecularGraph:
    """Test suite for molecular graph featurization and batching."""

    def test_graph_conversion_common_molecules(self):
        """Verify graph featurization for aliphatic, aromatic, and complex structures."""
        featurizer = MolecularGraphFeaturizer()

        # Ethanol: 3 heavy atoms, 2 bonds (4 directed edges)
        g_eth = featurizer("CCO")
        assert g_eth.num_atoms == 3
        assert g_eth.edge_index.shape == (2, 4)
        assert g_eth.atom_features.shape == (3, MolecularGraphFeaturizer.ATOM_FEATURE_DIM)
        assert g_eth.bond_features.shape == (4, MolecularGraphFeaturizer.BOND_FEATURE_DIM)

        # Benzene: 6 aromatic atoms, 6 aromatic bonds (12 directed edges)
        g_ben = featurizer("c1ccccc1")
        assert g_ben.num_atoms == 6
        assert g_ben.edge_index.shape == (2, 12)

        # Aspirin: 13 heavy atoms
        g_asp = featurizer("CC(=O)Oc1ccccc1C(=O)O")
        assert g_asp.num_atoms == 13
        assert g_asp.edge_index.shape[0] == 2
        assert g_asp.bond_features.shape[1] == MolecularGraphFeaturizer.BOND_FEATURE_DIM

    def test_graph_conversion_deterministic(self):
        """Verify identical SMILES produces bit-for-bit identical graph tensors."""
        featurizer = MolecularGraphFeaturizer()
        g1 = featurizer("COc1cc(C=O)ccc1O")
        g2 = featurizer("COc1cc(C=O)ccc1O")

        assert torch.equal(g1.atom_features, g2.atom_features)
        assert torch.equal(g1.edge_index, g2.edge_index)
        assert torch.equal(g1.bond_features, g2.bond_features)

    def test_graph_conversion_invalid_smiles_raises_error(self):
        """Verify invalid SMILES strings raise structured InvalidSMILESError."""
        featurizer = MolecularGraphFeaturizer()
        with pytest.raises(InvalidSMILESError):
            featurizer("NotAValidChemicalStructure")
        with pytest.raises(InvalidSMILESError):
            featurizer("C1CC")
        with pytest.raises(InvalidSMILESError):
            featurizer("")
        with pytest.raises(InvalidSMILESError):
            featurizer(None)

    def test_graph_batching_collate(self):
        """Verify disjoint union batch collation and edge index offset shifts."""
        featurizer = MolecularGraphFeaturizer()
        g1 = featurizer("CCO")       # 3 atoms, 4 edges
        g2 = featurizer("c1ccccc1")  # 6 atoms, 12 edges

        batch = MolecularGraphFeaturizer.collate_graphs([g1, g2])
        assert batch.num_graphs == 2
        assert batch.atom_features.shape[0] == 3 + 6
        assert batch.edge_index.shape[1] == 4 + 12
        assert batch.batch_idx.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1]

        # Edges for second graph must be shifted by 3
        second_graph_edges = batch.edge_index[:, 4:]
        assert int(second_graph_edges.min()) >= 3
        assert int(second_graph_edges.max()) == 8


class TestPhase4GNNModel:
    """Test suite for GNN architecture, forward pass, and training loop on CPU."""

    def test_gnn_model_forward_pass_cpu(self):
        """Verify GNN forward pass produces expected logit dimensions on CPU."""
        featurizer = MolecularGraphFeaturizer()
        g1 = featurizer("CCO")
        g2 = featurizer("CCCC")
        batch = MolecularGraphFeaturizer.collate_graphs([g1, g2])

        net = MolecularGNNNetwork(
            atom_dim=MolecularGraphFeaturizer.ATOM_FEATURE_DIM,
            bond_dim=MolecularGraphFeaturizer.BOND_FEATURE_DIM,
            hidden_dim=32,
            num_layers=2,
            output_dim=10,
            dropout=0.1,
        )
        logits = net(batch)
        assert logits.shape == (2, 10)
        assert logits.device.type == "cpu"

    def test_gnn_model_fit_and_early_stopping(self, tmp_path):
        """Verify GNN trains on CPU, records losses, and supports model serialization."""
        featurizer = MolecularGraphFeaturizer()
        smiles_list = ["CCO", "CCCC", "CC(=O)O", "c1ccccc1", "CCN", "CCCCO", "CC(=O)OCC", "CCOCC"]
        graphs = [featurizer(s) for s in smiles_list]
        np.random.seed(42)
        y = np.random.binomial(1, 0.3, size=(len(graphs), 4))
        labels = ["fruity", "floral", "woody", "sweet"]

        model = MolecularGNNSmellModel(
            atom_dim=MolecularGraphFeaturizer.ATOM_FEATURE_DIM,
            bond_dim=MolecularGraphFeaturizer.BOND_FEATURE_DIM,
            hidden_dim=32,
            num_layers=2,
            output_dim=4,
            dropout=0.1,
            epochs=5,
            batch_size=4,
            patience=2,
            random_state=42,
        )

        # Train on 6, validate on 2
        model.fit(
            X=graphs[:6],
            y=y[:6],
            target_labels=labels,
            validation_data=(graphs[6:], y[6:]),
        )

        assert model.is_trained is True
        assert len(model.train_losses) > 0
        assert len(model.val_losses) > 0
        assert model.train_time_sec > 0.0

        probs = model.predict_proba(graphs[6:])
        assert probs.shape == (2, 4)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

        # Checkpoint serialization roundtrip
        save_file = tmp_path / "gnn_test.pt"
        model.save(save_file)

        loaded = MolecularGNNSmellModel(
            atom_dim=MolecularGraphFeaturizer.ATOM_FEATURE_DIM,
            bond_dim=MolecularGraphFeaturizer.BOND_FEATURE_DIM,
            hidden_dim=32,
            num_layers=2,
            output_dim=4,
        ).load(save_file)

        loaded_probs = loaded.predict_proba(graphs[6:])
        np.testing.assert_allclose(probs, loaded_probs, atol=1e-5)

