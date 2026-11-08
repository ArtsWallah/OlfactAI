"""CPU-compatible Molecular Message-Passing Graph Neural Network (MPNN) for olfactory classification.

Architecture:
- Atom & bond linear embedding projections
- Message passing with edge-conditioned updates and residual connections
- Global dual graph pooling (Mean + Sum)
- Fully-connected multi-label prediction head outputting 138 odor logits
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from backend.app.core.logging import get_logger
from backend.app.ml.features.graph import (
    MolecularGraph,
    MolecularGraphBatch,
    MolecularGraphFeaturizer,
)
from backend.app.ml.models.base import BaseSmellModel

logger = get_logger("ml.models.gnn")


class MessagePassingLayer(nn.Module):
    """Custom edge-conditioned message-passing layer for molecular graphs."""

    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message MLP transforming [h_src + e_feat]
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Node update MLP combining old node state and aggregated incoming messages
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        e_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Execute one round of message passing.

        Args:
            h: Node representations (Total_atoms, hidden_dim).
            edge_index: Directed graph edges (2, Total_edges).
            e_feat: Projected bond features (Total_edges, hidden_dim).

        Returns:
            Updated node representations (Total_atoms, hidden_dim).
        """
        src, dst = edge_index[0], edge_index[1]
        num_nodes = h.size(0)

        # 1. Compute messages along edges (source node + bond features)
        msg_inputs = h[src] + e_feat
        messages = self.msg_mlp(msg_inputs)

        # 2. Aggregate messages at destination nodes using fast native index_add_
        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=h.device, dtype=h.dtype)
        aggregated.index_add_(0, dst, messages)

        # 3. Update node representations with residual connection
        updated = self.update_mlp(torch.cat([h, aggregated], dim=-1))
        h_out = self.layer_norm(h + updated)
        return h_out


class MolecularGNNNetwork(nn.Module):
    """Deep graph neural network for molecular multi-label smell prediction."""

    def __init__(
        self,
        atom_dim: int = 50,
        bond_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 138,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Input projections
        self.atom_proj = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bond_proj = nn.Sequential(
            nn.Linear(bond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing layers
        self.layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Multi-label prediction head (dual pooling: mean + sum)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: MolecularGraphBatch) -> torch.Tensor:
        """Forward pass outputting raw multi-label classification logits."""
        # 1. Project atom and bond features into hidden dimension
        h = self.atom_proj(batch.atom_features)
        e_feat = self.bond_proj(batch.bond_features)

        # 2. Message-passing iterations
        for layer in self.layers:
            h = layer(h, batch.edge_index, e_feat)

        # 3. Graph-level pooling (Mean + Sum across node representations per graph)
        num_graphs = batch.num_graphs
        device = h.device

        sum_pool = torch.zeros(num_graphs, self.hidden_dim, device=device, dtype=h.dtype)
        sum_pool.index_add_(0, batch.batch_idx, h)

        # Calculate counts per graph for exact mean pooling
        counts = torch.clamp(batch.num_atoms_per_graph.float().unsqueeze(-1).to(device), min=1.0)
        mean_pool = sum_pool / counts

        graph_embedding = torch.cat([mean_pool, sum_pool], dim=-1)

        # 4. Multi-label prediction head
        logits = self.head(graph_embedding)
        return logits


class GraphDataset(Dataset):
    """PyTorch Dataset wrapper for molecular graphs and multi-label targets."""

    def __init__(self, graphs: List[MolecularGraph], labels: Optional[np.ndarray] = None):
        self.graphs = graphs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[MolecularGraph, Optional[torch.Tensor]]:
        graph = self.graphs[idx]
        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return graph, label_tensor
        return graph, None


def collate_graph_dataset(batch: List[Tuple[MolecularGraph, Optional[torch.Tensor]]]) -> Tuple[MolecularGraphBatch, Optional[torch.Tensor]]:
    """Collate function for DataLoader yielding MolecularGraphBatch."""
    graphs = [item[0] for item in batch]
    batch_graphs = MolecularGraphFeaturizer.collate_graphs(graphs)

    if batch[0][1] is not None:
        labels = torch.stack([item[1] for item in batch], dim=0)
        return batch_graphs, labels
    return batch_graphs, None


class MolecularGNNSmellModel(BaseSmellModel):
    """CPU-only Molecular Graph Neural Network wrapper implementing BaseSmellModel interface."""

    def __init__(
        self,
        name: str = "molecular_gnn_odor",
        version: str = "1.0.0",
        atom_dim: int = 50,
        bond_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 138,
        dropout: float = 0.2,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        random_state: int = 42,
    ):
        super().__init__(name=name, version=version)
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.random_state = random_state

        self.device = torch.device("cpu")  # Strictly CPU execution
        self.network = MolecularGNNNetwork(
            atom_dim=self.atom_dim,
            bond_dim=self.bond_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=self.dropout,
        )

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_epoch: int = 0
        self.train_time_sec: float = 0.0
        self.inference_time_sec: float = 0.0

    def fit(
        self,
        X: Union[List[MolecularGraph], List[str]],
        y: np.ndarray,
        target_labels: Optional[List[str]] = None,
        validation_data: Optional[Tuple[Union[List[MolecularGraph], List[str]], np.ndarray]] = None,
        **kwargs: Any,
    ) -> "MolecularGNNSmellModel":
        """Train GNN on molecular graphs using BCEWithLogitsLoss with positive weighting on CPU."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        start = time.perf_counter()

        # Handle SMILES vs MolecularGraph inputs
        featurizer = MolecularGraphFeaturizer()
        train_graphs: List[MolecularGraph] = []
        for item in X:
            if isinstance(item, str):
                train_graphs.append(featurizer(item))
            elif isinstance(item, MolecularGraph):
                train_graphs.append(item)
            else:
                raise TypeError(f"Expected SMILES string or MolecularGraph, got {type(item)}")

        self.output_dim = y.shape[1]
        self.target_labels = target_labels or [f"odor_{i}" for i in range(self.output_dim)]

        self.network = MolecularGNNNetwork(
            atom_dim=self.atom_dim,
            bond_dim=self.bond_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=self.dropout,
        )
        self.network.to(self.device)

        # Compute positive class weights strictly on training partition
        n_samples = y.shape[0]
        pos_counts = torch.tensor(np.sum(y, axis=0), dtype=torch.float32)
        neg_counts = n_samples - pos_counts
        pos_weight = neg_counts / torch.clamp(pos_counts, min=1.0)
        pos_weight = torch.clamp(pos_weight, max=50.0).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        train_dataset = GraphDataset(train_graphs, y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_graph_dataset,
        )

        val_loader = None
        if validation_data is not None:
            val_X, val_y = validation_data
            val_graphs: List[MolecularGraph] = []
            for item in val_X:
                if isinstance(item, str):
                    val_graphs.append(featurizer(item))
                elif isinstance(item, MolecularGraph):
                    val_graphs.append(item)
            val_dataset = GraphDataset(val_graphs, val_y)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_graph_dataset,
            )

        self.train_losses = []
        self.val_losses = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        logger.info(
            f"Training MolecularGNNSmellModel on CPU ({len(train_graphs)} graphs, "
            f"{self.output_dim} odor targets, {self.epochs} max epochs)..."
        )

        for epoch in range(1, self.epochs + 1):
            self.network.train()
            running_loss = 0.0
            total_batches = 0

            for batch_graphs, batch_y in train_loader:
                batch_graphs = batch_graphs.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.network(batch_graphs)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
                optimizer.step()

                running_loss += float(loss.item())
                total_batches += 1

            epoch_train_loss = running_loss / max(total_batches, 1)
            self.train_losses.append(epoch_train_loss)

            # Validation evaluation
            if val_loader is not None:
                self.network.eval()
                val_running_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for v_graphs, v_y in val_loader:
                        v_graphs = v_graphs.to(self.device)
                        v_y = v_y.to(self.device)
                        v_logits = self.network(v_graphs)
                        v_loss = criterion(v_logits, v_y)
                        val_running_loss += float(v_loss.item())
                        val_batches += 1

                epoch_val_loss = val_running_loss / max(val_batches, 1)
                self.val_losses.append(epoch_val_loss)

                if epoch_val_loss < best_val_loss - 1e-4:
                    best_val_loss = epoch_val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                    self.best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            f"Early stopping at epoch {epoch} (best epoch {self.best_epoch}, "
                            f"best val loss: {best_val_loss:.4f})"
                        )
                        break

        if best_state is not None:
            self.network.load_state_dict(best_state)
            self.network.to(self.device)

        self.is_trained = True
        self.train_time_sec = float(time.perf_counter() - start)
        logger.info(f"GNN training completed in {self.train_time_sec:.2f}s")
        return self

    def predict_logits(
        self, X: Union[List[MolecularGraph], List[str], MolecularGraphBatch]
    ) -> np.ndarray:
        """Predict unnormalized logits on CPU."""
        self._check_is_trained()
        self.network.eval()

        if isinstance(X, MolecularGraphBatch):
            with torch.no_grad():
                logits = self.network(X.to(self.device))
                return logits.cpu().numpy().astype(np.float32)

        featurizer = MolecularGraphFeaturizer()
        graphs: List[MolecularGraph] = []
        for item in X:
            if isinstance(item, str):
                graphs.append(featurizer(item))
            elif isinstance(item, MolecularGraph):
                graphs.append(item)

        loader = DataLoader(
            GraphDataset(graphs),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_graph_dataset,
        )

        all_logits = []
        with torch.no_grad():
            for batch_graphs, _ in loader:
                batch_graphs = batch_graphs.to(self.device)
                logits = self.network(batch_graphs)
                all_logits.append(logits.cpu().numpy())

        return np.vstack(all_logits).astype(np.float32)

    def predict_proba(
        self, X: Union[List[MolecularGraph], List[str], MolecularGraphBatch]
    ) -> np.ndarray:
        """Predict sigmoid probability scores on CPU."""
        start = time.perf_counter()
        logits = self.predict_logits(X)
        # Numerically stable sigmoid: 1 / (1 + exp(-x))
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        self.inference_time_sec = float(time.perf_counter() - start)
        return probs.astype(np.float32)

    def predict(
        self, X: Union[List[MolecularGraph], List[str], MolecularGraphBatch]
    ) -> np.ndarray:
        """Predict binary indicators using standard 0.50 cutoff."""
        probs = self.predict_proba(X)
        return (probs >= 0.50).astype(int)

    def save(self, path: Path) -> Path:
        """Serialize GNN weights and architecture configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": self.name,
                "version": self.version,
                "atom_dim": self.atom_dim,
                "bond_dim": self.bond_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "output_dim": self.output_dim,
                "dropout": self.dropout,
                "target_labels": self.target_labels,
                "is_trained": self.is_trained,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_epoch": self.best_epoch,
                "train_time_sec": self.train_time_sec,
                "inference_time_sec": self.inference_time_sec,
                "state_dict": self.network.state_dict(),
            },
            path,
        )
        logger.info(f"Molecular GNN model saved to {path}")
        return path

    def load(self, path: Path) -> "MolecularGNNSmellModel":
        """Load GNN weights and configuration from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.name = checkpoint["name"]
        self.version = checkpoint["version"]
        self.atom_dim = checkpoint["atom_dim"]
        self.bond_dim = checkpoint["bond_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.num_layers = checkpoint["num_layers"]
        self.output_dim = checkpoint["output_dim"]
        self.dropout = checkpoint.get("dropout", 0.2)
        self.target_labels = checkpoint["target_labels"]
        self.is_trained = checkpoint["is_trained"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.train_time_sec = checkpoint.get("train_time_sec", 0.0)
        self.inference_time_sec = checkpoint.get("inference_time_sec", 0.0)

        self.network = MolecularGNNNetwork(
            atom_dim=self.atom_dim,
            bond_dim=self.bond_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=self.dropout,
        )
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.to(self.device)
        return self

