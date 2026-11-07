"""PyTorch CPU multi-layer perceptron (MLP) for multi-label odor prediction."""

from pathlib import Path
from typing import Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.app.core.logging import get_logger
from backend.app.ml.models.base import BaseSmellModel

logger = get_logger("ml.models.pytorch_mlp")


class OdorMLPNetwork(nn.Module):
    """Deep neural network architecture for chemical odor multi-label classification."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [512, 256, 128], dropout: float = 0.2):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.BatchNorm1d(h_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_d = h_d
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting raw logits."""
        return self.net(x)


class PyTorchMLPSmellModel(BaseSmellModel):
    """PyTorch MLP wrapper implementing BaseSmellModel interface running on CPU."""

    def __init__(
        self,
        name: str = "pytorch_mlp_odor",
        version: str = "0.1.0",
        input_dim: int = 2048,
        output_dim: int = 10,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
    ):
        super().__init__(name=name, version=version)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cpu")  # Strict CPU execution
        self.network = OdorMLPNetwork(input_dim=self.input_dim, output_dim=self.output_dim)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "PyTorchMLPSmellModel":
        """Train neural network on CPU using BCEWithLogitsLoss."""
        logger.info(f"Training PyTorchMLPSmellModel on CPU ({X.shape[0]} samples)...")
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]
        self.network = OdorMLPNetwork(self.input_dim, self.output_dim)
        self.network.to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.network.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.network(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.is_trained = True
        self.target_labels = target_labels or [f"odor_{i}" for i in range(self.output_dim)]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict sigmoid probabilities on CPU."""
        self._check_is_trained()
        self.network.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.network(tensor_x)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using standard 0.5 threshold."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def save(self, path: Path) -> Path:
        """Save PyTorch model weights and architecture configuration."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": self.name,
                "version": self.version,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "target_labels": self.target_labels,
                "is_trained": self.is_trained,
                "state_dict": self.network.state_dict(),
            },
            path,
        )
        logger.info(f"PyTorch model saved to {path}")
        return path

    def load(self, path: Path) -> "PyTorchMLPSmellModel":
        """Load PyTorch model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.name = checkpoint["name"]
        self.version = checkpoint["version"]
        self.input_dim = checkpoint["input_dim"]
        self.output_dim = checkpoint["output_dim"]
        self.target_labels = checkpoint["target_labels"]
        self.is_trained = checkpoint["is_trained"]
        self.network = OdorMLPNetwork(self.input_dim, self.output_dim)
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.to(self.device)
        return self

