"""PyTorch CPU multi-layer perceptron (MLP) for multi-label odor prediction."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.app.core.logging import get_logger
from backend.app.ml.models.base import BaseSmellModel

logger = get_logger("ml.models.pytorch_mlp")


class OdorMLPNetwork(nn.Module):
    """Deep neural network architecture for chemical odor multi-label classification."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

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
        input_dim: int = 2057,
        output_dim: int = 138,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        random_state: int = 42,
    ):
        super().__init__(name=name, version=version)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [512, 256, 128]
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.random_state = random_state

        self.device = torch.device("cpu")  # Strict CPU execution
        self.network = OdorMLPNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_time_sec: float = 0.0
        self.inference_time_sec: float = 0.0
        self.best_epoch: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_labels: Optional[List[str]] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "PyTorchMLPSmellModel":
        """Train neural network on CPU using BCEWithLogitsLoss with positive weighting."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        start = time.perf_counter()
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]
        self.target_labels = target_labels or [f"odor_{i}" for i in range(self.output_dim)]

        self.network = OdorMLPNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )
        self.network.to(self.device)

        # Compute positive class weights on training split
        n_samples = y.shape[0]
        pos_counts = torch.tensor(np.sum(y, axis=0), dtype=torch.float32)
        neg_counts = n_samples - pos_counts
        pos_weight = neg_counts / torch.clamp(pos_counts, min=1.0)
        # Cap max pos_weight to avoid instability on ultra-rare labels
        pos_weight = torch.clamp(pos_weight, max=50.0).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_tensor_x = None
        val_tensor_y = None
        if validation_data is not None:
            val_x, val_y = validation_data
            val_tensor_x = torch.tensor(val_x, dtype=torch.float32).to(self.device)
            val_tensor_y = torch.tensor(val_y, dtype=torch.float32).to(self.device)

        self.train_losses = []
        self.val_losses = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        logger.info(
            f"Training PyTorchMLPSmellModel on CPU ({n_samples} samples, {self.output_dim} labels, "
            f"{self.epochs} max epochs)..."
        )

        for epoch in range(1, self.epochs + 1):
            self.network.train()
            running_loss = 0.0
            total_batches = 0

            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.network(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                total_batches += 1

            epoch_train_loss = running_loss / max(total_batches, 1)
            self.train_losses.append(epoch_train_loss)

            if val_tensor_x is not None:
                self.network.eval()
                with torch.no_grad():
                    val_logits = self.network(val_tensor_x)
                    val_loss = float(criterion(val_logits, val_tensor_y).item())
                self.val_losses.append(val_loss)

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                    self.best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch} (best epoch {self.best_epoch}, "
                            f"best val loss {best_val_loss:.4f})"
                        )
                        break

        if best_state is not None:
            self.network.load_state_dict(best_state)
            self.network.to(self.device)

        self.is_trained = True
        self.train_time_sec = float(time.perf_counter() - start)
        logger.info(f"PyTorchMLPSmellModel training completed in {self.train_time_sec:.2f}s")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict sigmoid probabilities on CPU."""
        self._check_is_trained()
        start = time.perf_counter()
        self.network.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.network(tensor_x)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        self.inference_time_sec = float(time.perf_counter() - start)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using standard 0.5 threshold."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def save(self, path: Path) -> Path:
        """Save PyTorch model weights, architecture configuration, and training logs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": self.name,
                "version": self.version,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "target_labels": self.target_labels,
                "is_trained": self.is_trained,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "train_time_sec": self.train_time_sec,
                "inference_time_sec": self.inference_time_sec,
                "best_epoch": self.best_epoch,
                "state_dict": self.network.state_dict(),
            },
            path,
        )
        logger.info(f"PyTorch model saved to {path}")
        return path

    def load(self, path: Path) -> "PyTorchMLPSmellModel":
        """Load PyTorch model weights and state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.name = checkpoint["name"]
        self.version = checkpoint["version"]
        self.input_dim = checkpoint["input_dim"]
        self.output_dim = checkpoint["output_dim"]
        self.hidden_dims = checkpoint.get("hidden_dims", [512, 256, 128])
        self.dropout = checkpoint.get("dropout", 0.2)
        self.target_labels = checkpoint["target_labels"]
        self.is_trained = checkpoint["is_trained"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.train_time_sec = checkpoint.get("train_time_sec", 0.0)
        self.inference_time_sec = checkpoint.get("inference_time_sec", 0.0)
        self.best_epoch = checkpoint.get("best_epoch", 0)

        self.network = OdorMLPNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.to(self.device)
        return self
