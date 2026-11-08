"""Machine learning model architectures and registry."""

from backend.app.ml.models.base import BaseSmellModel
from backend.app.ml.models.dummy import DummyFrequencySmellModel
from backend.app.ml.models.gnn import (
    MessagePassingLayer,
    MolecularGNNNetwork,
    MolecularGNNSmellModel,
)
from backend.app.ml.models.pytorch_mlp import OdorMLPNetwork, PyTorchMLPSmellModel
from backend.app.ml.models.random_forest import RandomForestSmellModel
from backend.app.ml.models.registry import ModelRegistry
from backend.app.ml.models.xgboost_model import XGBoostSmellModel

__all__ = [
    "BaseSmellModel",
    "DummyFrequencySmellModel",
    "RandomForestSmellModel",
    "XGBoostSmellModel",
    "OdorMLPNetwork",
    "PyTorchMLPSmellModel",
    "MessagePassingLayer",
    "MolecularGNNNetwork",
    "MolecularGNNSmellModel",
    "ModelRegistry",
]
