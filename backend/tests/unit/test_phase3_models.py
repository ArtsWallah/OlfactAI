"""Unit tests for Phase 3 baseline models (Dummy, RF, XGBoost, PyTorch MLP)."""

from pathlib import Path
import numpy as np
import pytest
import torch

from backend.app.ml.models.dummy import DummyFrequencySmellModel
from backend.app.ml.models.pytorch_mlp import PyTorchMLPSmellModel
from backend.app.ml.models.random_forest import RandomForestSmellModel
from backend.app.ml.models.xgboost_model import XGBoostSmellModel


@pytest.fixture
def synthetic_multi_label_data():
    """Generate small synthetic dataset for testing model mechanics."""
    np.random.seed(42)
    n_samples = 40
    n_features = 25
    n_classes = 4

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Binary multi-label target with varying class frequencies
    y = np.random.binomial(1, p=[0.7, 0.4, 0.1, 0.0], size=(n_samples, n_classes))  # Note 4th class is all zeros
    target_labels = ["frequent", "moderate", "rare", "zero_positive"]
    return X, y, target_labels


class TestPhase3Models:
    """Test suite for Phase 3 model implementations."""

    def test_dummy_frequency_model(self, synthetic_multi_label_data, tmp_path):
        """Verify DummyFrequencySmellModel learns empirical priors."""
        X, y, labels = synthetic_multi_label_data
        model = DummyFrequencySmellModel(strategy="prior")
        model.fit(X, y, target_labels=labels)

        assert model.is_trained is True
        assert model.train_time_sec >= 0.0

        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 4)
        np.testing.assert_allclose(probs[0], np.mean(y, axis=0), atol=1e-5)

        preds = model.predict(X[:5])
        assert preds.shape == (5, 4)
        assert set(np.unique(preds)).issubset({0, 1})

        # Save and load roundtrip
        save_file = tmp_path / "dummy.joblib"
        model.save(save_file)
        loaded = DummyFrequencySmellModel().load(save_file)
        assert np.array_equal(loaded.predict(X[:5]), preds)

    def test_random_forest_model(self, synthetic_multi_label_data, tmp_path):
        """Verify RandomForestSmellModel fits multi-label and extracts probabilities safely."""
        X, y, labels = synthetic_multi_label_data
        model = RandomForestSmellModel(n_estimators=10, max_depth=4, random_state=42)
        model.fit(X, y, target_labels=labels)

        assert model.is_trained is True
        assert model.train_time_sec > 0.0

        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 4)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        # Zero positive class should output 0.0
        assert np.all(probs[:, 3] == 0.0)

        preds = model.predict(X[:5])
        assert preds.shape == (5, 4)
        assert set(np.unique(preds)).issubset({0, 1})

        # Save and load roundtrip
        save_file = tmp_path / "rf.joblib"
        model.save(save_file)
        loaded = RandomForestSmellModel().load(save_file)
        assert np.array_equal(loaded.predict(X[:5]), preds)

    def test_xgboost_model(self, synthetic_multi_label_data, tmp_path):
        """Verify XGBoostSmellModel uses CPU hist, handles single-class columns, and predicts probabilities."""
        X, y, labels = synthetic_multi_label_data
        model = XGBoostSmellModel(n_estimators=10, max_depth=3, n_jobs=1, random_state=42)
        model.fit(X, y, target_labels=labels)

        assert model.is_trained is True
        assert model.train_time_sec > 0.0
        assert len(model.estimators) == 4

        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 4)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        # Fourth column had 0 positive examples, should predict 0.0
        assert np.all(probs[:, 3] == 0.0)

        preds = model.predict(X[:5])
        assert preds.shape == (5, 4)
        assert set(np.unique(preds)).issubset({0, 1})

        # Save and load roundtrip
        save_file = tmp_path / "xgb.joblib"
        model.save(save_file)
        loaded = XGBoostSmellModel().load(save_file)
        assert np.array_equal(loaded.predict(X[:5]), preds)

    def test_pytorch_mlp_model(self, synthetic_multi_label_data, tmp_path):
        """Verify PyTorchMLPSmellModel trains strictly on CPU with loss history and early stopping."""
        X, y, labels = synthetic_multi_label_data
        model = PyTorchMLPSmellModel(
            input_dim=X.shape[1],
            output_dim=y.shape[1],
            hidden_dims=[32, 16],
            dropout=0.1,
            epochs=10,
            batch_size=8,
            patience=3,
            random_state=42,
        )

        assert model.device == torch.device("cpu")

        # Split into train and val
        X_tr, y_tr = X[:30], y[:30]
        X_v, y_v = X[30:], y[30:]

        model.fit(X_tr, y_tr, target_labels=labels, validation_data=(X_v, y_v))

        assert model.is_trained is True
        assert len(model.train_losses) > 0
        assert len(model.val_losses) > 0
        assert model.train_time_sec > 0.0

        probs = model.predict_proba(X_v)
        assert probs.shape == (10, 4)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

        preds = model.predict(X_v)
        assert preds.shape == (10, 4)
        assert set(np.unique(preds)).issubset({0, 1})

        # Save and load roundtrip
        save_file = tmp_path / "mlp.pt"
        model.save(save_file)
        loaded = PyTorchMLPSmellModel().load(save_file)
        assert np.allclose(loaded.predict_proba(X_v), probs, atol=1e-5)

