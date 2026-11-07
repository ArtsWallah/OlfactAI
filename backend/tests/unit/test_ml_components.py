"""Unit tests for ML base classes, feature pipeline, and untrained model guards."""

import pytest
import numpy as np
from backend.app.core.exceptions import ModelNotTrainedError
from backend.app.ml.models.random_forest import RandomForestSmellModel
from backend.app.ml.models.xgboost_model import XGBoostSmellModel
from backend.app.ml.models.pytorch_mlp import PyTorchMLPSmellModel
from backend.app.ml.features.pipeline import MolecularFeaturePipeline


class TestMLComponents:
    """Test suite for ML components and interfaces."""

    def test_feature_pipeline_dimensions(self):
        """Verify molecular feature pipeline produces expected vector length (2048 bits + 9 descriptors)."""
        pipeline = MolecularFeaturePipeline(morgan_radius=2, morgan_nbits=2048, include_descriptors=True)
        vec = pipeline.transform_single("CCO")
        assert vec.shape == (2048 + 9,)
        assert isinstance(vec, np.ndarray)

    def test_untrained_random_forest_raises_error(self):
        """Verify untrained Random Forest model blocks inference."""
        model = RandomForestSmellModel()
        dummy_x = np.zeros((1, 2048))
        with pytest.raises(ModelNotTrainedError):
            model.predict(dummy_x)

    def test_untrained_xgboost_raises_error(self):
        """Verify untrained XGBoost model blocks inference."""
        model = XGBoostSmellModel()
        dummy_x = np.zeros((1, 2048))
        with pytest.raises(ModelNotTrainedError):
            model.predict(dummy_x)

    def test_untrained_pytorch_mlp_raises_error(self):
        """Verify untrained PyTorch MLP model blocks inference."""
        model = PyTorchMLPSmellModel()
        dummy_x = np.zeros((1, 2048))
        with pytest.raises(ModelNotTrainedError):
            model.predict(dummy_x)

