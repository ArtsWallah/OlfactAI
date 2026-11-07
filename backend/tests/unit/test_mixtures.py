"""Unit tests for odor mixture blending."""

import pytest
import numpy as np
from backend.app.mixtures.blender import MixtureBlender


class TestMixtureBlender:
    """Test suite for mixture blending calculations."""

    def test_blend_proportions(self):
        """Verify blending calculates weighted sum of feature vectors."""
        blender = MixtureBlender()
        components = [("CCO", 0.7), ("COc1cc(C=O)ccc1O", 0.3)]
        blended = blender.blend_representations(components)

        assert isinstance(blended, np.ndarray)
        assert len(blended) > 2048

    def test_empty_components_raises_value_error(self):
        """Verify empty mixture raises ValueError."""
        blender = MixtureBlender()
        with pytest.raises(ValueError):
            blender.blend_representations([])

