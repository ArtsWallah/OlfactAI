"""Unit tests for virtual electronic nose simulation."""

import pytest
from backend.app.digital_nose.simulator import VirtualElectronicNose


class TestVirtualElectronicNose:
    """Test suite for virtual e-nose sensor simulation."""

    def test_sensor_channels_count_and_output(self):
        """Verify 16-channel virtual sensor array generates expected response map."""
        enose = VirtualElectronicNose(n_channels=16)
        response = enose.simulate_response("CCO")

        assert len(response) == 16
        for ch, val in response.items():
            assert ch.startswith("channel_")
            assert -1.0 <= val <= 1.0  # Tanh bounds

