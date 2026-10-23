"""
tests/test_devices.py
=====================
Unit tests for the Device data model.
"""

import pytest
from datetime import datetime
from digital_twin.devices import Device, VALID_STATUSES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_device() -> Device:
    return Device(
        device_id="CCTV-01",
        name="Academic CCTV 01",
        device_type="IoT",
        ip_address="192.168.20.10",
        mac_address="AA:BB:CC:DD:EE:01",
        location="Academic Block",
        department="Security",
        vlan="IOT",
    )


# ---------------------------------------------------------------------------
# Creation tests
# ---------------------------------------------------------------------------

class TestDeviceCreation:
    def test_valid_creation(self, sample_device):
        assert sample_device.device_id == "CCTV-01"
        assert sample_device.status == "NORMAL"
        assert sample_device.risk_score == 0.0
        assert sample_device.last_seen is None

    def test_invalid_device_type(self):
        with pytest.raises(ValueError, match="Invalid device_type"):
            Device(
                device_id="X-01", name="X", device_type="Unknown",
                ip_address="0.0.0.0", mac_address="00:00:00:00:00:00",
                location="X", department="X", vlan="X",
            )

    def test_invalid_status(self):
        with pytest.raises(ValueError, match="Invalid status"):
            Device(
                device_id="X-01", name="X", device_type="Endpoint",
                ip_address="0.0.0.0", mac_address="00:00:00:00:00:00",
                location="X", department="X", vlan="X",
                status="COMPROMISED",
            )

    def test_risk_score_out_of_range(self):
        with pytest.raises(ValueError, match="risk_score"):
            Device(
                device_id="X-01", name="X", device_type="Endpoint",
                ip_address="0.0.0.0", mac_address="00:00:00:00:00:00",
                location="X", department="X", vlan="X",
                risk_score=1.5,
            )

    def test_empty_device_id(self):
        with pytest.raises(ValueError, match="device_id"):
            Device(
                device_id="", name="X", device_type="Endpoint",
                ip_address="0.0.0.0", mac_address="00:00:00:00:00:00",
                location="X", department="X", vlan="X",
            )


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------

class TestStateTransitions:
    def test_normal_to_suspicious(self, sample_device):
        sample_device.transition_to("SUSPICIOUS", reason="Test")
        assert sample_device.status == "SUSPICIOUS"

    def test_suspicious_to_under_attack(self, sample_device):
        sample_device.transition_to("SUSPICIOUS")
        sample_device.transition_to("UNDER_ATTACK")
        assert sample_device.status == "UNDER_ATTACK"

    def test_illegal_transition(self, sample_device):
        # NORMAL → UNDER_ATTACK is not directly allowed
        with pytest.raises(ValueError, match="Transition"):
            sample_device.transition_to("UNDER_ATTACK")

    def test_event_recorded(self, sample_device):
        sample_device.transition_to("WARNING")
        history = sample_device.event_history
        assert len(history) == 1
        assert history[0]["previous_status"] == "NORMAL"
        assert history[0]["new_status"] == "WARNING"

    def test_last_seen_updated(self, sample_device):
        before = datetime.now()
        sample_device.transition_to("WARNING")
        assert sample_device.last_seen is not None
        assert sample_device.last_seen >= before


# ---------------------------------------------------------------------------
# Risk score tests
# ---------------------------------------------------------------------------

class TestRiskScore:
    def test_update_valid(self, sample_device):
        sample_device.update_risk_score(0.75)
        assert sample_device.risk_score == 0.75

    def test_clamp_above_one(self, sample_device):
        sample_device.update_risk_score(1.5)
        assert sample_device.risk_score == 1.0

    def test_clamp_below_zero(self, sample_device):
        sample_device.update_risk_score(-0.3)
        assert sample_device.risk_score == 0.0

    def test_is_compromised_false(self, sample_device):
        assert not sample_device.is_compromised

    def test_is_compromised_true(self, sample_device):
        sample_device.transition_to("SUSPICIOUS")
        assert sample_device.is_compromised


# ---------------------------------------------------------------------------
# Serialisation tests
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_to_dict(self, sample_device):
        d = sample_device.to_dict()
        assert d["device_id"] == "CCTV-01"
        assert "risk_score" in d
        assert "status" in d

    def test_from_dict_roundtrip(self, sample_device):
        d = sample_device.to_dict()
        restored = Device.from_dict(d)
        assert restored.device_id == sample_device.device_id
        assert restored.status == sample_device.status
