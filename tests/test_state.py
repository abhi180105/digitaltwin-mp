"""
tests/test_state.py
===================
Unit tests for the Digital Twin state manager.
"""

import pytest
from digital_twin.topology import create_campus_topology
from digital_twin.state import DigitalTwinStateManager, risk_to_status


@pytest.fixture
def mgr():
    topo = create_campus_topology()
    return DigitalTwinStateManager(topo)


class TestRiskToStatus:
    def test_low_risk_normal(self):
        assert risk_to_status(0.1) == "NORMAL"

    def test_medium_risk_warning(self):
        assert risk_to_status(0.45) == "WARNING"

    def test_high_risk_suspicious(self):
        assert risk_to_status(0.70) == "SUSPICIOUS"

    def test_very_high_risk_attack(self):
        assert risk_to_status(0.90) == "UNDER_ATTACK"


class TestStateManager:
    def test_update_device_state(self, mgr):
        event = mgr.update_device_state("CCTV-01", "WARNING", reason="Test")
        assert event["new_status"] == "WARNING"
        assert event["device_id"] == "CCTV-01"

    def test_event_recorded(self, mgr):
        mgr.update_device_state("CCTV-01", "WARNING")
        history = mgr.get_event_history()
        assert len(history) >= 1

    def test_update_risk_auto_transition(self, mgr):
        mgr.update_risk_score("CCTV-01", 0.82, reason="ML anomaly", auto_transition=True)
        dev_state = mgr.get_device_state("CCTV-01")
        assert dev_state["status"] == "UNDER_ATTACK"

    def test_update_risk_no_transition(self, mgr):
        mgr.update_risk_score("RFID-01", 0.1, auto_transition=False)
        dev_state = mgr.get_device_state("RFID-01")
        assert dev_state["status"] == "NORMAL"

    def test_record_anomaly(self, mgr):
        event = mgr.record_anomaly("SRV-WEB", 0.88, "ATTACK", 0.91, "10.0.99.1")
        assert event["prediction"] == "ATTACK"
        assert event["device_id"] == "SRV-WEB"

    def test_reset_device(self, mgr):
        mgr.update_device_state("CCTV-01", "WARNING")
        mgr.reset_device("CCTV-01")
        dev_state = mgr.get_device_state("CCTV-01")
        assert dev_state["status"] == "NORMAL"
        assert dev_state["risk_score"] == 0.0

    def test_get_compromised_empty_initially(self, mgr):
        compromised = mgr.get_compromised_devices()
        assert isinstance(compromised, list)
        assert all(d["status"] in {"SUSPICIOUS", "UNDER_ATTACK"} for d in compromised)

    def test_event_history_limit(self, mgr):
        for i in range(10):
            mgr.update_device_state("CCTV-01", "WARNING" if i % 2 == 0 else "NORMAL")
        history = mgr.get_event_history(limit=3)
        assert len(history) <= 3

    def test_topology_summary(self, mgr):
        s = mgr.get_topology_summary()
        assert "total_devices" in s
        assert s["total_devices"] > 0
