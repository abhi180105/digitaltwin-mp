"""
tests/test_topology.py
======================
Unit tests for the campus topology module.
"""

import pytest
from digital_twin.topology import create_campus_topology, CampusTopology, DEFAULT_DEVICES
from digital_twin.devices import Device


@pytest.fixture
def topo() -> CampusTopology:
    return create_campus_topology()


class TestTopologyCreation:
    def test_expected_device_count(self, topo):
        assert topo.device_count() == len(DEFAULT_DEVICES)

    def test_expected_connection_count(self, topo):
        # Adjust if DEFAULT_EDGES changes
        assert topo.edge_count() > 0

    def test_all_devices_present(self, topo):
        ids = {d.device_id for d in topo.all_devices()}
        expected = {d["device_id"] for d in DEFAULT_DEVICES}
        assert ids == expected


class TestDeviceQueries:
    def test_get_device_known(self, topo):
        dev = topo.get_device("CCTV-01")
        assert dev.name == "Academic CCTV 01"

    def test_get_device_unknown(self, topo):
        with pytest.raises(KeyError):
            topo.get_device("DOES-NOT-EXIST")

    def test_get_device_by_ip(self, topo):
        dev = topo.get_device_by_ip("192.168.20.10")
        assert dev is not None
        assert dev.device_id == "CCTV-01"

    def test_get_device_by_ip_unknown(self, topo):
        assert topo.get_device_by_ip("1.2.3.4") is None

    def test_get_neighbors_firewall(self, topo):
        neighbors = topo.get_neighbors("FW-01")
        neighbor_ids = {d.device_id for d in neighbors}
        assert "RTR-01" in neighbor_ids

    def test_get_neighbors_unknown(self, topo):
        with pytest.raises(KeyError):
            topo.get_neighbors("FAKE-01")


class TestStateUpdates:
    def test_update_device_status(self, topo):
        topo.update_device_status("CCTV-01", "WARNING", reason="Test")
        dev = topo.get_device("CCTV-01")
        assert dev.status == "WARNING"

    def test_update_risk_score(self, topo):
        topo.update_risk_score("SRV-WEB", 0.65)
        dev = topo.get_device("SRV-WEB")
        assert dev.risk_score == pytest.approx(0.65, abs=0.001)

    def test_summary_keys(self, topo):
        s = topo.summary()
        assert "total_devices" in s
        assert "total_connections" in s
        assert "status_counts" in s


class TestAddRemoveDevices:
    def test_add_duplicate_raises(self, topo):
        dev = Device(
            device_id="CCTV-01", name="Dup", device_type="IoT",
            ip_address="1.1.1.1", mac_address="AA:AA:AA:AA:AA:AA",
            location="X", department="X", vlan="X",
        )
        with pytest.raises(ValueError):
            topo.add_device(dev)

    def test_remove_device(self, topo):
        topo.remove_device("IOT-HOSTEL")
        with pytest.raises(KeyError):
            topo.get_device("IOT-HOSTEL")

    def test_connect_unknown_device_raises(self, topo):
        with pytest.raises(KeyError):
            topo.connect_devices("CCTV-01", "GHOST-01")
