"""
digital_twin/topology.py
========================
Campus network topology for the AI-Driven Digital Twin Framework.

Builds a NetworkX DiGraph representing a realistic educational institution
network with ~26 logical devices across five network zones.
Device metadata is stored on each graph node; connection metadata on each edge.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import networkx as nx

from digital_twin.devices import Device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default device catalogue — realistic but entirely fictional
# ---------------------------------------------------------------------------

DEFAULT_DEVICES: list[dict] = [
    # ── External ────────────────────────────────────────────────────────────
    {
        "device_id": "EXT-01",
        "name": "Internet Gateway",
        "device_type": "External",
        "ip_address": "0.0.0.0",
        "mac_address": "00:00:00:00:00:00",
        "location": "External",
        "department": "External",
        "vlan": "EXTERNAL",
    },
    # ── Security Layer ───────────────────────────────────────────────────────
    {
        "device_id": "FW-01",
        "name": "Perimeter Firewall",
        "device_type": "Security",
        "ip_address": "10.0.0.1",
        "mac_address": "AA:00:00:00:00:01",
        "location": "Server Room",
        "department": "IT",
        "vlan": "DMZ",
    },
    {
        "device_id": "IDS-01",
        "name": "IDS/Monitoring Node",
        "device_type": "Monitoring",
        "ip_address": "10.0.0.2",
        "mac_address": "AA:00:00:00:00:02",
        "location": "Server Room",
        "department": "IT Security",
        "vlan": "MANAGEMENT",
    },
    # ── Core Layer ───────────────────────────────────────────────────────────
    {
        "device_id": "RTR-01",
        "name": "Core Router",
        "device_type": "Router",
        "ip_address": "192.168.0.1",
        "mac_address": "AA:BB:00:00:00:01",
        "location": "Server Room",
        "department": "IT",
        "vlan": "CORE",
    },
    # ── Server Layer ─────────────────────────────────────────────────────────
    {
        "device_id": "SRV-LMS",
        "name": "LMS Server",
        "device_type": "Server",
        "ip_address": "192.168.10.10",
        "mac_address": "AA:BB:10:00:00:01",
        "location": "Server Room",
        "department": "Academic",
        "vlan": "SERVERS",
    },
    {
        "device_id": "SRV-WEB",
        "name": "Web Server",
        "device_type": "Server",
        "ip_address": "192.168.10.11",
        "mac_address": "AA:BB:10:00:00:02",
        "location": "Server Room",
        "department": "IT",
        "vlan": "DMZ",
    },
    {
        "device_id": "SRV-DB",
        "name": "Database Server",
        "device_type": "Server",
        "ip_address": "192.168.10.12",
        "mac_address": "AA:BB:10:00:00:03",
        "location": "Server Room",
        "department": "IT",
        "vlan": "SERVERS",
    },
    {
        "device_id": "SRV-AUTH",
        "name": "Authentication Server",
        "device_type": "Server",
        "ip_address": "192.168.10.13",
        "mac_address": "AA:BB:10:00:00:04",
        "location": "Server Room",
        "department": "IT Security",
        "vlan": "SERVERS",
    },
    {
        "device_id": "SRV-FILE",
        "name": "File Server",
        "device_type": "Server",
        "ip_address": "192.168.10.14",
        "mac_address": "AA:BB:10:00:00:05",
        "location": "Server Room",
        "department": "IT",
        "vlan": "SERVERS",
    },
    # ── Academic Network ──────────────────────────────────────────────────────
    {
        "device_id": "SW-ACAD",
        "name": "Academic Switch",
        "device_type": "Switch",
        "ip_address": "192.168.20.1",
        "mac_address": "AA:BB:20:00:00:01",
        "location": "Academic Block",
        "department": "IT",
        "vlan": "ACADEMIC",
    },
    {
        "device_id": "FAC-PC01",
        "name": "Faculty PC 01",
        "device_type": "Endpoint",
        "ip_address": "192.168.20.101",
        "mac_address": "AA:BB:20:01:00:01",
        "location": "Academic Block",
        "department": "CSE",
        "vlan": "ACADEMIC",
    },
    {
        "device_id": "FAC-PC02",
        "name": "Faculty PC 02",
        "device_type": "Endpoint",
        "ip_address": "192.168.20.102",
        "mac_address": "AA:BB:20:01:00:02",
        "location": "Academic Block",
        "department": "ECE",
        "vlan": "ACADEMIC",
    },
    {
        "device_id": "STU-PC01",
        "name": "Student PC 01",
        "device_type": "Endpoint",
        "ip_address": "192.168.20.201",
        "mac_address": "AA:BB:20:02:00:01",
        "location": "Academic Block",
        "department": "Labs",
        "vlan": "ACADEMIC",
    },
    {
        "device_id": "CCTV-01",
        "name": "Academic CCTV 01",
        "device_type": "IoT",
        "ip_address": "192.168.20.10",
        "mac_address": "AA:BB:CC:DD:EE:01",
        "location": "Academic Block",
        "department": "Security",
        "vlan": "IOT",
    },
    {
        "device_id": "RFID-01",
        "name": "RFID Attendance Reader",
        "device_type": "IoT",
        "ip_address": "192.168.20.11",
        "mac_address": "AA:BB:CC:DD:EE:02",
        "location": "Academic Block",
        "department": "Administration",
        "vlan": "IOT",
    },
    {
        "device_id": "AP-ACAD",
        "name": "Academic Wi-Fi AP",
        "device_type": "AccessPoint",
        "ip_address": "192.168.20.50",
        "mac_address": "AA:BB:20:FF:00:01",
        "location": "Academic Block",
        "department": "IT",
        "vlan": "ACADEMIC",
    },
    # ── Administration Network ────────────────────────────────────────────────
    {
        "device_id": "SW-ADMIN",
        "name": "Admin Switch",
        "device_type": "Switch",
        "ip_address": "192.168.30.1",
        "mac_address": "AA:BB:30:00:00:01",
        "location": "Administration Block",
        "department": "IT",
        "vlan": "ADMIN",
    },
    {
        "device_id": "ADMIN-PC01",
        "name": "Admin PC 01",
        "device_type": "Endpoint",
        "ip_address": "192.168.30.101",
        "mac_address": "AA:BB:30:01:00:01",
        "location": "Administration Block",
        "department": "HR",
        "vlan": "ADMIN",
    },
    {
        "device_id": "ADMIN-SRV",
        "name": "Admin Server",
        "device_type": "Server",
        "ip_address": "192.168.30.10",
        "mac_address": "AA:BB:30:00:00:10",
        "location": "Administration Block",
        "department": "Administration",
        "vlan": "ADMIN",
    },
    {
        "device_id": "ACS-01",
        "name": "Access Control System",
        "device_type": "IoT",
        "ip_address": "192.168.30.20",
        "mac_address": "AA:BB:CC:DD:FF:01",
        "location": "Administration Block",
        "department": "Security",
        "vlan": "IOT",
    },
    # ── Hostel Network ────────────────────────────────────────────────────────
    {
        "device_id": "SW-HOSTEL",
        "name": "Hostel Switch",
        "device_type": "Switch",
        "ip_address": "192.168.40.1",
        "mac_address": "AA:BB:40:00:00:01",
        "location": "Hostel Block",
        "department": "IT",
        "vlan": "HOSTEL",
    },
    {
        "device_id": "STU-DEV01",
        "name": "Student Device 01",
        "device_type": "Endpoint",
        "ip_address": "192.168.40.101",
        "mac_address": "AA:BB:40:01:00:01",
        "location": "Hostel Block",
        "department": "Students",
        "vlan": "HOSTEL",
    },
    {
        "device_id": "STU-DEV02",
        "name": "Student Device 02",
        "device_type": "Endpoint",
        "ip_address": "192.168.40.102",
        "mac_address": "AA:BB:40:01:00:02",
        "location": "Hostel Block",
        "department": "Students",
        "vlan": "HOSTEL",
    },
    {
        "device_id": "AP-HOSTEL",
        "name": "Hostel Wi-Fi AP",
        "device_type": "AccessPoint",
        "ip_address": "192.168.40.50",
        "mac_address": "AA:BB:40:FF:00:01",
        "location": "Hostel Block",
        "department": "IT",
        "vlan": "HOSTEL",
    },
    {
        "device_id": "IOT-HOSTEL",
        "name": "Hostel IoT Device",
        "device_type": "IoT",
        "ip_address": "192.168.40.20",
        "mac_address": "AA:BB:CC:40:00:01",
        "location": "Hostel Block",
        "department": "Facilities",
        "vlan": "IOT",
    },
    {
        "device_id": "CCTV-02",
        "name": "Hostel CCTV",
        "device_type": "IoT",
        "ip_address": "192.168.40.21",
        "mac_address": "AA:BB:CC:40:00:02",
        "location": "Hostel Block",
        "department": "Security",
        "vlan": "IOT",
    },
]

# (source_id, target_id, edge_attrs)
DEFAULT_EDGES: list[tuple[str, str, dict]] = [
    # External → Security
    ("EXT-01",   "FW-01",    {"bandwidth": "1Gbps",  "protocol": "Ethernet", "zone": "perimeter"}),
    # Firewall → IDS (span/mirror)
    ("FW-01",    "IDS-01",   {"bandwidth": "1Gbps",  "protocol": "Ethernet", "zone": "management"}),
    # Firewall → Core Router
    ("FW-01",    "RTR-01",   {"bandwidth": "10Gbps", "protocol": "Ethernet", "zone": "core"}),
    # Core → Servers
    ("RTR-01",   "SRV-LMS",  {"bandwidth": "1Gbps",  "protocol": "Ethernet", "zone": "server"}),
    ("RTR-01",   "SRV-WEB",  {"bandwidth": "1Gbps",  "protocol": "Ethernet", "zone": "server"}),
    ("RTR-01",   "SRV-DB",   {"bandwidth": "1Gbps",  "protocol": "Ethernet", "zone": "server"}),
    ("RTR-01",   "SRV-AUTH", {"bandwidth": "1Gbps",  "protocol": "Ethernet", "zone": "server"}),
    ("RTR-01",   "SRV-FILE", {"bandwidth": "1Gbps",  "protocol": "Ethernet", "zone": "server"}),
    # Core → Zone Switches
    ("RTR-01",   "SW-ACAD",  {"bandwidth": "10Gbps", "protocol": "Ethernet", "zone": "academic"}),
    ("RTR-01",   "SW-ADMIN", {"bandwidth": "10Gbps", "protocol": "Ethernet", "zone": "admin"}),
    ("RTR-01",   "SW-HOSTEL",{"bandwidth": "10Gbps", "protocol": "Ethernet", "zone": "hostel"}),
    # Academic Zone
    ("SW-ACAD",  "FAC-PC01", {"bandwidth": "100Mbps","protocol": "Ethernet", "zone": "academic"}),
    ("SW-ACAD",  "FAC-PC02", {"bandwidth": "100Mbps","protocol": "Ethernet", "zone": "academic"}),
    ("SW-ACAD",  "STU-PC01", {"bandwidth": "100Mbps","protocol": "Ethernet", "zone": "academic"}),
    ("SW-ACAD",  "CCTV-01",  {"bandwidth": "100Mbps","protocol": "Ethernet", "zone": "iot"}),
    ("SW-ACAD",  "RFID-01",  {"bandwidth": "100Mbps","protocol": "Ethernet", "zone": "iot"}),
    ("SW-ACAD",  "AP-ACAD",  {"bandwidth": "1Gbps",  "protocol": "Ethernet", "zone": "academic"}),
    # Admin Zone
    ("SW-ADMIN", "ADMIN-PC01",{"bandwidth": "100Mbps","protocol": "Ethernet","zone": "admin"}),
    ("SW-ADMIN", "ADMIN-SRV", {"bandwidth": "1Gbps",  "protocol": "Ethernet","zone": "admin"}),
    ("SW-ADMIN", "ACS-01",    {"bandwidth": "100Mbps","protocol": "Ethernet","zone": "iot"}),
    # Hostel Zone
    ("SW-HOSTEL","STU-DEV01", {"bandwidth": "100Mbps","protocol": "Ethernet","zone": "hostel"}),
    ("SW-HOSTEL","STU-DEV02", {"bandwidth": "100Mbps","protocol": "Ethernet","zone": "hostel"}),
    ("SW-HOSTEL","AP-HOSTEL", {"bandwidth": "1Gbps",  "protocol": "Ethernet","zone": "hostel"}),
    ("SW-HOSTEL","IOT-HOSTEL",{"bandwidth": "100Mbps","protocol": "Ethernet","zone": "iot"}),
    ("SW-HOSTEL","CCTV-02",   {"bandwidth": "100Mbps","protocol": "Ethernet","zone": "iot"}),
]

# IP → device_id mapping for ML integration
IP_TO_DEVICE: dict[str, str] = {
    d["ip_address"]: d["device_id"] for d in DEFAULT_DEVICES
}


# ---------------------------------------------------------------------------
# Topology builder
# ---------------------------------------------------------------------------

class CampusTopology:
    """
    NetworkX-backed campus topology for the Digital Twin.

    Nodes carry :class:`Device` objects; edges carry connection metadata.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._devices: dict[str, Device] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_device(self, device: Device) -> None:
        """Add a device node to the topology."""
        if device.device_id in self._devices:
            raise ValueError(f"Device '{device.device_id}' already exists.")
        self._devices[device.device_id] = device
        self._graph.add_node(device.device_id, device=device)
        logger.debug("Added device: %s (%s)", device.device_id, device.device_type)

    def remove_device(self, device_id: str) -> None:
        """Remove a device and its edges from the topology."""
        if device_id not in self._devices:
            raise KeyError(f"Device '{device_id}' not found.")
        self._graph.remove_node(device_id)
        del self._devices[device_id]

    def connect_devices(
        self,
        source_id: str,
        target_id: str,
        **edge_attrs: Any,
    ) -> None:
        """Create a bidirectional link between two devices."""
        for dev_id in (source_id, target_id):
            if dev_id not in self._devices:
                raise KeyError(f"Device '{dev_id}' not found.")
        edge_attrs.setdefault("status", "ACTIVE")
        self._graph.add_edge(source_id, target_id, **edge_attrs)
        self._graph.add_edge(target_id, source_id, **edge_attrs)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_device(self, device_id: str) -> Device:
        """Return the Device object for the given ID."""
        if device_id not in self._devices:
            raise KeyError(f"Unknown device ID: '{device_id}'")
        return self._devices[device_id]

    def get_device_by_ip(self, ip_address: str) -> Optional[Device]:
        """Return the Device matching an IP address, or None."""
        for dev in self._devices.values():
            if dev.ip_address == ip_address:
                return dev
        return None

    def get_neighbors(self, device_id: str) -> list[Device]:
        """Return neighbour devices of the specified node."""
        if device_id not in self._graph:
            raise KeyError(f"Device '{device_id}' not in graph.")
        return [self._devices[n] for n in self._graph.neighbors(device_id)]

    def all_devices(self) -> list[Device]:
        return list(self._devices.values())

    def device_count(self) -> int:
        return len(self._devices)

    def edge_count(self) -> int:
        # Bidirectional — report undirected count
        return self._graph.number_of_edges() // 2

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_device_status(
        self,
        device_id: str,
        new_status: str,
        reason: str = "",
    ) -> None:
        """Transition a device to a new status."""
        device = self.get_device(device_id)
        device.transition_to(new_status, reason=reason)

    def update_risk_score(self, device_id: str, score: float) -> None:
        """Update the risk score of a device."""
        device = self.get_device(device_id)
        device.update_risk_score(score)

    # ------------------------------------------------------------------
    # Graph access
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        """Direct access to the underlying NetworkX graph."""
        return self._graph

    def summary(self) -> dict:
        statuses: dict[str, int] = {}
        for d in self._devices.values():
            statuses[d.status] = statuses.get(d.status, 0) + 1
        return {
            "total_devices": self.device_count(),
            "total_connections": self.edge_count(),
            "status_counts": statuses,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_campus_topology() -> CampusTopology:
    """
    Build and return the full representative educational campus topology.

    Returns
    -------
    CampusTopology
        Fully connected campus network with all devices and links.
    """
    topo = CampusTopology()

    for dev_data in DEFAULT_DEVICES:
        device = Device(**dev_data)
        topo.add_device(device)

    for src, dst, attrs in DEFAULT_EDGES:
        topo.connect_devices(src, dst, **attrs)

    logger.info(
        "Campus topology created: %d devices, %d connections.",
        topo.device_count(),
        topo.edge_count(),
    )
    return topo
