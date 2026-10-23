"""
digital_twin/__init__.py
Exposes the primary Digital Twin interface.
"""

from digital_twin.topology import create_campus_topology, CampusTopology
from digital_twin.devices import Device
from digital_twin.state import DigitalTwinStateManager

__all__ = [
    "create_campus_topology",
    "CampusTopology",
    "Device",
    "DigitalTwinStateManager",
]
