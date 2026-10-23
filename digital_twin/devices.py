"""
digital_twin/devices.py
=======================
Device data model for the AI-Driven Digital Twin Framework.

Each campus device is represented as a Python dataclass with
structured attributes, explicit status states, and validation helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATUSES = frozenset({"NORMAL", "WARNING", "SUSPICIOUS", "UNDER_ATTACK", "OFFLINE"})

VALID_TYPES = frozenset({
    "External",
    "Security",
    "Router",
    "Switch",
    "Server",
    "IoT",
    "Endpoint",
    "AccessPoint",
    "Monitoring",
})

# Allowed state transitions (source → set of reachable states)
STATE_TRANSITIONS: dict[str, frozenset[str]] = {
    "NORMAL":       frozenset({"WARNING", "SUSPICIOUS", "OFFLINE"}),
    "WARNING":      frozenset({"NORMAL", "SUSPICIOUS", "UNDER_ATTACK", "OFFLINE"}),
    "SUSPICIOUS":   frozenset({"NORMAL", "WARNING", "UNDER_ATTACK", "OFFLINE"}),
    "UNDER_ATTACK": frozenset({"SUSPICIOUS", "WARNING", "NORMAL", "OFFLINE"}),
    "OFFLINE":      frozenset({"NORMAL"}),
}


# ---------------------------------------------------------------------------
# Device dataclass
# ---------------------------------------------------------------------------

@dataclass
class Device:
    """Represents a single network device in the campus Digital Twin."""

    device_id: str
    name: str
    device_type: str
    ip_address: str
    mac_address: str
    location: str
    department: str
    vlan: str
    status: str = "NORMAL"
    risk_score: float = 0.0
    last_seen: Optional[datetime] = None

    # Internal — not passed by callers
    _event_log: list[dict] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.device_id:
            raise ValueError("device_id must not be empty.")
        if self.device_type not in VALID_TYPES:
            raise ValueError(
                f"Invalid device_type '{self.device_type}'. "
                f"Must be one of: {sorted(VALID_TYPES)}"
            )
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. "
                f"Must be one of: {sorted(VALID_STATUSES)}"
            )
        if not (0.0 <= self.risk_score <= 1.0):
            raise ValueError(
                f"risk_score must be in [0.0, 1.0], got {self.risk_score}"
            )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def transition_to(self, new_status: str, reason: str = "") -> None:
        """
        Transition the device to a new status.

        Parameters
        ----------
        new_status:
            Target status string.
        reason:
            Human-readable reason for the transition.

        Raises
        ------
        ValueError:
            If the transition is not permitted.
        """
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Unknown status: '{new_status}'")

        allowed = STATE_TRANSITIONS.get(self.status, frozenset())
        if new_status not in allowed:
            raise ValueError(
                f"Transition '{self.status}' → '{new_status}' is not allowed "
                f"for device '{self.device_id}'."
            )

        old_status = self.status
        self.status = new_status
        self.last_seen = datetime.now()

        event = {
            "timestamp": self.last_seen.isoformat(),
            "device_id": self.device_id,
            "previous_status": old_status,
            "new_status": new_status,
            "risk_score": self.risk_score,
            "reason": reason,
        }
        self._event_log.append(event)

        logger.info(
            "Device %s: %s → %s | risk=%.2f | reason=%s",
            self.device_id, old_status, new_status, self.risk_score, reason
        )

    def update_risk_score(self, score: float) -> None:
        """Update risk score clamped to [0.0, 1.0]."""
        if not isinstance(score, (int, float)):
            raise TypeError(f"risk_score must be numeric, got {type(score)}")
        self.risk_score = float(max(0.0, min(1.0, score)))
        self.last_seen = datetime.now()

    def reset(self) -> None:
        """Reset device to NORMAL with zero risk."""
        self.status = "NORMAL"
        self.risk_score = 0.0
        self.last_seen = datetime.now()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def is_compromised(self) -> bool:
        return self.status in {"SUSPICIOUS", "UNDER_ATTACK"}

    @property
    def event_history(self) -> list[dict]:
        return list(self._event_log)

    def to_dict(self) -> dict:
        """Return a serialisable representation of the device."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "device_type": self.device_type,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "location": self.location,
            "department": self.department,
            "vlan": self.vlan,
            "status": self.status,
            "risk_score": round(self.risk_score, 4),
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "Device":
        """Create a Device from a dictionary (e.g. loaded from JSON)."""
        last_seen = data.get("last_seen")
        if isinstance(last_seen, str):
            last_seen = datetime.fromisoformat(last_seen)
        return cls(
            device_id=data["device_id"],
            name=data["name"],
            device_type=data["device_type"],
            ip_address=data["ip_address"],
            mac_address=data["mac_address"],
            location=data["location"],
            department=data["department"],
            vlan=data["vlan"],
            status=data.get("status", "NORMAL"),
            risk_score=data.get("risk_score", 0.0),
            last_seen=last_seen,
        )
