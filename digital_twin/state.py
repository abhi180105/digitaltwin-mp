"""
digital_twin/state.py
=====================
Digital Twin state manager for the AI-Driven Digital Twin Framework.

Manages device state transitions, risk score updates, and a lightweight
event history that feeds the monitoring dashboard.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from digital_twin.topology import CampusTopology

logger = logging.getLogger(__name__)

# Risk-score → status thresholds
RISK_THRESHOLDS = {
    "NORMAL":       (0.00, 0.30),
    "WARNING":      (0.30, 0.60),
    "SUSPICIOUS":   (0.60, 0.80),
    "UNDER_ATTACK": (0.80, 1.01),
}


def risk_to_status(risk: float) -> str:
    """Map a risk score to the corresponding device status."""
    for status, (low, high) in RISK_THRESHOLDS.items():
        if low <= risk < high:
            return status
    return "UNDER_ATTACK"


class DigitalTwinStateManager:
    """
    Manages the live state of the campus Digital Twin.

    Parameters
    ----------
    topology:
        An initialised :class:`CampusTopology`.
    """

    def __init__(self, topology: CampusTopology) -> None:
        self._topology = topology
        self._event_history: list[dict] = []

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_device_state(
        self,
        device_id: str,
        new_status: str,
        reason: str = "",
    ) -> dict:
        """
        Transition a device to *new_status* and record the event.

        Returns
        -------
        dict
            The recorded event.
        """
        device = self._topology.get_device(device_id)
        old_status = device.status
        self._topology.update_device_status(device_id, new_status, reason=reason)

        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device_id": device_id,
            "device_name": device.name,
            "location": device.location,
            "previous_status": old_status,
            "new_status": new_status,
            "risk_score": device.risk_score,
            "reason": reason,
        }
        self._event_history.append(event)
        logger.warning(
            "STATE CHANGE | %s | %s → %s | risk=%.2f | %s",
            device_id, old_status, new_status, device.risk_score, reason,
        )
        return event

    def update_risk_score(
        self,
        device_id: str,
        score: float,
        reason: str = "",
        auto_transition: bool = True,
    ) -> Optional[dict]:
        """
        Update the risk score of a device and optionally auto-transition its status.

        ML predictions can jump from any status to any other status based on the
        risk score, so we use direct assignment instead of the validated transition
        table (which enforces sequential state changes for manual operations).

        Parameters
        ----------
        device_id:
            Target device.
        score:
            New risk score in [0, 1].
        reason:
            Human-readable reason string.
        auto_transition:
            If True, derive the correct status from *score* and force-set it.

        Returns
        -------
        dict or None
            Event dict if a state transition occurred, else None.
        """
        self._topology.update_risk_score(device_id, score)
        device = self._topology.get_device(device_id)

        if auto_transition:
            target_status = risk_to_status(score)
            if target_status != device.status:
                # Force-set status (bypasses sequential transition rules for ML updates)
                old_status = device.status
                device.status = target_status
                from datetime import datetime
                device.last_seen = datetime.now()

                event = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "device_id": device_id,
                    "device_name": device.name,
                    "location": device.location,
                    "previous_status": old_status,
                    "new_status": target_status,
                    "risk_score": round(score, 4),
                    "reason": reason or "ML risk score update",
                }
                self._event_history.append(event)
                logger.warning(
                    "RISK UPDATE | %s | %s → %s | risk=%.2f | %s",
                    device_id, old_status, target_status, score, reason,
                )
                return event
        return None

    def record_anomaly(
        self,
        device_id: str,
        risk_score: float,
        prediction: str,
        confidence: float,
        source_ip: Optional[str] = None,
    ) -> dict:
        """
        Record an ML-derived anomaly event on a device.

        Parameters
        ----------
        device_id:
            Digital Twin device to update.
        risk_score:
            Normalised risk score from the ML model [0, 1].
        prediction:
            Model prediction label (e.g. "ATTACK", "NORMAL").
        confidence:
            Model confidence/probability [0, 1].
        source_ip:
            Source IP from the network flow (optional).
        """
        self._topology.update_risk_score(device_id, risk_score)
        device = self._topology.get_device(device_id)
        target_status = risk_to_status(risk_score)

        event: dict = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device_id": device_id,
            "device_name": device.name,
            "location": device.location,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "risk_score": round(risk_score, 4),
            "previous_status": device.status,
            "new_status": target_status,
            "source_ip": source_ip,
        }

        if target_status != device.status:
            # Force-set status (ML can jump any status level)
            device.status = target_status
            device.last_seen = datetime.now()

        self._event_history.append(event)
        logger.info(
            "ANOMALY | %s | prediction=%s | confidence=%.2f | risk=%.2f",
            device_id, prediction, confidence, risk_score,
        )
        return event

    def reset_device(self, device_id: str, reason: str = "Manual reset") -> dict:
        """Reset a device to NORMAL status with zero risk."""
        device = self._topology.get_device(device_id)
        old_status = device.status
        device.reset()
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device_id": device_id,
            "device_name": device.name,
            "location": device.location,
            "previous_status": old_status,
            "new_status": "NORMAL",
            "risk_score": 0.0,
            "reason": reason,
        }
        self._event_history.append(event)
        return event

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_device_state(self, device_id: str) -> dict:
        """Return the current state dict of a device."""
        return self._topology.get_device(device_id).to_dict()

    def get_event_history(self, limit: Optional[int] = None) -> list[dict]:
        """
        Return the event history, most-recent first.

        Parameters
        ----------
        limit:
            Maximum number of events to return (None = all).
        """
        history = list(reversed(self._event_history))
        return history[:limit] if limit else history

    def get_topology_summary(self) -> dict:
        return self._topology.summary()

    def get_compromised_devices(self) -> list[dict]:
        """Return state dicts for devices that are SUSPICIOUS or UNDER_ATTACK."""
        return [
            d.to_dict()
            for d in self._topology.all_devices()
            if d.is_compromised
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def topology(self) -> CampusTopology:
        return self._topology
