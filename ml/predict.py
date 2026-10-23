"""
ml/predict.py
=============
Prediction module — maps network flow data to Digital Twin device states.

Provides:
  - predict_flow():     predict a single flow record
  - predict_batch():    predict a batch of flow records and update the Digital Twin
  - demo_predict():     demonstration pipeline with CICIDS2017 test data

Usage
-----
    from ml.predict import FlowPredictor
    python -m ml.predict
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from digital_twin.state import DigitalTwinStateManager
from digital_twin.topology import IP_TO_DEVICE, create_campus_topology

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")

# ---------------------------------------------------------------------------
# Demo Mapping Layer
# ---------------------------------------------------------------------------
# CICIDS2017 dataset does not contain source/destination IP addresses in the
# processed CSV files (they are dropped during preprocessing as metadata).
# This mapping layer provides a controlled, deterministic way to associate
# dataset flows with simulated campus devices for demonstration purposes.
#
# IMPORTANT: This is a DEMO mapping. The public dataset IP addresses are NOT
# real campus addresses. The mapping is explicitly configured for reproducible
# demonstrations and should not be confused with real network traffic analysis.
# ---------------------------------------------------------------------------

# Deterministic mapping: dataset row index modulo device count -> device IP
# This ensures the same flow always maps to the same simulated device.
DEMO_DEVICE_IPS = [
    "192.168.20.10",   # CCTV-01 (Academic Block)
    "192.168.20.11",   # RFID-01 (Academic Block)
    "192.168.20.101",  # FAC-PC01 (Academic Block)
    "192.168.20.102",  # FAC-PC02 (Academic Block)
    "192.168.20.201",  # STU-PC01 (Academic Block)
    "192.168.10.11",   # SRV-WEB (Server Room)
    "192.168.10.10",   # SRV-LMS (Server Room)
    "192.168.10.12",   # SRV-DB (Server Room)
    "192.168.10.13",   # SRV-AUTH (Server Room)
    "192.168.10.14",   # SRV-FILE (Server Room)
    "192.168.30.101",  # ADMIN-PC01 (Admin Block)
    "192.168.30.10",   # ADMIN-SRV (Admin Block)
    "192.168.30.20",   # ACS-01 (Admin Block)
    "192.168.40.101",  # STU-DEV01 (Hostel Block)
    "192.168.40.102",  # STU-DEV02 (Hostel Block)
    "192.168.40.20",   # IOT-HOSTEL (Hostel Block)
    "192.168.40.21",   # CCTV-02 (Hostel Block)
]


def get_demo_device_ip(row_index: int) -> str:
    """
    Deterministically map a dataset row index to a simulated campus device IP.

    Parameters
    ----------
    row_index:
        Index of the flow in the dataset (0-based).

    Returns
    -------
    str
        IP address of a simulated campus device.
    """
    return DEMO_DEVICE_IPS[row_index % len(DEMO_DEVICE_IPS)]


def add_demo_ips(flows: pd.DataFrame) -> pd.DataFrame:
    """
    Add destination_ip column to flows using deterministic demo mapping.

    Parameters
    ----------
    flows:
        DataFrame of flow records (without IP columns).

    Returns
    -------
    pd.DataFrame
        Flows with added 'destination_ip' column.
    """
    result = flows.copy()
    result["destination_ip"] = [get_demo_device_ip(i) for i in range(len(result))]
    return result


class FlowPredictor:
    """
    Wraps the trained Random Forest model and maps predictions
    to Digital Twin device state updates.

    Parameters
    ----------
    model_path:
        Path to the saved ``random_forest.joblib`` model.
    scaler_path:
        Path to the saved ``scaler.joblib``.
    feature_names_path:
        Path to the saved ``feature_names.joblib``.
    """

    def __init__(
        self,
        model_path: str | Path = MODELS_DIR / "random_forest.joblib",
        scaler_path: str | Path = MODELS_DIR / "scaler.joblib",
        feature_names_path: str | Path = MODELS_DIR / "feature_names.joblib",
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                "Model not found. Run the training pipeline first:\n"
                "  python -m ml.train"
            )
        self._model = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)
        self._features: list[str] = joblib.load(feature_names_path)
        logger.info("FlowPredictor loaded from %s", model_path)

    # ------------------------------------------------------------------

    def predict_flow(self, flow: dict) -> dict:
        """
        Predict whether a single network flow is an attack.

        Parameters
        ----------
        flow:
            Dict of feature_name → value.

        Returns
        -------
        dict with keys: prediction, confidence, risk_score
        """
        row = pd.DataFrame([flow])
        # Fill missing features with 0
        for f in self._features:
            if f not in row.columns:
                row[f] = 0.0
        row = row[self._features].astype(np.float32)

        # Replace inf/nan
        row.replace([np.inf, -np.inf], np.nan, inplace=True)
        row.fillna(0.0, inplace=True)

        scaled = self._scaler.transform(row)
        label = int(self._model.predict(scaled)[0])
        proba = self._model.predict_proba(scaled)[0]
        confidence = float(proba[label])
        risk = float(proba[1])  # probability of attack class

        return {
            "prediction": "ATTACK" if label == 1 else "NORMAL",
            "label": label,
            "confidence": round(confidence, 4),
            "risk_score": round(risk, 4),
        }

    def predict_batch(
        self,
        flows: pd.DataFrame,
        state_manager: Optional[DigitalTwinStateManager] = None,
        dst_ip_col: str = "destination_ip",
        pre_scaled: bool = False,
    ) -> pd.DataFrame:
        """
        Predict a batch of flows and optionally update the Digital Twin.

        Parameters
        ----------
        flows:
            DataFrame of flow records.
        state_manager:
            If provided, updates device states in the Digital Twin.
        dst_ip_col:
            Column containing the destination IP (used for device mapping).
        pre_scaled:
            If True, assumes `flows` already contains scaled features matching
            `self._features` and skips the scaling step. Use when passing data
            from `ml.preprocess.load_processed()` which returns pre-scaled data.

        Returns
        -------
        pd.DataFrame
            Input flows augmented with prediction/risk columns.
        """
        result = flows.copy()

        feature_df = result.reindex(columns=self._features, fill_value=0.0)
        feature_df = feature_df.astype(np.float32)
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_df.fillna(0.0, inplace=True)

        if pre_scaled:
            scaled = feature_df.values
        else:
            scaled = self._scaler.transform(feature_df)

        labels = self._model.predict(scaled)
        proba = self._model.predict_proba(scaled)
        risk_scores = proba[:, 1]

        result["prediction"] = np.where(labels == 1, "ATTACK", "NORMAL")
        result["risk_score"] = risk_scores.round(4)
        result["confidence"] = np.max(proba, axis=1).round(4)

        # Update Digital Twin if state_manager provided
        if state_manager and dst_ip_col in result.columns:
            for _, row in result.iterrows():
                ip = str(row.get(dst_ip_col, ""))
                device_id = IP_TO_DEVICE.get(ip)
                if device_id:
                    state_manager.record_anomaly(
                        device_id=device_id,
                        risk_score=float(row["risk_score"]),
                        prediction=str(row["prediction"]),
                        confidence=float(row["confidence"]),
                        source_ip=ip,
                    )

        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstration: run predictions on CICIDS2017 test data and update Digital Twin."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("  ML Prediction + Digital Twin Integration Demo")
    print("=" * 60)

    # Load test data
    from ml.preprocess import load_processed
    try:
        _, X_test, _, y_test = load_processed()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        return

    print(f"Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Create predictor
    predictor = FlowPredictor()

    # Create Digital Twin
    topo = create_campus_topology()
    mgr = DigitalTwinStateManager(topo)

    # Add demo IPs to test data (deterministic mapping)
    test_flows = add_demo_ips(X_test)

    # Run batch prediction with Digital Twin updates
    # X_test from load_processed() is already scaled, so use pre_scaled=True
    print("\nRunning batch prediction with Digital Twin integration...")
    result = predictor.predict_batch(test_flows, state_manager=mgr, dst_ip_col="destination_ip", pre_scaled=True)

    # Show results
    attack_count = (result["prediction"] == "ATTACK").sum()
    normal_count = (result["prediction"] == "NORMAL").sum()
    print(f"\nPredictions: {normal_count} NORMAL, {attack_count} ATTACK")

    # Show a few examples
    print("\n--- Sample Predictions ---")
    for i in range(min(5, len(result))):
        row = result.iloc[i]
        print(f"  Flow {i}: dst_ip={row['destination_ip']}, "
              f"pred={row['prediction']}, risk={row['risk_score']:.4f}, "
              f"conf={row['confidence']:.4f}, true_label={'ATTACK' if y_test.iloc[i]==1 else 'NORMAL'}")

    # Show an attack example if available
    attack_rows = result[result["prediction"] == "ATTACK"]
    if len(attack_rows) > 0:
        row = attack_rows.iloc[0]
        print(f"\n--- Attack Example ---")
        print(f"  Flow index: {attack_rows.index[0]}")
        print(f"  Destination IP: {row['destination_ip']}")
        print(f"  Prediction: {row['prediction']}")
        print(f"  Risk Score: {row['risk_score']:.4f}")
        print(f"  Confidence: {row['confidence']:.4f}")

    # Show Digital Twin state
    print("\n--- Digital Twin State ---")
    compromised = mgr.get_compromised_devices()
    if compromised:
        for dev in compromised:
            print(f"  {dev['device_id']} ({dev['name']}): {dev['status']}, risk={dev['risk_score']:.4f}")
    else:
        print("  No compromised devices")

    # Show recent events
    print("\n--- Recent Events ---")
    events = mgr.get_event_history(limit=5)
    for e in events:
        print(f"  [{e['timestamp']}] {e['device_id']}: {e['previous_status']} -> {e['new_status']} "
              f"(risk={e['risk_score']:.4f}, pred={e.get('prediction', 'N/A')})")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
