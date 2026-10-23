"""
digital_twin/demo.py
====================
Controlled cyberattack demonstration mode for the AI-Driven Digital Twin Framework.

This module provides safe, reproducible attack simulations that operate entirely
against the simulated Digital Twin and synthetic telemetry. No real network
traffic is generated, no sockets are opened, and no real systems are scanned.

Usage
-----
    python -m digital_twin.demo --attack port_scan --device CCTV-01
    python -m digital_twin.demo --attack ddos --device SRV-WEB
    python -m digital_twin.demo --attack brute_force --device SRV-AUTH
    python -m digital_twin.demo --attack unauthorized_access --device ADMIN-SRV
    python -m digital_twin.demo --reset
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from digital_twin.state import DigitalTwinStateManager
from digital_twin.topology import create_campus_topology, IP_TO_DEVICE
from ml.predict import FlowPredictor
from ml.preprocess import load_processed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attack Scenario Definitions
# ---------------------------------------------------------------------------

ATTACK_SCENARIOS = {
    "port_scan": {
        "name": "Port Scan",
        "description": "Simulated reconnaissance scanning multiple ports on target device",
        "risk_base": 0.65,
        "confidence_base": 0.75,
        "traffic_pattern": "Multiple SYN packets to different ports from single source",
    },
    "ddos": {
        "name": "DDoS-like Traffic Spike",
        "description": "Simulated volumetric attack flooding target with high-volume traffic",
        "risk_base": 0.85,
        "confidence_base": 0.90,
        "traffic_pattern": "High packet rate, small packets, many sources to single destination",
    },
    "brute_force": {
        "name": "Brute Force Authentication",
        "description": "Simulated repeated login attempts against authentication service",
        "risk_base": 0.75,
        "confidence_base": 0.80,
        "traffic_pattern": "Repeated failed auth attempts, short intervals, single source",
    },
    "unauthorized_access": {
        "name": "Unauthorized Access Attempt",
        "description": "Simulated access to restricted resources or privilege escalation",
        "risk_base": 0.80,
        "confidence_base": 0.85,
        "traffic_pattern": "Anomalous protocol usage, unusual command sequences, privilege escalation",
    },
}


def get_attack_scenario(attack_type: str) -> dict:
    """Get attack scenario configuration."""
    if attack_type not in ATTACK_SCENARIOS:
        raise ValueError(
            f"Unknown attack type: {attack_type}. "
            f"Available: {list(ATTACK_SCENARIOS.keys())}"
        )
    return ATTACK_SCENARIOS[attack_type]


# ---------------------------------------------------------------------------
# Synthetic Telemetry Generation
# ---------------------------------------------------------------------------

def generate_synthetic_telemetry(
    attack_type: str,
    target_device_id: str,
    topo,
    num_flows: int = 10,
) -> pd.DataFrame:
    """
    Generate synthetic flow telemetry for a specific attack scenario.

    This creates feature vectors that resemble the attack pattern without
    using real network packets. The features are derived from the test
    distribution but modified to match the attack characteristics.

    Parameters
    ----------
    attack_type:
        Type of attack to simulate.
    target_device_id:
        Device ID to target.
    topo:
        Campus topology for IP lookup.
    num_flows:
        Number of synthetic flows to generate.

    Returns
    -------
    pd.DataFrame
        Synthetic flows with features matching the model's expected input.
    """
    scenario = get_attack_scenario(attack_type)
    target_ip = topo.get_device(target_device_id).ip_address

    # Load test data to get feature distribution
    _, X_test, _, _ = load_processed()

    # Sample base flows from test data
    base_flows = X_test.sample(n=min(num_flows, len(X_test)), random_state=42).copy()

    # Modify features to match attack pattern
    # These modifications are heuristic and designed to push the model
    # toward ATTACK classification for demonstration
    if attack_type == "port_scan":
        # Port scan: many small packets, short duration, many destinations
        base_flows["Total Fwd Packets"] = np.random.randint(5, 20, num_flows)
        base_flows["Flow Duration"] = np.random.randint(1, 100, num_flows)
        base_flows["Flow Packets/s"] = np.random.uniform(100, 1000, num_flows)

    elif attack_type == "ddos":
        # DDoS: high volume, high packet rate
        base_flows["Total Fwd Packets"] = np.random.randint(1000, 10000, num_flows)
        base_flows["Flow Bytes/s"] = np.random.uniform(1e6, 1e8, num_flows)
        base_flows["Flow Packets/s"] = np.random.uniform(10000, 100000, num_flows)

    elif attack_type == "brute_force":
        # Brute force: repeated connections, moderate volume
        base_flows["Total Fwd Packets"] = np.random.randint(10, 50, num_flows)
        base_flows["Flow Duration"] = np.random.randint(100, 1000, num_flows)
        base_flows["SYN Flag Count"] = np.random.randint(5, 20, num_flows)

    elif attack_type == "unauthorized_access":
        # Unauthorized access: anomalous protocol behavior
        base_flows["Total Fwd Packets"] = np.random.randint(20, 100, num_flows)
        base_flows["Flow Duration"] = np.random.randint(500, 5000, num_flows)
        base_flows["PSH Flag Count"] = np.random.randint(10, 50, num_flows)

    # Add destination IP for device mapping
    base_flows["destination_ip"] = target_ip

    return base_flows


# ---------------------------------------------------------------------------
# Demo Execution
# ---------------------------------------------------------------------------

def run_attack_demo(
    attack_type: str,
    target_device_id: str,
    topo,
    mgr: DigitalTwinStateManager,
    num_flows: int = 10,
) -> dict:
    """
    Run a complete attack demonstration.

    Pipeline:
    1. Generate synthetic telemetry for attack type
    2. Pass through ML model (FlowPredictor)
    3. Map predictions to Digital Twin device
    4. Update device state via state manager
    5. Return summary of results

    Parameters
    ----------
    attack_type:
        Type of attack to simulate.
    target_device_id:
        Device ID to target.
    topo:
        Campus topology.
    mgr:
        Digital Twin state manager.
    num_flows:
        Number of synthetic flows to generate.

    Returns
    -------
    dict
        Summary of demo execution.
    """
    scenario = get_attack_scenario(attack_type)
    target_ip = topo.get_device(target_device_id).ip_address

    logger.info(
        "Starting %s demo on %s (%s)",
        scenario["name"],
        target_device_id,
        target_ip,
    )

    # Generate synthetic telemetry
    synthetic_flows = generate_synthetic_telemetry(
        attack_type, target_device_id, topo, num_flows
    )

    # Run through ML predictor
    predictor = FlowPredictor()
    result = predictor.predict_batch(
        synthetic_flows,
        state_manager=mgr,
        dst_ip_col="destination_ip",
        pre_scaled=True,
    )

    # Count predictions
    attack_count = (result["prediction"] == "ATTACK").sum()
    normal_count = (result["prediction"] == "NORMAL").sum()
    avg_risk = result["risk_score"].mean()
    max_risk = result["risk_score"].max()

    # Get final device state
    device = topo.get_device(target_device_id)

    summary = {
        "attack_type": attack_type,
        "scenario_name": scenario["name"],
        "target_device": target_device_id,
        "target_ip": target_ip,
        "flows_generated": num_flows,
        "attack_predictions": int(attack_count),
        "normal_predictions": int(normal_count),
        "avg_risk_score": round(float(avg_risk), 4),
        "max_risk_score": round(float(max_risk), 4),
        "final_device_status": device.status,
        "final_risk_score": device.risk_score,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    logger.info(
        "Demo complete: %d/%d flows classified as ATTACK, "
        "device %s -> %s (risk=%.4f)",
        attack_count, num_flows, target_device_id, device.status, device.risk_score,
    )

    return summary


def reset_demo_state(mgr: DigitalTwinStateManager, topo) -> None:
    """Reset all devices to NORMAL state and clear event history."""
    for dev in topo.all_devices():
        if dev.status != "NORMAL":
            mgr.reset_device(dev.device_id, reason="Demo reset")
    # Clear event history by creating new manager (or we could add a clear method)
    # For now, just reset device states
    logger.info("Demo state reset: all devices returned to NORMAL")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled cyberattack demonstration for Digital Twin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m digital_twin.demo --attack port_scan --device CCTV-01
  python -m digital_twin.demo --attack ddos --device SRV-WEB
  python -m digital_twin.demo --attack brute_force --device SRV-AUTH
  python -m digital_twin.demo --attack unauthorized_access --device ADMIN-SRV
  python -m digital_twin.demo --reset
        """,
    )
    parser.add_argument(
        "--attack",
        choices=list(ATTACK_SCENARIOS.keys()),
        help="Type of attack to simulate",
    )
    parser.add_argument(
        "--device",
        help="Target device ID (e.g., CCTV-01, SRV-WEB)",
    )
    parser.add_argument(
        "--flows",
        type=int,
        default=10,
        help="Number of synthetic flows to generate (default: 10)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all devices to NORMAL state",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available devices and exit",
    )
    parser.add_argument(
        "--list-attacks",
        action="store_true",
        help="List available attack scenarios and exit",
    )

    args = parser.parse_args()

    # Handle list commands
    if args.list_devices:
        topo = create_campus_topology()
        print("Available devices:")
        for dev in sorted(topo.all_devices(), key=lambda d: d.device_id):
            print(f"  {dev.device_id:12} {dev.name:30} {dev.ip_address:15} {dev.location}")
        return

    if args.list_attacks:
        print("Available attack scenarios:")
        for key, scenario in ATTACK_SCENARIOS.items():
            print(f"  {key:20} {scenario['name']}")
            print(f"  {'':20} {scenario['description']}")
        return

    if args.reset:
        topo = create_campus_topology()
        mgr = DigitalTwinStateManager(topo)
        reset_demo_state(mgr, topo)
        print("Digital Twin reset to NORMAL state.")
        return

    if not args.attack or not args.device:
        parser.error("--attack and --device are required (unless using --reset, --list-devices, or --list-attacks)")

    # Validate device exists
    topo = create_campus_topology()
    try:
        topo.get_device(args.device)
    except KeyError:
        print(f"Error: Unknown device '{args.device}'")
        print("Use --list-devices to see available devices.")
        sys.exit(1)

    # Run demo
    mgr = DigitalTwinStateManager(topo)
    summary = run_attack_demo(
        attack_type=args.attack,
        target_device_id=args.device,
        topo=topo,
        mgr=mgr,
        num_flows=args.flows,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"  DEMO COMPLETE: {summary['scenario_name']}")
    print("=" * 60)
    print(f"  Target Device     : {summary['target_device']} ({summary['target_ip']})")
    print(f"  Flows Generated   : {summary['flows_generated']}")
    print(f"  ATTACK Predictions: {summary['attack_predictions']}")
    print(f"  NORMAL Predictions: {summary['normal_predictions']}")
    print(f"  Avg Risk Score    : {summary['avg_risk_score']:.4f}")
    print(f"  Max Risk Score    : {summary['max_risk_score']:.4f}")
    print(f"  Final Status      : {summary['final_device_status']}")
    print(f"  Final Risk        : {summary['final_risk_score']:.4f}")
    print(f"  Timestamp         : {summary['timestamp']}")
    print("=" * 60)


if __name__ == "__main__":
    main()