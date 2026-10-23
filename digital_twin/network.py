"""
digital_twin/network.py
=======================
Entry-point for the campus Digital Twin.

Running this module builds the topology, prints a summary, and saves the
network visualisation to ``models/topology.png`` (no interactive window needed).

Usage
-----
    python -m digital_twin.network
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from digital_twin.topology import create_campus_topology, CampusTopology
from digital_twin.state import DigitalTwinStateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual encoding
# ---------------------------------------------------------------------------

TYPE_COLORS: dict[str, str] = {
    "External":    "#E74C3C",   # red
    "Security":    "#8E44AD",   # purple
    "Monitoring":  "#D35400",   # orange
    "Router":      "#2C3E50",   # dark
    "Switch":      "#2980B9",   # blue
    "Server":      "#27AE60",   # green
    "IoT":         "#F39C12",   # amber
    "Endpoint":    "#BDC3C7",   # light grey
    "AccessPoint": "#1ABC9C",   # teal
}

TYPE_SHAPES: dict[str, str] = {
    "External":    "s",   # square
    "Security":    "D",   # diamond
    "Monitoring":  "D",
    "Router":      "h",   # hexagon
    "Switch":      "^",   # triangle
    "Server":      "8",   # octagon
    "IoT":         "*",   # star
    "Endpoint":    "o",   # circle
    "AccessPoint": "p",   # pentagon
}

STATUS_BORDER: dict[str, str] = {
    "NORMAL":       "#2ECC71",
    "WARNING":      "#F1C40F",
    "SUSPICIOUS":   "#E67E22",
    "UNDER_ATTACK": "#C0392B",
    "OFFLINE":      "#7F8C8D",
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

FIXED_POSITIONS: dict[str, tuple[float, float]] = {
    # External
    "EXT-01":     (0, 5),
    # Security / Monitoring
    "FW-01":      (2, 5),
    "IDS-01":     (2, 6.5),
    # Core
    "RTR-01":     (4, 5),
    # Servers
    "SRV-LMS":    (6, 8),
    "SRV-WEB":    (6, 6.5),
    "SRV-DB":     (6, 5),
    "SRV-AUTH":   (6, 3.5),
    "SRV-FILE":   (6, 2),
    # Academic zone
    "SW-ACAD":    (8, 8),
    "FAC-PC01":   (10, 9),
    "FAC-PC02":   (10, 8),
    "STU-PC01":   (10, 7),
    "CCTV-01":    (10, 6),
    "RFID-01":    (10, 5),
    "AP-ACAD":    (10, 4),
    # Admin zone
    "SW-ADMIN":   (8, 3),
    "ADMIN-PC01": (10, 3),
    "ADMIN-SRV":  (10, 2),
    "ACS-01":     (10, 1),
    # Hostel zone
    "SW-HOSTEL":  (8, -1),
    "STU-DEV01":  (10, -0.5),
    "STU-DEV02":  (10, -1.5),
    "AP-HOSTEL":  (10, -2.5),
    "IOT-HOSTEL": (10, -3.5),
    "CCTV-02":    (10, -4.5),
}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_topology(
    topo: CampusTopology,
    output_path: str | Path = "models/topology.png",
    show: bool = False,
) -> str:
    """
    Render the campus topology to a PNG file.

    Parameters
    ----------
    topo:
        The campus topology to render.
    output_path:
        Destination path for the saved image.
    show:
        If True, call plt.show() (only useful with an interactive display).

    Returns
    -------
    str
        Absolute path to the saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    G = topo.graph
    devices = {dev.device_id: dev for dev in topo.all_devices()}

    # Use fixed positions; fall back to spring layout for unknown nodes
    pos = {}
    for node in G.nodes():
        pos[node] = FIXED_POSITIONS.get(node, (0, 0))

    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_facecolor("#0D1117")
    fig.patch.set_facecolor("#0D1117")

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color="#334155",
        width=1.2,
        alpha=0.7,
        arrows=False,
        ax=ax,
    )

    # Draw nodes per device type (to use different shapes)
    for dev_type, shape in TYPE_SHAPES.items():
        node_ids = [
            n for n in G.nodes()
            if devices.get(n) and devices[n].device_type == dev_type
        ]
        if not node_ids:
            continue
        node_colors = [TYPE_COLORS.get(dev_type, "#BDC3C7")] * len(node_ids)
        border_colors = [
            STATUS_BORDER.get(devices[n].status, "#2ECC71") for n in node_ids
        ]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=node_ids,
            node_color=node_colors,
            edgecolors=border_colors,
            node_shape=shape,
            node_size=800,
            linewidths=2.5,
            ax=ax,
        )

    # Labels — short IDs to keep the graph readable
    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=6,
        font_color="#F8FAFC",
        font_weight="bold",
        ax=ax,
    )

    # Legend — device types
    legend_patches = [
        mpatches.Patch(color=color, label=dev_type)
        for dev_type, color in TYPE_COLORS.items()
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower left",
        fontsize=8,
        framealpha=0.3,
        labelcolor="white",
        facecolor="#1E293B",
        edgecolor="#334155",
        title="Device Type",
        title_fontsize=9,
    )

    ax.set_title(
        "AI-Driven Digital Twin — Educational Campus Network Topology",
        fontsize=14,
        fontweight="bold",
        color="#F8FAFC",
        pad=15,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    if show:
        os.system(f"xdg-open {output_path}")  # open in default viewer

    logger.info("Topology saved to: %s", output_path.resolve())
    return str(output_path.resolve())


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  AI-Driven Digital Twin — Educational Campus Network")
    print("=" * 60)

    # Build topology
    topo = create_campus_topology()
    summary = topo.summary()

    print(f"\nDevices      : {summary['total_devices']}")
    print(f"Connections  : {summary['total_connections']}")
    print(f"\nDevice Status Counts:")
    for status, count in summary["status_counts"].items():
        print(f"  {status:<15} {count}")

    # List all devices
    print("\nDevice Inventory:")
    print(f"  {'ID':<12} {'Name':<30} {'Type':<12} {'IP':<16} {'Zone'}")
    print("  " + "-" * 80)
    for dev in sorted(topo.all_devices(), key=lambda d: d.device_id):
        print(
            f"  {dev.device_id:<12} {dev.name:<30} {dev.device_type:<12} "
            f"{dev.ip_address:<16} {dev.location}"
        )

    # Demonstrate state manager
    print("\nDemonstrating State Manager...")
    state_mgr = DigitalTwinStateManager(topo)

    # Simulate a suspicious event on CCTV-01
    state_mgr.update_risk_score("CCTV-01", 0.75, reason="Unusual traffic pattern detected")
    state_mgr.update_risk_score("SRV-WEB", 0.85, reason="Potential DDoS signature")

    print(f"\nEvent History (last 5):")
    for ev in state_mgr.get_event_history(limit=5):
        print(
            f"  [{ev['timestamp']}] {ev['device_id']:<12} "
            f"{ev.get('previous_status','?')!s:<13} → {ev.get('new_status', ev.get('status','?')):<13}"
        )

    # Save topology image
    saved = visualize_topology(topo)
    print(f"\nTopology image saved: {saved}")
    print("\nDigital Twin initialised successfully.")


if __name__ == "__main__":
    main()