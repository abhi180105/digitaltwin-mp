"""
digital_twin/interactive_topology.py
====================================
Interactive Plotly-based network topology visualization for the Digital Twin.

Features:
- Zoom, pan, hover tooltips
- Click to select device and see details
- Real-time status/risk score updates
- Color-coded by device type and status
- Responsive layout for Streamlit embedding

Usage:
    from digital_twin.interactive_topology import create_interactive_topology
    fig = create_interactive_topology(topo, state_manager)
    st.plotly_chart(fig, use_container_width=True)
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import plotly.graph_objects as go

from digital_twin.topology import CampusTopology
from digital_twin.state import DigitalTwinStateManager

logger = logging.getLogger(__name__)

# Visual encoding
type_colors = {
    "External":    "#E74C3C",
    "Security":    "#8E44AD",
    "Monitoring":  "#D35400",
    "Router":      "#2C3E50",
    "Switch":      "#2980B9",
    "Server":      "#27AE60",
    "IoT":         "#F39C12",
    "Endpoint":    "#BDC3C7",
    "AccessPoint": "#1ABC9C",
}

type_symbols = {
    "External":    "square",
    "Security":    "diamond",
    "Monitoring":  "diamond",
    "Router":      "hexagon",
    "Switch":      "triangle-up",
    "Server":      "octagon",
    "IoT":         "star",
    "Endpoint":    "circle",
    "AccessPoint": "pentagon",
}

status_border = {
    "NORMAL":       "#2ECC71",
    "WARNING":      "#F1C40F",
    "SUSPICIOUS":   "#E67E22",
    "UNDER_ATTACK": "#C0392B",
    "OFFLINE":      "#7F8C8D",
}

fixed_positions = {
    "EXT-01":     (0, 5),
    "FW-01":      (2, 5),
    "IDS-01":     (2, 6.5),
    "RTR-01":     (4, 5),
    "SRV-LMS":    (6, 8),
    "SRV-WEB":    (6, 6.5),
    "SRV-DB":     (6, 5),
    "SRV-AUTH":   (6, 3.5),
    "SRV-FILE":   (6, 2),
    "SW-ACAD":    (8, 8),
    "FAC-PC01":   (10, 9),
    "FAC-PC02":   (10, 8),
    "STU-PC01":   (10, 7),
    "CCTV-01":    (10, 6),
    "RFID-01":    (10, 5),
    "AP-ACAD":    (10, 4),
    "SW-ADMIN":   (8, 3),
    "ADMIN-PC01": (10, 3),
    "ADMIN-SRV":  (10, 2),
    "ACS-01":     (10, 1),
    "SW-HOSTEL":  (8, -1),
    "STU-DEV01":  (10, -0.5),
    "STU-DEV02":  (10, -1.5),
    "AP-HOSTEL":  (10, -2.5),
    "IOT-HOSTEL": (10, -3.5),
    "CCTV-02":    (10, -4.5),
}
def create_interactive_topology(topo, state_manager=None, height=700):
    """Create an interactive Plotly network topology figure."""
    G = topo.graph
    devices = topo._devices

    pos = {}
    for node in G.nodes():
        pos[node] = fixed_positions.get(node, (0, 0))

    edge_x = []
    edge_y = []
    edge_info = []

    for src, dst, attrs in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        bw = attrs.get("bandwidth_mbps", 1000)
        lat = attrs.get("latency_ms", 1)
        edge_info.append(f"{src} <-> {dst}<br>Bandwidth: {bw} Mbps<br>Latency: {lat} ms")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.5, color="#334155"),
        hoverinfo="text",
        text=[edge_info[i // 3] for i in range(len(edge_x)) if i % 3 != 2],
        showlegend=False, opacity=0.6,
    )

    node_traces = []
    for dev_type in type_colors:
        type_nodes = [n for n in G.nodes() if devices[n].device_type == dev_type]
        if not type_nodes:
            continue

        x_vals, y_vals, node_text, node_custom = [], [], [], []
        border_colors, marker_sizes, marker_symbols = [], [], []

        for node in type_nodes:
            device = devices[node]
            x, y = pos[node]
            x_vals.append(x)
            y_vals.append(y)

            if state_manager:
                status = state_manager.get_device_state(node)["status"]
                risk = state_manager.get_device_state(node)["risk_score"]
            else:
                status = device.status
                risk = device.risk_score

            border_colors.append(status_border.get(status, "#7F8C8D"))
            marker_sizes.append(25 + risk * 30)
            marker_symbols.append(type_symbols.get(dev_type, "circle"))

            hover = (f"<b>{node}</b>: {device.name}<br>"
                     f"Type: {device.device_type}<br>"
                     f"IP: {device.ip_address}<br>"
                     f"VLAN: {device.vlan}<br>"
                     f"Location: {device.location}<br>"
                     f"Status: <b>{status}</b><br>"
                     f"Risk Score: {risk:.2f}")
            node_text.append(hover)
            node_custom.append(node)

        node_trace = go.Scatter(
            x=x_vals, y=y_vals, mode="markers+text",
            text=[n for n in type_nodes], textposition="bottom center",
            textfont=dict(size=8, color="#F8FAFC", family="Inter"),
            marker=dict(
                size=marker_sizes, color=type_colors[dev_type],
                symbol=marker_symbols,
                line=dict(width=2.5, color=border_colors), opacity=0.95,
            ),
            hoverinfo="text", hovertext=node_text,
            customdata=node_custom, name=dev_type, showlegend=True,
            legendgroup=dev_type,
        )
        node_traces.append(node_trace)

    fig = go.Figure()
    fig.add_trace(edge_trace)
    for trace in node_traces:
        fig.add_trace(trace)

    fig.update_layout(
        title=dict(text="Campus Network - Interactive Digital Twin", x=0.5,
                   font=dict(size=16, color="#F8FAFC", family="Inter")),
        showlegend=True,
        legend=dict(title="Device Type", font=dict(size=10, color="#F8FAFC"),
                    bgcolor="rgba(30, 41, 59, 0.8)", bordercolor="#334155",
                    borderwidth=1, yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="closest", plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
        font=dict(color="#F8FAFC", family="Inter"),
        margin=dict(l=20, r=20, t=60, b=20), height=height,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 11.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-5.5, 10], scaleanchor="x", scaleratio=1),
        dragmode="pan", clickmode="event+select",
    )

    status_items = [
        ("NORMAL", "#2ECC71"), ("WARNING", "#F1C40F"),
        ("SUSPICIOUS", "#E67E22"), ("UNDER_ATTACK", "#C0392B"),
        ("OFFLINE", "#7F8C8D"),
    ]
    legend_text = "<b>Status:</b><br>"
    for status, color in status_items:
        legend_text += f'<span style="color:{color}">\u25CF</span> {status}<br>'

    fig.add_annotation(x=0.99, y=0.01, xref="paper", yref="paper",
                       text=legend_text, showarrow=False, align="left",
                       bgcolor="rgba(30, 41, 59, 0.9)", bordercolor="#334155",
                       borderwidth=1, font=dict(size=9, color="#F8FAFC"),
                       xanchor="right", yanchor="bottom")

    return fig
def create_mini_topology(topo, state_manager=None, height=300):
    """Create a compact version for sidebar."""
    fig = create_interactive_topology(topo, state_manager, height=height)
    fig.update_layout(title=None, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig

def get_node_info(topo, node_id, state_manager=None):
    """Get detailed information for a specific node."""
    device = topo.get_device(node_id)
    if state_manager:
        status = state_manager.get_device_state(node_id)["status"]
        risk = state_manager.get_device_state(node_id)["risk_score"]
        events = state_manager.get_event_history(limit=5)
    else:
        status = device.status
        risk = device.risk_score
        events = []
    neighbors = [d.device_id for d in topo.get_neighbors(node_id)]
    return {
        "device_id": device.device_id, "name": device.name,
        "type": device.device_type, "ip_address": device.ip_address,
        "mac_address": device.mac_address, "location": device.location,
        "department": device.department, "vlan": device.vlan,
        "status": status, "risk_score": risk, "neighbors": neighbors,
        "recent_events": [
            {"timestamp": e.get("timestamp"), "previous_status": e.get("previous_status"),
             "new_status": e.get("new_status"), "reason": e.get("reason")}
            for e in events
        ],
    }