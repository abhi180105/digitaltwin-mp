"""
dashboard/app.py
================
Streamlit monitoring dashboard for the AI-Driven Digital Twin Framework.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Allow imports from project root ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import streamlit as st

# Increase max elements for Pandas Styler to handle large event tables
pd.set_option("styler.render.max_elements", 500000)

from digital_twin.network import FIXED_POSITIONS, TYPE_COLORS, STATUS_BORDER, TYPE_SHAPES
from digital_twin.topology import create_campus_topology
from digital_twin.state import DigitalTwinStateManager
from digital_twin.interactive_topology import create_interactive_topology, get_node_info

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Digital Twin — Campus Cyber Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .main { background-color: #0D1117; }
        .block-container { padding: 1.5rem 2rem; }

        .metric-card {
            background: linear-gradient(135deg, #1E293B, #0F172A);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }
        .metric-card h3 { margin: 0; font-size: 2.4rem; font-weight: 700; }
        .metric-card p  { margin: 0; color: #94A3B8; font-size: 0.85rem; }

        .normal-color     { color: #22C55E; }
        .warning-color    { color: #EAB308; }
        .suspicious-color { color: #F97316; }
        .attack-color     { color: #EF4444; }
        .neutral-color    { color: #60A5FA; }

        .section-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: #94A3B8;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border-bottom: 1px solid #1E293B;
            padding-bottom: 0.4rem;
            margin-bottom: 1rem;
        }

        .event-row {
            background: #1E293B;
            border-left: 4px solid #334155;
            border-radius: 6px;
            padding: 0.6rem 1rem;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
        }
        .attack-border  { border-left-color: #EF4444; }
        .warning-border { border-left-color: #EAB308; }
        .normal-border  { border-left-color: #22C55E; }

        .demo-badge {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-size: 0.8rem;
            color: #EAB308;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 1rem;
        }

        .device-detail-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state — build Digital Twin once per session
# ---------------------------------------------------------------------------

@st.cache_resource
def get_digital_twin():
    """Build and return the campus topology + state manager (cached)."""
    topo = create_campus_topology()
    mgr = DigitalTwinStateManager(topo)
    return topo, mgr


# ---------------------------------------------------------------------------
# Load models results (if trained)
# ---------------------------------------------------------------------------

def load_model_results() -> list[dict]:
    paths = [
        Path("results/evaluation_results.json"),
        Path("models/training_results.json"),
    ]
    for p in paths:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATUS_COLOR_MAP = {
    "NORMAL":       "#22C55E",
    "WARNING":      "#EAB308",
    "SUSPICIOUS":   "#F97316",
    "UNDER_ATTACK": "#EF4444",
    "OFFLINE":      "#64748B",
}

STATUS_EMOJI = {
    "NORMAL":       "🟢",
    "WARNING":      "🟡",
    "SUSPICIOUS":   "🟠",
    "UNDER_ATTACK": "🔴",
    "OFFLINE":      "⚫",
}


def render_metric(label: str, value: str, css_class: str = "neutral-color") -> str:
    return f"""
    <div class="metric-card">
        <h3 class="{css_class}">{value}</h3>
        <p>{label}</p>
    </div>
    """


# ---------------------------------------------------------------------------
# Topology figure
# ---------------------------------------------------------------------------

def build_topology_figure(topo, selected_status: str = "All") -> plt.Figure:
    G = topo.graph
    devices = {d.device_id: d for d in topo.all_devices()}

    pos = {}
    for node in G.nodes():
        pos[node] = FIXED_POSITIONS.get(node, (0, 0))

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_facecolor("#0D1117")
    fig.patch.set_facecolor("#0D1117")

    # Highlight selected status
    if selected_status != "All":
        highlight = {n for n, d in devices.items() if d.status == selected_status}
        edge_alpha = [(0.6 if (u in highlight or v in highlight) else 0.1) for u, v in G.edges()]
    else:
        highlight = set(devices.keys())
        edge_alpha = [0.5] * G.number_of_edges()

    edges = list(G.edges())
    for (u, v), alpha in zip(edges, edge_alpha):
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color="#334155",
            alpha=alpha,
            linewidth=1.2,
            zorder=1,
        )

    # Nodes
    for dev_type, shape in TYPE_SHAPES.items():
        node_ids = [
            n for n in G.nodes()
            if devices.get(n) and devices[n].device_type == dev_type
        ]
        if not node_ids:
            continue
        node_colors = [TYPE_COLORS.get(dev_type, "#BDC3C7")] * len(node_ids)
        border_colors = [STATUS_BORDER.get(devices[n].status, "#22C55E") for n in node_ids]
        sizes = [1000 if n in highlight else 400 for n in node_ids]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=node_ids,
            node_color=node_colors,
            edgecolors=border_colors,
            node_shape=shape,
            node_size=sizes,
            linewidths=2.5,
            ax=ax,
        )

    # Labels
    nx.draw_networkx_labels(
        G, pos,
        labels={n: n for n in G.nodes()},
        font_size=5.5,
        font_color="#F8FAFC",
        font_weight="bold",
        ax=ax,
    )

    legend_patches = [
        mpatches.Patch(color=c, label=t)
        for t, c in TYPE_COLORS.items()
    ]
    ax.legend(
        handles=legend_patches, loc="lower left", fontsize=7,
        framealpha=0.3, labelcolor="white", facecolor="#1E293B",
        edgecolor="#334155", title="Device Type", title_fontsize=8,
    )

    status_patches = [
        mpatches.Patch(color=c, label=s)
        for s, c in STATUS_COLOR_MAP.items()
    ]
    ax2 = ax.twinx() if False else ax  # reuse axis
    ax.legend(
        handles=legend_patches + status_patches,
        loc="lower left", fontsize=7,
        framealpha=0.3, labelcolor="white", facecolor="#1E293B",
        edgecolor="#334155",
        ncol=2,
    )

    ax.set_title(
        "Campus Digital Twin — Live Network Topology",
        fontsize=13, fontweight="bold", color="#F8FAFC", pad=12,
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Demo Controls
# ---------------------------------------------------------------------------

def run_sample_traffic(mgr: DigitalTwinStateManager, topo) -> int:
    """Run a sample of real ML predictions on test data and update the Digital Twin."""
    from ml.preprocess import load_processed
    from ml.predict import FlowPredictor, add_demo_ips

    try:
        _, X_test, _, y_test = load_processed()
    except FileNotFoundError:
        return 0

    predictor = FlowPredictor()
    test_flows = add_demo_ips(X_test)
    result = predictor.predict_batch(test_flows, state_manager=mgr, dst_ip_col="destination_ip", pre_scaled=True)
    return len(result)


def simulate_attack(mgr: DigitalTwinStateManager, topo) -> None:
    """Simulate a targeted attack on a specific device."""
    # Pick a device and force it to UNDER_ATTACK with high risk
    target_devices = ["SRV-WEB", "SRV-DB", "CCTV-01", "STU-PC01"]
    import random
    target = random.choice(target_devices)
    mgr.update_risk_score(target, 0.95, reason="Simulated targeted attack", auto_transition=True)


def reset_twin(mgr: DigitalTwinStateManager, topo) -> None:
    """Reset all devices to NORMAL state."""
    for dev in topo.all_devices():
        if dev.status != "NORMAL":
            mgr.reset_device(dev.device_id, reason="Manual reset via dashboard")


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    # Sidebar
    st.sidebar.image(
        "https://img.icons8.com/color/96/security-shield-green.png", width=80
    )
    st.sidebar.title("Digital Twin")
    st.sidebar.markdown("**Campus Cyber Defense**")
    st.sidebar.markdown("---")

    # Demo badge
    st.sidebar.markdown(
        '<div class="demo-badge">⚠️ Controlled Simulation — No Live Network Traffic</div>',
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview", "🗺️ Network Topology", "📋 Device Details",
         "⚠️ Security Events", "📈 ML Performance", "🎮 Demo Controls"],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Amity School of Engg. & Tech.**")
    st.sidebar.markdown("Minor Project — B.Tech CSE (IoT)")
    st.sidebar.markdown("*Team: Abhishek, Tanmay, Atharva*")

    topo, mgr = get_digital_twin()
    summary = mgr.get_topology_summary()
    status_counts = summary["status_counts"]

    # ── Overview ──────────────────────────────────────────────────────────────
    if page == "📊 Overview":
        st.title("🛡️ Campus Cyber Defense Dashboard")
        st.markdown(
            "AI-Driven Digital Twin Framework — real-time security monitoring "
            "for educational institution networks."
        )
        st.markdown("---")
        st.markdown('<p class="section-header">Network Status Overview</p>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(render_metric("Total Devices", str(summary["total_devices"]), "neutral-color"), unsafe_allow_html=True)
        with c2:
            st.markdown(render_metric("Normal", str(status_counts.get("NORMAL", 0)), "normal-color"), unsafe_allow_html=True)
        with c3:
            st.markdown(render_metric("Warning", str(status_counts.get("WARNING", 0)), "warning-color"), unsafe_allow_html=True)
        with c4:
            st.markdown(render_metric("Suspicious", str(status_counts.get("SUSPICIOUS", 0)), "suspicious-color"), unsafe_allow_html=True)
        with c5:
            st.markdown(render_metric("Under Attack", str(status_counts.get("UNDER_ATTACK", 0)), "attack-color"), unsafe_allow_html=True)

        st.markdown("---")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown('<p class="section-header">Status Distribution</p>', unsafe_allow_html=True)
            if status_counts:
                fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
                fig_pie.patch.set_facecolor("#0D1117")
                ax_pie.set_facecolor("#0D1117")
                labels = list(status_counts.keys())
                sizes = list(status_counts.values())
                colors = [STATUS_COLOR_MAP.get(l, "#94A3B8") for l in labels]
                wedges, texts, autotexts = ax_pie.pie(
                    sizes, labels=labels, colors=colors,
                    autopct="%1.0f%%", startangle=90,
                    textprops={"color": "#F8FAFC", "fontsize": 9},
                )
                for at in autotexts:
                    at.set_fontsize(8)
                ax_pie.set_title("Device Status Distribution", color="#F8FAFC", fontsize=11)
                st.pyplot(fig_pie)
                plt.close(fig_pie)

        with col_right:
            st.markdown('<p class="section-header">Compromised Devices</p>', unsafe_allow_html=True)
            compromised = mgr.get_compromised_devices()
            if compromised:
                for dev in compromised:
                    emoji = STATUS_EMOJI.get(dev["status"], "❓")
                    border = "attack-border" if dev["status"] == "UNDER_ATTACK" else "warning-border"
                    st.markdown(
                        f'<div class="event-row {border}">'
                        f'{emoji} <b>{dev["device_id"]}</b> — {dev["name"]}<br>'
                        f'<small>Status: {dev["status"]} | Risk: {dev["risk_score"]:.2f} | {dev["location"]}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ No compromised devices detected.")

        # Recent events
        st.markdown("---")
        st.markdown('<p class="section-header">Recent Events</p>', unsafe_allow_html=True)
        events = mgr.get_event_history(limit=8)
        if events:
            for ev in events:
                status = ev.get("new_status", "")
                border = "attack-border" if status == "UNDER_ATTACK" else (
                    "warning-border" if status in {"WARNING", "SUSPICIOUS"} else "normal-border"
                )
                emoji = STATUS_EMOJI.get(status, "❓")
                pred = ev.get("prediction", ev.get("reason", ""))
                st.markdown(
                    f'<div class="event-row {border}">'
                    f'{emoji} <b>[{ev["timestamp"]}]</b> — {ev["device_id"]} ({ev.get("device_name","")})<br>'
                    f'<small>→ {status} | {pred} | risk={ev.get("risk_score",0):.2f}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No events recorded yet.")

    # ── Network Topology ─────────────────────────────────────────────────────
    elif page == "🗺️ Network Topology":
        st.title("🗺️ Campus Network Topology")
        col1, col2 = st.columns([3, 1])
        with col2:
            status_filter = st.selectbox(
                "Highlight Status",
                ["All", "NORMAL", "WARNING", "SUSPICIOUS", "UNDER_ATTACK", "OFFLINE"],
            )
        fig = create_interactive_topology(topo, mgr, height=700)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="section-header">Zone Summary</p>', unsafe_allow_html=True)
        zone_data: dict[str, list] = {}
        for dev in topo.all_devices():
            zone_data.setdefault(dev.location, []).append(dev)
        for zone, devs in sorted(zone_data.items()):
            attacked = sum(1 for d in devs if d.is_compromised)
            bar = "🔴" * attacked + "🟢" * (len(devs) - attacked)
            st.markdown(f"**{zone}**: {len(devs)} devices {bar}")

    # ── Device Details ────────────────────────────────────────────────────────
    elif page == "📋 Device Details":
        st.title("📋 Device Details")

        # Device selector
        device_options = {f"{d.device_id} — {d.name}": d.device_id for d in sorted(topo.all_devices(), key=lambda d: d.device_id)}
        selected_label = st.selectbox("Select Device", list(device_options.keys()))
        selected_id = device_options[selected_label]
        dev = topo.get_device(selected_id)

        # Device detail card
        st.markdown('<p class="section-header">Device Information</p>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"""
            <div class="device-detail-card">
                <h4 style="margin: 0 0 0.5rem 0; color: #F8FAFC;">{dev.device_id}</h4>
                <p style="margin: 0.25rem 0; color: #94A3B8;"><b>Name:</b> {dev.name}</p>
                <p style="margin: 0.25rem 0; color: #94A3B8;"><b>Type:</b> {dev.device_type}</p>
                <p style="margin: 0.25rem 0; color: #94A3B8;"><b>IP Address:</b> {dev.ip_address}</p>
                <p style="margin: 0.25rem 0; color: #94A3B8;"><b>MAC Address:</b> {dev.mac_address}</p>
                <p style="margin: 0.25rem 0; color: #94A3B8;"><b>Location:</b> {dev.location}</p>
                <p style="margin: 0.25rem 0; color: #94A3B8;"><b>Department:</b> {dev.department}</p>
                <p style="margin: 0.25rem 0; color: #94A3B8;"><b>VLAN:</b> {dev.vlan}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            status_color = STATUS_COLOR_MAP.get(dev.status, "#94A3B8")
            status_emoji = STATUS_EMOJI.get(dev.status, "❓")
            st.markdown(f"""
            <div class="device-detail-card">
                <h4 style="margin: 0 0 0.5rem 0; color: #F8FAFC;">Current State</h4>
                <p style="margin: 0.25rem 0; font-size: 1.5rem;">
                    {status_emoji} <span style="color: {status_color}; font-weight: 700;">{dev.status}</span>
                </p>
                <p style="margin: 0.25rem 0; color: #94A3B8;"><b>Risk Score:</b> <span style="color: {status_color};">{dev.risk_score:.4f}</span></p>
                <p style="margin: 0.25rem 0; color: #94A3B8;"><b>Last Seen:</b> {dev.last_seen.isoformat() if dev.last_seen else 'Never'}</p>
                <p style="margin: 0.25rem 0; color: #94A3B8;"><b>Compromised:</b> {'Yes' if dev.is_compromised else 'No'}</p>
            </div>
            """, unsafe_allow_html=True)

        # Neighbors
        st.markdown('<p class="section-header">Network Neighbors</p>', unsafe_allow_html=True)
        neighbors = topo.get_neighbors(dev.device_id)
        if neighbors:
            neighbor_rows = []
            for n in neighbors:
                neighbor_rows.append({
                    "ID": n.device_id,
                    "Name": n.name,
                    "Type": n.device_type,
                    "IP": n.ip_address,
                    "Status": f"{STATUS_EMOJI.get(n.status,'')} {n.status}",
                    "Risk": round(n.risk_score, 3),
                })
            st.dataframe(pd.DataFrame(neighbor_rows), use_container_width=True, height=200)
        else:
            st.info("No neighbors found.")

        # Event history for this device
        st.markdown('<p class="section-header">Device Event History</p>', unsafe_allow_html=True)
        all_events = mgr.get_event_history()
        device_events = [e for e in all_events if e.get("device_id") == dev.device_id]
        if device_events:
            for ev in device_events[:10]:
                status = ev.get("new_status", "")
                border = "attack-border" if status == "UNDER_ATTACK" else (
                    "warning-border" if status in {"WARNING", "SUSPICIOUS"} else "normal-border"
                )
                emoji = STATUS_EMOJI.get(status, "❓")
                pred = ev.get("prediction", ev.get("reason", ""))
                st.markdown(
                    f'<div class="event-row {border}">'
                    f'{emoji} <b>[{ev["timestamp"]}]</b> {ev["device_id"]} — {ev.get("device_name","")}<br>'
                    f'<small>→ {status} | {pred} | risk={ev.get("risk_score",0):.3f}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No events for this device.")

    # ── Security Events ────────────────────────────────────────────────────────
    elif page == "⚠️ Security Events":
        st.title("⚠️ Security Events")
        events = mgr.get_event_history()
        if not events:
            st.info("No events recorded yet.")
            return

        ev_df = pd.DataFrame(events)
        col1, col2 = st.columns(2)
        with col1:
            pred_filter = st.multiselect(
                "Prediction",
                ev_df["prediction"].dropna().unique().tolist() if "prediction" in ev_df.columns else [],
                default=[],
            )
        with col2:
            dev_filter = st.multiselect(
                "Device",
                ev_df["device_id"].unique().tolist(),
                default=[],
            )

        filtered_ev = ev_df.copy()
        if pred_filter and "prediction" in filtered_ev.columns:
            filtered_ev = filtered_ev[filtered_ev["prediction"].isin(pred_filter)]
        if dev_filter:
            filtered_ev = filtered_ev[filtered_ev["device_id"].isin(dev_filter)]

        # Display as table
        display_cols = ["timestamp", "device_id", "device_name", "prediction", "confidence", "risk_score", "previous_status", "new_status", "source_ip"]
        available_cols = [c for c in display_cols if c in filtered_ev.columns]
        st.dataframe(
            filtered_ev[available_cols].style.background_gradient(subset=["risk_score"], cmap="YlOrRd") if "risk_score" in available_cols else filtered_ev[available_cols],
            use_container_width=True,
            height=500,
        )
        st.caption(f"Total events: {len(ev_df)} | Showing: {len(filtered_ev)}")

    # ── ML Performance ────────────────────────────────────────────────────────
    elif page == "📈 ML Performance":
        st.title("📈 ML Model Performance")
        results = load_model_results()

        if not results:
            st.warning(
                "No trained model results found.\n\n"
                "To train the models, first place a dataset in `data/raw/` then run:\n"
                "```\npython -m ml.preprocess\npython -m ml.train\npython -m ml.evaluate\n```"
            )
            st.info(
                "See `docs/dataset.md` for CICIDS2017 / UNSW-NB15 download instructions."
            )
            return

        for m in results:
            st.markdown(f"### {m['model']} ({m['type']})")
            cols = st.columns(5)
            metrics_display = [
                ("Accuracy", f"{m['accuracy']*100:.2f}%", "normal-color"),
                ("Precision", f"{m['precision']:.4f}", "neutral-color"),
                ("Recall", f"{m['recall']:.4f}", "neutral-color"),
                ("F1-Score", f"{m['f1_score']:.4f}", "neutral-color"),
                ("ROC-AUC", f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "N/A", "neutral-color"),
            ]
            for col, (label, val, css) in zip(cols, metrics_display):
                col.markdown(render_metric(label, val, css), unsafe_allow_html=True)

            if "note" in m:
                st.info(f"ℹ️ {m['note']}")

            # CM image
            safe_name = m["model"].replace(" ", "_").replace("/", "_")
            cm_img = Path(f"results/{safe_name}_confusion_matrix.png")
            roc_img = Path(f"results/{safe_name}_roc_curve.png")
            col_cm, col_roc = st.columns(2)
            with col_cm:
                if cm_img.exists():
                    st.image(str(cm_img), caption="Confusion Matrix")
            with col_roc:
                if roc_img.exists():
                    st.image(str(roc_img), caption="ROC Curve")
            st.markdown("---")

        if len(results) > 1:
            st.subheader("Comparison Table")
            comp_rows = []
            for m in results:
                comp_rows.append({
                    "Model": m["model"],
                    "Type": m["type"],
                    "Accuracy": f"{m['accuracy']*100:.2f}%",
                    "Precision": m["precision"],
                    "Recall": m["recall"],
                    "F1-Score": m["f1_score"],
                    "ROC-AUC": m.get("roc_auc", "N/A"),
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
            st.caption(
                "⚠️ Note: Random Forest (supervised) and Isolation Forest (unsupervised) "
                "are trained differently. Direct metric comparison should be interpreted carefully."
            )

    # ── Demo Controls ─────────────────────────────────────────────────────────
    elif page == "🎮 Demo Controls":
        st.title("🎮 Demo Controls")
        st.markdown(
            "Controlled demonstration interface for the Digital Twin. "
            "All actions operate entirely in software — no live network traffic."
        )
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<p class="section-header">Run Sample Traffic</p>', unsafe_allow_html=True)
            st.markdown(
                "Processes a sample of CICIDS2017 test flows through the trained "
                "Random Forest model and updates the Digital Twin with real predictions."
            )
            if st.button("▶️ Run Sample Traffic", type="primary", use_container_width=True):
                with st.spinner("Running ML predictions on test data..."):
                    count = run_sample_traffic(mgr, topo)
                st.success(f"Processed {count} flows. Digital Twin updated.")
                st.rerun()

        with col2:
            st.markdown('<p class="section-header">Simulate Attack</p>', unsafe_allow_html=True)
            st.markdown(
                "Forces a targeted attack on a random critical device "
                "(Web Server, Database, CCTV, or Student PC) to demonstrate "
                "state transitions and event recording."
            )
            if st.button("⚡ Simulate Attack", type="secondary", use_container_width=True):
                simulate_attack(mgr, topo)
                st.success("Simulated attack executed. Check device states.")
                st.rerun()

        with col3:
            st.markdown('<p class="section-header">Reset Digital Twin</p>', unsafe_allow_html=True)
            st.markdown(
                "Restores all devices to NORMAL state with zero risk. "
                "Clears event history for a clean demonstration."
            )
            if st.button("🔄 Reset Twin", use_container_width=True):
                reset_twin(mgr, topo)
                st.success("Digital Twin reset to NORMAL state.")
                st.rerun()

        st.markdown("---")
        st.markdown('<p class="section-header">Current System State</p>', unsafe_allow_html=True)

        # Show current state summary
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Devices", summary["total_devices"])
        with c2:
            st.metric("Normal", status_counts.get("NORMAL", 0))
        with c3:
            st.metric("Warning", status_counts.get("WARNING", 0))
        with c4:
            st.metric("Suspicious", status_counts.get("SUSPICIOUS", 0))
        with c5:
            st.metric("Under Attack", status_counts.get("UNDER_ATTACK", 0))

        # Compromised devices
        compromised = mgr.get_compromised_devices()
        if compromised:
            st.markdown("**Compromised Devices:**")
            for dev in compromised:
                emoji = STATUS_EMOJI.get(dev["status"], "❓")
                st.markdown(f"{emoji} **{dev['device_id']}** ({dev['name']}) — {dev['status']}, risk={dev['risk_score']:.4f}")
        else:
            st.success("✅ No compromised devices detected.")


if __name__ == "__main__":
    main()
