# System Architecture

## Overview

The system is structured as a layered pipeline:

```
┌─────────────────────────────────────────┐
│              Dashboard Layer             │
│         Streamlit (dashboard/app.py)     │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│           Digital Twin Layer             │
│  CampusTopology + DigitalTwinStateManager│
│      digital_twin/ package               │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│            ML Integration Layer          │
│   FlowPredictor (ml/predict.py)          │
│   IP → device_id mapping                 │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│              ML Layer                    │
│   Random Forest  |  Isolation Forest     │
│   ml/train.py    |  ml/evaluate.py       │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│            Data Layer                    │
│   ml/preprocess.py                       │
│   data/raw/ → data/processed/            │
└─────────────────────────────────────────┘
```

---

## Physical Campus Model

The Digital Twin represents a generic educational institution with:

| Zone | Devices |
|------|---------|
| Server Room | Firewall, IDS, Core Router, LMS/Web/DB/Auth/File Servers |
| Academic Block | Switch, Faculty PCs, Student PC, CCTV, RFID, Wi-Fi AP |
| Administration Block | Switch, Admin PC, Admin Server, Access Control |
| Hostel Block | Switch, Student Devices, Wi-Fi AP, IoT, CCTV |
| External | Internet Gateway |

**Total: 26 devices, 25 bidirectional connections**

---

## Data Layer

- Raw CSV datasets stored in `data/raw/` (not committed)
- Preprocessing produces train/test splits in `data/processed/`
- Scaler and feature list persisted in `models/`

---

## Digital Twin Layer

### Device Model (`devices.py`)

```
Device (dataclass)
├── Identity: device_id, name, device_type, ip_address, mac_address
├── Location: location, department, vlan
├── State: status (NORMAL|WARNING|SUSPICIOUS|UNDER_ATTACK|OFFLINE)
├── Risk: risk_score [0.0, 1.0]
└── History: event_log (private)
```

State transitions (manual):
```
NORMAL → WARNING, SUSPICIOUS, OFFLINE
WARNING → NORMAL, SUSPICIOUS, UNDER_ATTACK, OFFLINE
SUSPICIOUS → NORMAL, WARNING, UNDER_ATTACK, OFFLINE
UNDER_ATTACK → SUSPICIOUS, WARNING, NORMAL, OFFLINE
OFFLINE → NORMAL
```

ML-driven transitions bypass these rules (direct assignment).

### Topology (`topology.py`)

- NetworkX DiGraph (bidirectional edges stored as two directed edges)
- Node data: Device object
- Edge data: bandwidth, protocol, zone, status

### State Manager (`state.py`)

- Wraps CampusTopology
- Converts risk scores → device statuses automatically
- Maintains event history for the dashboard
- `record_anomaly()` — primary integration point for ML predictions

---

## ML Layer

### Random Forest (Supervised)
- Trained on labelled CICIDS2017 / UNSW-NB15 data
- Outputs binary label + class probabilities
- `predict_proba()[:,1]` used as risk score

### Isolation Forest (Unsupervised)
- Trained without labels (anomaly detection)
- `predict()` returns +1 (inlier) or -1 (anomaly)
- Mapped to 0/1 for comparison with RF
- Metrics evaluated post-hoc (not truly comparable to supervised RF)

---

## Dashboard Layer

Five pages:
1. **Overview** — Status summary, pie chart, compromised devices, recent events
2. **Network Topology** — NetworkX graph rendered with Matplotlib, status highlighting
3. **Device Inventory** — Filterable table with risk score gradient
4. **Threat Events** — Anomaly event log
5. **Model Performance** — Actual metrics from `results/evaluation_results.json`

---

## Extensibility for Major Project

The architecture supports future additions:

```
XAI/
├── shap/          ← SHAP explanations for RF predictions
└── lime/          ← LIME local explanations

prediction/
├── attack_paths.py    ← Graph-based attack path traversal
└── forecasting.py     ← Time-series threat forecasting

response/
└── recommendations.py ← Automated remediation suggestions
```
