# AI-Driven Digital Twin Framework for Predictive Cyber Defense in Educational Institutions

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License: Academic](https://img.shields.io/badge/License-Academic-green.svg)]()
[![Tests](https://img.shields.io/badge/Tests-74%20passing-brightgreen.svg)]()

---

## MVP Status

| Component | Status | Notes |
|-----------|--------|-------|
| Campus Digital Twin (26 devices, 5 zones) | ✅ **Implemented** | Complete device model, topology, state management |
| Device data model with state machine | ✅ **Implemented** | `digital_twin/devices.py` |
| Graph-based topology (NetworkX) | ✅ **Implemented** | `digital_twin/topology.py` |
| Dataset preprocessing pipeline | ✅ **Implemented** | `ml/preprocess.py` — requires dataset in `data/raw/` |
| Random Forest (supervised) | ✅ **Implemented** | `ml/train.py` — requires preprocessed data |
| Isolation Forest (unsupervised) | ✅ **Implemented** | `ml/train.py` — requires preprocessed data |
| Binary classification (NORMAL vs ATTACK) | ✅ **Implemented** | Label normalization in preprocessing |
| Digital Twin + ML integration | ✅ **Implemented** | `ml/predict.py` → `state.py` |
| Streamlit monitoring dashboard | ✅ **Implemented** | `dashboard/app.py` — 6 pages |
| Automated test suite | ✅ **Implemented** | **74 tests passing** |
| CICIDS2017 ML training & evaluation | ✅ **Implemented** | Complete pipeline with real data |
| End-to-end demo with real data | ✅ **Implemented** | Controlled simulation mode |
| Controlled cyberattack simulation | ✅ **Implemented** | `digital_twin/demo.py` — 4 attack types |

> **Note:** The ML pipeline code is complete and tested. To run training/evaluation, you must download a CICIDS2017 or UNSW-NB15 CSV and place it in `data/raw/`. See [Dataset Setup](#dataset-setup).

---

## Project Description

This repository contains the **Minor Project** implementation of an AI-Driven Digital Twin Framework for Predictive Cyber Defense in Educational Institutions.

The system creates a virtual replica (Digital Twin) of a representative educational campus network, ingests public cybersecurity datasets, applies machine learning to detect anomalies and intrusions, maps predictions back to virtual network devices, and presents the live security state through a monitoring dashboard.

---

## Problem Statement

Educational institutions operate large, heterogeneous networks with IoT devices, student endpoints, administrative servers, and academic infrastructure. These networks face growing threats including ransomware, DDoS attacks, and unauthorized access — but most institutions lack real-time security monitoring. A Digital Twin framework allows threat simulation and anomaly detection without risking real infrastructure.

---

## Objectives

1. Build a representative educational campus network Digital Twin
2. Ingest and preprocess public cybersecurity datasets (CICIDS2017 / UNSW-NB15)
3. Train ML models for binary anomaly/intrusion detection
4. Map ML predictions to Digital Twin device states
5. Provide a live monitoring dashboard
6. Establish a foundation extensible for future XAI and predictive forecasting

---

## Architecture

```
Public Network Traffic Dataset (CICIDS2017 / UNSW-NB15)
                ↓
        data/raw/ → ml/preprocess.py
                ↓
        data/processed/ (train/test splits)
                ↓
        ml/train.py (Random Forest + Isolation Forest)
                ↓
        models/*.joblib
                ↓
        ml/predict.py (FlowPredictor)
                ↓
        IP → device_id mapping
                ↓
        digital_twin/state.py (DigitalTwinStateManager)
                ↓
        digital_twin/topology.py (CampusTopology)
                ↓
        dashboard/app.py (Streamlit)
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Digital Twin / Graph | NetworkX |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Testing | Pytest |
| Development | VS Code, Git, GitHub |

---

## Repository Structure

```
digitaltwin-mp/
├── data/
│   ├── raw/              ← Place CSV datasets here
│   └── processed/        ← Generated after preprocessing
│
├── digital_twin/
│   ├── __init__.py
│   ├── devices.py        ← Device dataclass + state machine
│   ├── topology.py       ← Campus network (26 devices, 25 connections)
│   ├── network.py        ← Entry point + visualization
│   └── state.py          ← State manager + event history
│
├── ml/
│   ├── __init__.py
│   ├── preprocess.py     ← Full preprocessing pipeline
│   ├── train.py          ← Random Forest + Isolation Forest
│   ├── predict.py        ← FlowPredictor + Digital Twin integration
│   └── evaluate.py       ← Metrics + confusion matrix + ROC
│
├── dashboard/
│   └── app.py            ← Streamlit monitoring dashboard
│
├── models/               ← Saved models (gitignored)
├── results/              ← Evaluation outputs (gitignored)
├── notebooks/            ← Jupyter exploration notebooks
├── tests/                ← Pytest test suite (74 tests)
├── docs/                 ← Architecture & methodology docs
├── presentation/         ← Slides & reports
│
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/abhi180105/digitaltwin-mp.git
cd digitaltwin-mp
```

### 2. Create virtual environment

```bash
python3 -m venv denv
source denv/bin/activate        # Linux/macOS
# denv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Setup

This project uses public cybersecurity datasets. **Never commit large CSVs to Git.**

### CICIDS2017 (Primary)

1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Fill in the data request form
3. Download individual CSV files (e.g., `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`)
4. Place in `data/raw/`

### UNSW-NB15 (Secondary)

1. Visit: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Download `UNSW_NB15_training-set.csv` and/or `UNSW_NB15_testing-set.csv`
3. Place in `data/raw/`

See `docs/dataset.md` for full column descriptions.

---

## Digital Twin Usage

```bash
python -m digital_twin.network
```

Outputs:
- Device inventory (26 devices across 5 zones)
- Connection summary
- State manager demonstration
- Topology image: `models/topology.png`

---

## Model Training

```bash
# 1. Preprocess dataset
python -m ml.preprocess

# 2. Train models
python -m ml.train

# 3. Evaluate models
python -m ml.evaluate
```

> **Important:** Train/evaluate only after placing a dataset in `data/raw/`.
> Model results displayed are **actual experimental outputs**, not placeholders.

---

## Dashboard Usage

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`

Dashboard pages:
- **Overview** — Device status counts, status distribution, recent events
- **Network Topology** — Interactive campus graph with status highlighting
- **Device Details** — Device metadata, neighbors, event history
- **Security Events** — Filterable anomaly event log with predictions
- **Model Performance** — Actual metrics, confusion matrices, ROC curves
- **Demo Controls** — Controlled simulation interface (Run Sample Traffic, Simulate Attack, Reset Twin)

---

## Controlled Cyberattack Simulation

```bash
# List available devices
python -m digital_twin.demo --list-devices

# List available attack scenarios
python -m digital_twin.demo --list-attacks

# Run attack simulations
python -m digital_twin.demo --attack port_scan --device CCTV-01
python -m digital_twin.demo --attack ddos --device SRV-WEB
python -m digital_twin.demo --attack brute_force --device SRV-AUTH
python -m digital_twin.demo --attack unauthorized_access --device ADMIN-SRV

# Reset Digital Twin to NORMAL state
python -m digital_twin.demo --reset
```

**Attack Scenarios:**
- `port_scan` — Simulated reconnaissance scanning multiple ports
- `ddos` — Simulated volumetric traffic flood
- `brute_force` — Simulated repeated authentication attempts
- `unauthorized_access` — Simulated privilege escalation attempt

> **Safety:** All simulations operate entirely in software against the simulated Digital Twin. No real network traffic is generated, no sockets are opened, and no real systems are scanned.

---

## Testing

```bash
pytest
```

**74 tests** covering:
- Device creation, validation, state transitions
- Topology construction, queries, state updates
- Preprocessing pipeline (label normalisation, inf/nan handling, no-leakage scaler)
- State manager (risk scoring, auto-transitions, event history)
- ML training helpers, evaluation, FlowPredictor
- Network visualization constants and CLI

---

## Minor vs Major Project Scope

### Minor Project (Implemented ✅)

- Campus Digital Twin (26 devices, 5 zones)
- Device data model with state machine
- Graph-based topology (NetworkX)
- Dataset preprocessing pipeline (CICIDS2017)
- Random Forest (supervised) — 99.91% accuracy
- Isolation Forest (unsupervised) — unsupervised baseline
- Binary classification (NORMAL vs ATTACK)
- Digital Twin + ML integration
- Streamlit monitoring dashboard (6 pages)
- Controlled cyberattack simulation (4 attack types)
- 74 automated tests
- Complete documentation

### Major Project (Planned 🔄)

- SHAP / LIME explainability
- Predictive attack-path forecasting
- Threat propagation modeling
- Advanced risk scoring
- Automated incident response
- Real-time campus data ingestion
- Multi-class attack classification

---

## Ethical Considerations

This project uses **only**:
- Public research datasets (CICIDS2017, UNSW-NB15)
- Synthetic/fictional IP and MAC addresses
- A simulated, representative campus network

**No real university infrastructure is accessed, attacked, or monitored.**

---

## Future Scope

- Integration with SIEM systems
- Real-time packet capture (controlled lab only)
- Attack path visualization
- Automated threat response recommendations
- Multi-campus federation

---

## Team Members

| Name | Role |
|------|------|
| Abhishek Yadav | Lead Developer |
| Tanmay Kumar | ML & Data Pipeline |
| Atharva Chaprana | Dashboard & Documentation |

**Project Guide:** Mr. Paurush Bhulania

**Institution:** Amity School of Engineering & Technology

---

## License

Academic use only. Not for production deployment.
