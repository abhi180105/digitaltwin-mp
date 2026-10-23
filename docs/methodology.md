# Methodology

## Literature Review

This project builds on established research in:
- Network Digital Twins for cyber security (ITU-T Y.3090)
- Intrusion Detection Systems using ML (Tavallaee et al., 2009)
- CICIDS2017 benchmark evaluations (Sharafaldin et al., 2018)

## Topology Design

1. Identified common educational institution network zones
2. Created representative (not real) IP addressing: 192.168.x.x subnets
3. Modelled 5 zones: Server Room, Academic, Admin, Hostel, External
4. 26 devices, 25 bidirectional connections

## Dataset Preparation

- Source: CICIDS2017 (primary), UNSW-NB15 (secondary)
- Binary classification: BENIGN → 0, all attack types → 1
- Train/test split: 80/20 stratified

## ML Training

- **Random Forest**: supervised, class-balanced, n_estimators=100
- **Isolation Forest**: unsupervised, contamination=0.1
- Evaluation metrics: Accuracy, Precision, Recall, F1, FPR, ROC-AUC

## Digital Twin Integration

- IP address → device_id mapping in `digital_twin/topology.py`
- Risk scores drive automatic device status transitions
- Event history captures all state changes

## Ethical Constraints

- All datasets are public research benchmarks
- No real network infrastructure is accessed
- IP/MAC addresses are fictional
