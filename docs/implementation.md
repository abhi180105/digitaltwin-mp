# Implementation Guide

## Prerequisites

- Python 3.11 or newer
- Git
- ~2 GB disk space for datasets

---

## Setup

```bash
git clone https://github.com/abhi180105/digitaltwin-mp.git
cd digitaltwin-mp
python3 -m venv denv
source denv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 1 — Run the Digital Twin

```bash
python -m digital_twin.network
```

Expected output:
```
============================================================
  AI-Driven Digital Twin — Educational Campus Network
============================================================
Devices      : 26
Connections  : 25
...
Topology image saved: models/topology.png
```

The topology PNG is saved to `models/topology.png`.

---

## Step 2 — Dataset Setup

Place a CSV in `data/raw/`:
- CICIDS2017 CSV: https://www.unb.ca/cic/datasets/ids-2017.html
- UNSW-NB15 CSV: https://research.unsw.edu.au/projects/unsw-nb15-dataset

```bash
ls data/raw/
# Should show: your_dataset.csv
```

---

## Step 3 — Preprocess

```bash
python -m ml.preprocess
```

Expected output:
```
Loaded 123456 rows × 79 columns.
Dropped 4 metadata columns.
Selected 72 numeric features.
Train: 98764 rows | Test: 24692 rows | Attack ratio (train): 45.23%
Preprocessing complete.
```

Output files:
- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `models/scaler.joblib`
- `models/feature_names.joblib`

---

## Step 4 — Train Models

```bash
python -m ml.train
```

Trains:
- `models/random_forest.joblib`
- `models/isolation_forest.joblib`
- `models/training_results.json`

---

## Step 5 — Evaluate Models

```bash
python -m ml.evaluate
```

Generates:
- `results/evaluation_results.json`
- `results/Random_Forest_confusion_matrix.png`
- `results/Random_Forest_roc_curve.png`
- `results/Isolation_Forest_*` equivalents

---

## Step 6 — Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open browser at: http://localhost:8501

---

## Step 7 — Run Tests

```bash
pytest
```

Expected: **61 passed**

---

## Troubleshooting

### Dataset not found
```
Dataset not found.
Place a CICIDS2017 or UNSW-NB15 CSV inside data/raw/
```
→ Download and place a CSV in `data/raw/`.

### Model not trained
```
Model not found. Run the training pipeline first: python -m ml.train
```
→ Run preprocessing then training.

### Streamlit port in use
```bash
streamlit run dashboard/app.py --server.port 8502
```

### Import errors
Make sure the virtual environment is activated:
```bash
source denv/bin/activate
```
