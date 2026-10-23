# Dataset Information

## Primary Dataset: CICIDS2017

**Source:** Canadian Institute for Cybersecurity, University of New Brunswick  
**URL:** https://www.unb.ca/cic/datasets/ids-2017.html  
**Citation:** Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *ICISSP 2018*.

### Description

CICIDS2017 contains benign and common attack network traffic. Data was captured over 5 days (Monday–Friday) in a realistic network environment.

### Attack Categories

| Category | Day |
|----------|-----|
| BENIGN | All days |
| DDoS | Friday AM |
| PortScan | Friday Afternoon |
| Bot | Thursday |
| Web Attacks | Tuesday |
| Brute Force / FTP-BruteForce / SSH-Patator | Tuesday–Wednesday |
| DoS (Hulk, GoldenEye, slowloris, Heartbleed) | Wednesday |
| Infiltration | Thursday |

### Download Instructions

1. Go to: https://www.unb.ca/cic/datasets/ids-2017.html
2. Fill in the data request form
3. Download individual CSV files (e.g., `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`)
4. Place in `data/raw/`

### Expected Columns (sample)

```
Flow ID, Source IP, Source Port, Destination IP, Destination Port,
Protocol, Timestamp, Flow Duration, Total Fwd Packets, Total Bwd Packets,
Total Length of Fwd Packets, Total Length of Bwd Packets,
Fwd Packet Length Max, ..., Label
```

### Label Column

`Label` — values include: `BENIGN`, `DDoS`, `PortScan`, `Bot`, etc.

Our preprocessing normalises this to binary: `0` (BENIGN) and `1` (ATTACK).

---

## Secondary Dataset: UNSW-NB15

**Source:** Australian Centre for Cyber Security (ACCS), UNSW Canberra  
**URL:** https://research.unsw.edu.au/projects/unsw-nb15-dataset  
**Citation:** Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems. *MILCOM 2015*.

### Description

UNSW-NB15 contains contemporary network attack types alongside normal traffic. Includes 9 attack categories and 49 features.

### Attack Categories

Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms

### Download Instructions

1. Go to: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Download `UNSW_NB15_training-set.csv` and/or `UNSW_NB15_testing-set.csv`
3. Place in `data/raw/`

### Label Column

`label` (lowercase) — values: `0` (normal) or `1` (attack).  
Or `attack_cat` for multi-class.

---

## Preprocessing Pipeline

See `ml/preprocess.py` for implementation.

### Steps

1. **Column name cleanup** — strip whitespace
2. **Label column detection** — auto-detects common label column names
3. **Drop metadata columns** — IP addresses, ports, timestamps (prevent leakage)
4. **Duplicate removal**
5. **Inf / NaN handling** — replace inf with NaN, drop NaN rows
6. **Binary label normalisation** — BENIGN → 0, everything else → 1
7. **Feature selection** — numeric columns only
8. **Train/test split** — 80/20, stratified, `random_state=42`
9. **StandardScaler** — fit ONLY on train set; transform both sets

### Output

| File | Description |
|------|-------------|
| `data/processed/X_train.csv` | Scaled training features |
| `data/processed/X_test.csv` | Scaled test features |
| `data/processed/y_train.csv` | Training labels |
| `data/processed/y_test.csv` | Test labels |
| `models/scaler.joblib` | Fitted StandardScaler |
| `models/feature_names.joblib` | Feature column names |

> **Note:** Processed files and models are excluded from Git (see `.gitignore`).
> Regenerate by running `python -m ml.preprocess` after placing data in `data/raw/`.
