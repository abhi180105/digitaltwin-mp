# Experiments

## Status

> **✅ Experiments completed with real CICIDS2017 data.**
> Model metrics below are **actual experimental results** from `python -m ml.evaluate`.
> Dataset: `Wednesday-workingHours.pcap_ISCX.csv` (692,703 raw rows, 584,991 after cleaning)

---

## Experiment Log

| Date | Dataset | Model | Accuracy | F1 | AUC | Notes |
|------|---------|-------|----------|----|-----|-------|
| 2026-08-26 | CICIDS2017 (Wednesday) | Random Forest | 0.9991 | 0.9987 | 0.9999 | Supervised, class_weight=balanced |
| 2026-08-26 | CICIDS2017 (Wednesday) | Isolation Forest | 0.6414 | 0.1718 | N/A | Unsupervised, contamination=0.1 |

---

## Detailed Results

### Random Forest (Primary MVP Classifier)

**Configuration:**
- n_estimators: 100
- max_depth: None
- class_weight: balanced
- random_state: 42
- n_jobs: -1

**Training Data:**
- Training rows: 467,992
- Test rows: 116,999
- Features: 77 (numeric network-flow features)
- Classes: 2 (0=BENIGN, 1=ATTACK)
- Class distribution (train): BENIGN=312,987 (66.9%), ATTACK=155,005 (33.1%)

**Test Set Metrics:**
| Metric | Value |
|--------|-------|
| Accuracy | 0.9991 (99.91%) |
| Precision | 0.9981 |
| Recall | 0.9992 |
| F1-Score | 0.9987 |
| False Positive Rate | 0.0009 (0.09%) |
| ROC-AUC | 0.9999 |

**Confusion Matrix:**
| | Predicted NORMAL | Predicted ATTACK |
|---|---|---|
| **Actual NORMAL** | TN=78,176 | FP=72 |
| **Actual ATTACK** | FN=30 | TP=38,721 |

**Error Analysis:**
- False Positives: 72 (0.09% of NORMAL traffic incorrectly flagged)
- False Negatives: 30 (0.08% of ATTACK traffic missed)
- Class imbalance handled via `class_weight="balanced"`

---

### Isolation Forest (Unsupervised Baseline)

**Configuration:**
- n_estimators: 100
- contamination: 0.1
- random_state: 42
- n_jobs: -1

**Note:** Isolation Forest is trained WITHOUT labels. Metrics are evaluated post-hoc for comparison only and are not directly comparable to supervised Random Forest.

**Test Set Metrics:**
| Metric | Value |
|--------|-------|
| Accuracy | 0.6414 (64.14%) |
| Precision | 0.3655 |
| Recall | 0.1123 |
| F1-Score | 0.1718 |
| False Positive Rate | 0.0966 (9.66%) |
| ROC-AUC | N/A (no probability outputs) |

**Confusion Matrix:**
| | Predicted NORMAL | Predicted ATTACK |
|---|---|---|
| **Actual NORMAL** | TN=70,692 | FP=7,556 |
| **Actual ATTACK** | FN=34,399 | TP=4,352 |

**Error Analysis:**
- High false negative rate (88.8% of attacks missed) — expected for unsupervised method
- High false positive rate (9.66%) — many benign flows flagged as anomalies
- Useful as complementary anomaly detector, not primary classifier

---

## Reproducibility Information

| Parameter | Value |
|-----------|-------|
| Dataset source | CICIDS2017 (UNB) |
| Dataset file | Wednesday-workingHours.pcap_ISCX.csv |
| Dataset version | Original release (2017) |
| Random seed | 42 (all stages) |
| Train/test split | 80/20 stratified |
| Preprocessing | Column cleanup, duplicate removal, NaN/Inf handling, label normalization (BENIGN→0, ATTACK→1), numeric feature selection, StandardScaler (fit on train only) |
| Feature count | 77 |
| Model artifacts | `models/random_forest.joblib`, `models/isolation_forest.joblib`, `models/scaler.joblib`, `models/feature_names.joblib` |
| Results artifacts | `results/evaluation_results.json`, `results/*_confusion_matrix.png`, `results/*_roc_curve.png` |
| Timestamp | 2026-08-26T06:50:22 |

---

## How to Reproduce

```bash
# 1. Preprocess (requires CICIDS2017 CSV in data/raw/)
python -m ml.preprocess

# 2. Train models
python -m ml.train

# 3. Evaluate
python -m ml.evaluate
```

Results will be saved to `results/evaluation_results.json` and plots to `results/`.

---

## Dashboard Integration

The dashboard (`dashboard/app.py`) reads `results/evaluation_results.json` and `results/*_confusion_matrix.png` / `results/*_roc_curve.png` to display model performance on the **Model Performance** page.

---

## Notes

- **Never fabricate or estimate experimental results.** All numbers above come from actual executed runs.
- The Random Forest achieves near-perfect metrics on this dataset split. This may indicate the Wednesday dataset has distinct attack patterns that are easily separable. Cross-day validation would be needed for production deployment.
- Isolation Forest serves as an unsupervised baseline; its low recall is expected without label supervision.

---

## Cross-Day Generalization

### Why Cross-Day Validation?

The original Random Forest experiment used an 80/20 stratified split on a **single day** (Wednesday) of CICIDS2017 data. While this yields near-perfect metrics (99.91% accuracy), it does not test whether the model generalizes to network traffic from different days with different attack types and traffic patterns.

Cross-day validation answers: **How well does a model trained on one day generalize to completely unseen traffic from another day?**

### Methodology

**Training/Test Separation:**
- **No random splitting across days** — this would leak future information
- **Train day**: All records from one calendar day (e.g., Wednesday)
- **Test day**: All records from a different calendar day (e.g., Thursday, Friday)
- Multiple files belonging to the same calendar day are combined (e.g., Thursday has Morning WebAttacks + Afternoon Infiltration)

**Preprocessing (No Leakage):**
- All transformations fitted **only on training day**:
  - Column cleaning, duplicate removal, NaN/Inf handling
  - Label normalization (BENIGN→0, ATTACK→1)
  - Feature selection (77 numeric features)
  - **StandardScaler fitted on train day only**
- Test day receives **identical transformations** using fitted parameters

**Model Configuration (Same as Baseline):**
- Random Forest: n_estimators=100, max_depth=None, class_weight="balanced", random_state=42
### Dataset Inventory

| Day | Files | Total Rows | Attack Labels |
|-----|-------|------------|---------------|
| Monday | 1 | 529,918 | — |
| Tuesday | 1 | 445,909 | FTP-Patator, SSH-Patator |
| **Wednesday** | 1 | 692,703 | DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, Heartbleed |
| **Thursday** | 2 | 458,968 | Infiltration, Web Attack – Brute Force, Web Attack – Sql Injection, Web Attack – XSS |
| **Friday** | 3 | 703,245 | Bot, DDoS, PortScan |

### Experiments Run

| Experiment | Train Day | Test Day | Train Samples | Test Samples |
|------------|-----------|----------|---------------|--------------|
| A | Wednesday | Thursday | 584,991 | 356,302 |
| B | Wednesday | Friday | 584,991 | 507,198 |
| C | Thursday | Friday | 356,302 | 507,198 |

### Results

#### Experiment A: Wednesday → Thursday

| Metric | Value |
|--------|-------|
| Accuracy | 0.9932 (99.32%) |
| Precision | 0.2762 |
| Recall | 0.0702 |
| F1-Score | 0.1120 |
| ROC-AUC | 0.9763 |
| False Positive Rate | 0.0011 (0.11%) |

**Confusion Matrix:**
| | Predicted NORMAL | Predicted ATTACK |
|---|---|---|
| **Actual NORMAL** | TN=353,722 | FP=401 |
| **Actual ATTACK** | FN=2,026 | TP=153 |

#### Experiment B: Wednesday → Friday

| Metric | Value |
|--------|-------|
| Accuracy | 0.8512 (85.12%) |
| Precision | 0.9956 |
| Recall | 0.4277 |
| F1-Score | 0.5983 |
| ROC-AUC | 0.9433 |
| False Positive Rate | 0.0007 (0.07%) |

**Confusion Matrix:**
| | Predicted NORMAL | Predicted ATTACK |
|---|---|---|
| **Actual NORMAL** | TN=375,679 | FP=270 |
| **Actual ATTACK** | FN=145,072 | TP=108,877 |

#### Experiment C: Thursday → Friday

| Metric | Value |
|--------|-------|
| Accuracy | 0.7409 (74.09%) |
| Precision | 0.3333 |
| Recall | 0.0001 |
| F1-Score | 0.0002 |
| ROC-AUC | 0.7747 |
| False Positive Rate | 0.0001 (0.01%) |

**Confusion Matrix:**
| | Predicted NORMAL | Predicted ATTACK |
|---|---|---|
| **Actual NORMAL** | TN=375,769 | FP=22 |
| **Actual ATTACK** | FN=131,396 | TP=11 |

### Comparison with Random Split Baseline

| Metric | Random 80/20 (Wed) | Wed→Thu | Wed→Fri | Thu→Fri |
|--------|---|---|---|---|
| Accuracy | 0.9991 | 0.9932 | 0.8512 | 0.7409 |
| Precision | 0.9981 | 0.2762 | 0.9956 | 0.3333 |
| Recall | 0.9992 | 0.0702 | 0.4277 | 0.0001 |
| F1-Score | 0.9987 | 0.1120 | 0.5983 | 0.0002 |
| ROC-AUC | 0.9999 | 0.9763 | 0.9433 | 0.7747 |
| FPR | 0.0009 | 0.0011 | 0.0007 | 0.0001 |

### Analysis

**Performance decreases significantly across days:**

1. **Wednesday → Thursday**: Accuracy remains high (99.32%) but **Recall drops to 7%**. The model misses 93% of Thursday's attacks (Web Attacks, Infiltration) because these attack types were not present in Wednesday's DoS-heavy traffic.

2. **Wednesday → Friday**: Accuracy drops to 85%, Recall to 43%. Friday's mix of Bot, DDoS, and PortScan attacks are partially detected but many are missed.

3. **Thursday → Friday**: Accuracy 74%, Recall near 0%. The model trained on web attacks/infiltration fails completely on Friday's DDoS/PortScan/Bot traffic.

**Key Observations:**
- **High accuracy can be misleading** — it's driven by the majority class (BENIGN)
- **Recall is the critical metric** for security — all cross-day experiments show severe Recall degradation
- **Attack type distribution shift** is the primary cause: Wednesday=DoS, Thursday=Web/Infiltration, Friday=DDoS/PortScan/Bot
- **ROC-AUC remains relatively high** (0.97-0.77) indicating the model ranks attacks higher than benign, but the decision threshold (0.5) is poorly calibrated for new attack types

### Limitations

1. **Single training day** — only Wednesday was used as train day; more combinations (Mon→Tue, Tue→Wed) could be explored
2. **Class imbalance varies by day** — Thursday has only 0.6% attacks vs 33% on Wednesday
3. **No temporal adaptation** — model doesn't update with new data
4. **CICIDS2017 is dated (2017)** — modern attack patterns may differ further
5. **No feature engineering for day-invariant patterns** — all 77 raw flow features used

### Artifacts

- `results/cross_day_validation.json` — Full metrics per experiment
- `results/cross_day_validation.csv` — Comparison table
- `results/cross_day/wednesday_to_thursday_confusion_matrix.png`
- `results/cross_day/wednesday_to_friday_confusion_matrix.png`
- `results/cross_day/thursday_to_friday_confusion_matrix.png`

### How to Reproduce

```bash
# Run cross-day validation
python -m ml.cross_day_validation

# Results saved to results/cross_day_validation.json and results/cross_day/
```

### Dashboard Integration

The dashboard reads `results/cross_day_validation.json` and confusion matrix images from `results/cross_day/` for display on the Model Performance page.
- No hyperparameter tuning — this is a generalization test, not model optimization

---

## Multi-Day Training and Held-Out-Day Validation

### Motivation

The single-day cross-day validation experiments revealed severe generalization degradation when a model trained on one day of CICIDS2017 data is tested on a completely different day. The key finding was that **attack-type distribution shift** between days (Wednesday=DoS, Thursday=Web/Infiltration, Friday=DDoS/PortScan/Bot) causes Recall to drop from 99.92% (same-day) to as low as 0.01% (cross-day).

This raises the research question: **Does training on multiple days of diverse attack types improve generalization to a held-out day?**

### Research Question

> "Does training the Random Forest model on multiple CICIDS2017 capture days improve generalization to a previously unseen day?"

### Experimental Design

**Training/Test Separation (No Leakage):**
- Training data: All records from selected training days combined
- Test data: All records from a completely separate held-out day
- Preprocessing (scaler, feature selection) fitted ONLY on training data
- Test day transformed using training-day artifacts only
- No random splitting across days — strict day-level separation

**Random Forest Configuration (Identical to Baseline):**
- n_estimators: 100
- max_depth: None
- class_weight: "balanced"
- random_state: 42
- n_jobs: -1
- No hyperparameter tuning

**Experiments:**

| Experiment | Train Days | Test Day | Train Files | Test Files |
|------------|-----------|----------|-------------|------------|
| A | Wednesday + Thursday | Friday | 3 | 3 |
| B | Wednesday + Friday | Thursday | 4 | 2 |
| C | Thursday + Friday | Wednesday | 5 | 1 |

### Dataset Inventory (with Attack Types)

| Day | Files | Total Rows | Benign | Attack | Attack Types |
|-----|-------|------------|--------|--------|--------------|
| Monday | 1 | 529,918 | 529,918 | 0 | — |
| Tuesday | 1 | 445,909 | 432,074 | 13,835 | FTP-Patator, SSH-Patator |
| Wednesday | 1 | 692,703 | 440,031 | 252,672 | DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed |
| Thursday | 2 | 458,968 | 456,752 | 2,216 | Web Attack Brute Force, Web Attack XSS, Infiltration, Web Attack Sql Injection |
| Friday | 3 | 703,245 | 414,322 | 288,923 | PortScan, DDoS, Bot |
### Results

#### Experiment A: Wednesday + Thursday → Friday

| Metric | Value |
|--------|-------|
| Train Samples | 924,066 |
| Test Samples | 507,198 |
| Features | 77 |
| Training Time | 13.14s |
| Benign (Train) | 728,131 |
| Attack (Train) | 195,935 (21.20%) |
| Benign (Test) | 375,791 |
| Attack (Test) | 131,407 (25.91%) |
| **Accuracy** | **0.7829 (78.29%)** |
| **Precision** | **0.9892** |
| **Recall** | **0.1638** |
| **F1-Score** | **0.2811** |
| **ROC-AUC** | **0.9289** |
| **False Positive Rate** | **0.0006 (0.06%)** |

**Confusion Matrix:**
| | Predicted NORMAL | Predicted ATTACK |
|---|---|---|
| **Actual NORMAL** | TN=375,557 | FP=234 |
| **Actual ATTACK** | FN=109,883 | TP=21,524 |

**Test Attack Types:** PortScan (158,930), DDoS (128,027), Bot (1,966)

---

#### Experiment B: Wednesday + Friday → Thursday

| Metric | Value |
|--------|-------|
| Train Samples | 1,074,704 |
| Test Samples | 356,302 |
| Features | 77 |
| Training Time | 19.32s |
| Benign (Train) | 749,541 |
| Attack (Train) | 325,163 (30.26%) |
| Benign (Test) | 354,123 |
| Attack (Test) | 2,179 (0.61%) |
| **Accuracy** | **0.9968 (99.68%)** |
| **Precision** | **0.7803** |
| **Recall** | **0.6732** |
| **F1-Score** | **0.7228** |
| **ROC-AUC** | **0.9728** |
| **False Positive Rate** | **0.0012 (0.12%)** |

**Confusion Matrix:**
| | Predicted NORMAL | Predicted ATTACK |
|---|---|---|
| **Actual NORMAL** | TN=353,710 | FP=413 |
| **Actual ATTACK** | FN=712 | TP=1,467 |

**Test Attack Types:** Web Attack Brute Force (1,507), Web Attack XSS (652), Infiltration (36), Web Attack Sql Injection (21)

---

#### Experiment C: Thursday + Friday → Wednesday

| Metric | Value |
|--------|-------|
| Train Samples | 846,647 |
| Test Samples | 584,991 |
| Features | 77 |
| Training Time | 17.16s |
| Benign (Train) | 713,061 |
| Attack (Train) | 133,586 (15.78%) |
| Benign (Test) | 391,235 |
| Attack (Test) | 193,756 (33.12%) |
| **Accuracy** | **0.6727 (67.27%)** |
| **Precision** | **0.9761** |
| **Recall** | **0.0122** |
| **F1-Score** | **0.0242** |
| **ROC-AUC** | **0.9694** |
| **False Positive Rate** | **0.0001 (0.01%)** |

**Confusion Matrix:**
| | Predicted NORMAL | Predicted ATTACK |
|---|---|---|
| **Actual NORMAL** | TN=391,177 | FP=58 |
| **Actual ATTACK** | FN=191,386 | TP=2,370 |

---

### Comparison with Existing Results

| Experiment | Train Data | Test Data | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------------|------------|-----------|----------|-----------|--------|-----|---------|
| **Baseline (Random 80/20)** | Wednesday | Wednesday | **0.9991** | **0.9981** | **0.9992** | **0.9987** | **0.9999** |
| Single-day | Wednesday | Thursday | 0.9932 | 0.2762 | 0.0702 | 0.1120 | 0.9763 |
| Single-day | Wednesday | Friday | 0.8512 | 0.9956 | 0.4277 | 0.5983 | 0.9433 |
| Single-day | Thursday | Friday | 0.7409 | 0.3333 | 0.0001 | 0.0002 | 0.7747 |
| **Multi-day (A)** | **Wednesday+Thursday** | **Friday** | **0.7829** | **0.9892** | **0.1638** | **0.2811** | **0.9289** |
| **Multi-day (B)** | **Wednesday+Friday** | **Thursday** | **0.9968** | **0.7803** | **0.6732** | **0.7228** | **0.9728** |
| **Multi-day (C)** | **Thursday+Friday** | **Wednesday** | **0.6727** | **0.9761** | **0.0122** | **0.0242** | **0.9694** |

### Analysis

#### Does Multi-Day Training Improve Cross-Day Generalization?

**Experiment A (Wed+Thu → Fri):**
- **Recall**: 0.1638 (multi-day) vs 0.4277 (single-day Wed→Fri) — **DECREASED**
- **F1**: 0.2811 (multi-day) vs 0.5983 (single-day) — **DECREASED**
- **Precision**: 0.9892 (multi-day) vs 0.9956 (single-day) — similar
- **Observation**: Adding Thursday's web/infiltration data to Wednesday's DoS data did not help detect Friday's DDoS/PortScan/Bot attacks. The model became more conservative (higher precision, lower recall).

**Experiment B (Wed+Fri → Thu):**
- **Recall**: 0.6732 (multi-day) vs 0.0702 (single-day Wed→Thu) — **IMPROVED by 9.6x**
- **F1**: 0.7228 (multi-day) vs 0.1120 (single-day) — **IMPROVED by 6.5x**
- **Precision**: 0.7803 (multi-day) vs 0.2762 (single-day) — **IMPROVED by 2.8x**
- **Observation**: Adding Friday's DDoS/PortScan/Bot data to Wednesday's DoS data dramatically improved detection of Thursday's web attacks and infiltration. This is the **only experiment where multi-day training substantially improved cross-day generalization**.

**Experiment C (Thu+Fri → Wed):**
- **Recall**: 0.0122 (multi-day) vs 0.9992 (baseline same-day) — still catastrophic
- **F1**: 0.0242 (multi-day) — no meaningful improvement over single-day Thu→Fri (0.0002)
- **Observation**: Training on web/infiltration + DDoS/PortScan/Bot does not help detect Wednesday's DoS attacks.

#### Key Findings

1. **Multi-day training helps only when the test day's attack types are represented in the training days.** Experiment B succeeded because Friday's data (DDoS, PortScan) shares some network-level characteristics with Thursday's web attacks that Wednesday's DoS-only data lacks.

2. **Adding irrelevant attack types can hurt performance.** Experiment A showed that adding Thursday's sparse attack data (0.48% attack rate) to Wednesday's rich DoS data (36% attack rate) diluted the model's ability to detect Friday's attacks.

3. **Class imbalance matters.** Thursday has only 0.48% attacks vs 36% on Wednesday and 41% on Friday. Training on imbalanced mixtures can bias the model toward the majority class patterns.

4. **ROC-AUC remains relatively high (0.93-0.97)** across all multi-day experiments, indicating the model ranks attacks above benign traffic, but the 0.5 decision threshold is poorly calibrated for unseen attack distributions.

5. **No multi-day combination achieves same-day performance.** Even the best multi-day experiment (B: Wed+Fri→Thu) has Recall=67% vs same-day 99.9%.

### Expected Hypothesis vs. Actual Observation

| Hypothesis | Actual Observation |
|------------|-------------------|
| "Training on multiple days will improve generalization to unseen days" | **Partially true** — only one of three experiments (Wed+Fri→Thu) showed substantial improvement. The other two showed degradation or no meaningful improvement. |
| "More diverse training data always helps" | **False** — adding Thursday's sparse, dissimilar attack data (Experiment A) hurt performance on Friday. |
| "Multi-day training can close the gap with same-day baseline" | **False** — even the best multi-day experiment (Recall=67%) falls far short of same-day baseline (Recall=99.9%). |

### Limitations

1. **Only 3 attack-type clusters available** — DoS, Web/Infiltration, DDoS/PortScan/Bot. More diverse training days would be needed for robust conclusions.

2. **Severe class imbalance in Thursday** (0.48% attacks) makes it a poor training day for supervised learning without resampling.

3. **No temporal adaptation** — model doesn't update with new data; this is a static generalization test.

4. **CICIDS2017 is dated (2017)** — modern attack patterns and encrypted traffic (TLS 1.3) differ significantly.

5. **Feature set is raw flow statistics** — no engineered features for day-invariant attack signatures.

6. **No threshold optimization** — all experiments use default 0.5 threshold; different operating points might change conclusions.

### Artifacts

- `results/multi_day_validation.json` — Full metrics per experiment with attack distributions
- `results/multi_day_validation.csv` — Comparison table with class distributions
- `results/multi_day/wed_thu_to_fri_confusion_matrix.png`
- `results/multi_day/wed_fri_to_thu_confusion_matrix.png`
- `results/multi_day/thu_fri_to_wed_confusion_matrix.png`

### How to Reproduce

```bash
# Run multi-day validation
python -m ml.multi_day_validation

# Results saved to results/multi_day_validation.json and results/multi_day/
```

### Dashboard Integration

The dashboard reads `results/multi_day_validation.json` and confusion matrix images from `results/multi_day/` for display on the Model Performance page.
**Test Attack Types:** DoS Hulk (231,073), DoS GoldenEye (10,293), DoS slowloris (5,796), DoS Slowhttptest (5,499), Heartbleed (11)
