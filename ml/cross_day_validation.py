"""
ml/cross_day_validation.py
==========================
Cross-Day Generalization Validation for CICIDS2017.

Tests how well a model trained on one day of network traffic generalizes
to previously unseen traffic from another day.

Usage
-----
    python -m ml.cross_day_validation
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Reuse preprocessing helpers from ml.preprocess
from ml.preprocess import (
    _clean_column_names,
    _find_label_column,
    _normalise_labels,
    _remove_bad_values,
    _select_numeric_features,
    COLUMNS_TO_DROP,
    LABEL_COLUMN_CANDIDATES,
    BENIGN_LABELS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
CROSS_DAY_DIR = RESULTS_DIR / "cross_day"
MODELS_DIR = Path("models")

RANDOM_STATE = 42
N_ESTIMATORS = 100

# ---------------------------------------------------------------------------
# Day/File Discovery
# ---------------------------------------------------------------------------

DAY_PATTERN = re.compile(
    r"(Monday|Tuesday|Wednesday|Thursday|Friday)", re.IGNORECASE
)


def discover_day_files() -> dict[str, list[Path]]:
    """
    Discover CICIDS2017 CSV files and group them by calendar day.

    Returns
    -------
    dict[str, list[Path]]
        Mapping: day_name -> list of file paths for that day.
    """
    csv_files = list(RAW_DIR.glob("*.csv"))
    day_files: dict[str, list[Path]] = defaultdict(list)

    for f in csv_files:
        match = DAY_PATTERN.search(f.name)
        if match:
            day = match.group(1).capitalize()
            day_files[day].append(f)
        else:
            logger.warning("Could not determine day from filename: %s", f.name)

    # Sort files within each day for reproducibility
    for day in day_files:
        day_files[day].sort()

    return dict(day_files)


def print_dataset_inventory(day_files: dict[str, list[Path]]) -> None:
    """Print a summary of available data per day."""
    print("\n" + "=" * 80)
    print("  CICIDS2017 DATASET INVENTORY")
    print("=" * 80)
    print(f"{'Day':<12} {'Files':<3} {'Total Rows':>12} {'Attack Labels'}")
    print("-" * 80)

    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        files = day_files.get(day, [])
        if not files:
            print(f"{day:<12} {'0':<3} {'N/A':>12}")
            continue

        total_rows = 0
        all_labels = set()
        for f in files:
            try:
                df = pd.read_csv(f, usecols=[" Label"], low_memory=False)
                total_rows += len(df)
                labels = df[" Label"].astype(str).str.strip().unique()
                all_labels.update(labels)
            except Exception as e:
                logger.warning("Could not read labels from %s: %s", f, e)

        attack_labels = sorted([l for l in all_labels if l.lower() not in BENIGN_LABELS])
        print(f"{day:<12} {len(files):<3} {total_rows:>12,}  {', '.join(attack_labels[:5])}{'...' if len(attack_labels) > 5 else ''}")

    print("=" * 80 + "\n")
# ---------------------------------------------------------------------------
# Preprocessing (fit on train, apply to test -- NO LEAKAGE)
# ---------------------------------------------------------------------------

def load_and_preprocess_day(
    filepaths: list[Path],
    fit_scaler: StandardScaler = None,
    fit_feature_cols: list[str] = None,
) -> tuple:
    """
    Load and preprocess all files for a given day.

    Parameters
    ----------
    filepaths:
        List of CSV file paths for the day.
    fit_scaler:
        If provided, use this pre-fitted scaler (for test day).
        If None, fit a new scaler (for train day).
    fit_feature_cols:
        If provided, use these feature columns (for test day).
        If None, derive from data (for train day).

    Returns
    -------
    X_scaled, y, scaler, feature_cols
    """
    # Load and concatenate all files for this day
    dfs = []
    for f in filepaths:
        logger.info("Loading: %s", f.name)
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    logger.info("Combined %d rows for %d file(s)", len(df), len(filepaths))

    # 1. Clean column names
    df = _clean_column_names(df)

    # 2. Identify label column
    label_col = _find_label_column(df)
    logger.info("Label column: '%s'", label_col)

    # 3. Remove metadata columns
    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        logger.info("Dropped %d metadata columns", len(drop_cols))

    # 4. Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info("Removed %d duplicate rows", before - len(df))

    # 5. Handle bad values (inf/nan)
    df = _remove_bad_values(df)

    # 6. Normalise labels (binary: 0=BENIGN, 1=ATTACK)
    label_col_clean = label_col.strip()
    df["label"] = _normalise_labels(df[label_col_clean])
    if label_col_clean != "label":
        df.drop(columns=[label_col_clean], inplace=True)

    # 7. Feature selection
    if fit_feature_cols is not None:
        # Test day: ensure we have all required features
        feature_cols = fit_feature_cols
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Test day missing {len(missing)} features expected from train day: {missing[:10]}"
            )
        # Reorder to match train day
        df = df[feature_cols + ["label"]]
    else:
        # Train day: select numeric features
        feature_cols = _select_numeric_features(df, "label")
        logger.info("Selected %d numeric features", len(feature_cols))

    X = df[feature_cols].astype(np.float32)
    y = df["label"]

    # 8. Scaling
    if fit_scaler is not None:
        # Test day: use pre-fitted scaler
        scaler = fit_scaler
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=feature_cols,
            index=X.index,
        )
        logger.info("Applied pre-fitted scaler to test day")
    else:
        # Train day: fit new scaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=feature_cols,
            index=X.index,
        )
        logger.info("Fitted new scaler on train day")

    return X_scaled, y, scaler, feature_cols

# ---------------------------------------------------------------------------
# Model Training & Evaluation
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train Random Forest with standard configuration."""
    logger.info("Training Random Forest (n_estimators=%d)...", N_ESTIMATORS)
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(
    name: str,
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    auc = None
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            auc = None

    print("\n" + "=" * 60)
    print(f"  {name}")
    print("=" * 60)
    print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall           : {rec:.4f}")
    print(f"  F1-Score         : {f1:.4f}")
    print(f"  False Positive % : {fpr_val*100:.2f}%")
    if auc is not None:
        print(f"  ROC-AUC          : {auc:.4f}")
    print("\n  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print()
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "ATTACK"]))

    metrics = {
        "model": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr_val, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    return metrics


def save_confusion_matrix(
    name: str,
    cm: np.ndarray,
    output_path: Path,
) -> None:
    """Save confusion matrix as PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NORMAL", "ATTACK"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name} -- Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", output_path)
# ---------------------------------------------------------------------------
# Cross-Day Experiment Runner
# ---------------------------------------------------------------------------

def run_cross_day_experiment(
    train_day: str,
    test_day: str,
    day_files: dict[str, list[Path]],
) -> dict:
    """
    Run a single cross-day experiment.

    Parameters
    ----------
    train_day: Day name for training (e.g., "Wednesday")
    test_day: Day name for testing (e.g., "Thursday")
    day_files: Dict mapping day -> list of file paths

    Returns
    -------
    dict with experiment results
    """
    print("\n" + "#" * 60)
    print(f"#  EXPERIMENT: {train_day} -> {test_day}")
    print("#" * 60)

    train_files = day_files[train_day]
    test_files = day_files[test_day]

    # --- TRAIN DAY ---
    print(f"\n[TRAIN] {train_day}: {len(train_files)} file(s)")
    X_train, y_train, scaler, feature_cols = load_and_preprocess_day(train_files)
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Attack ratio: {100 * y_train.mean():.2f}%")
    print(f"  Features: {len(feature_cols)}")

    # Train model
    model = train_random_forest(X_train, y_train)

    # --- TEST DAY ---
    print(f"\n[TEST] {test_day}: {len(test_files)} file(s)")
    X_test, y_test, _, _ = load_and_preprocess_day(
        test_files, fit_scaler=scaler, fit_feature_cols=feature_cols
    )
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Attack ratio: {100 * y_test.mean():.2f}%")

    # Evaluate
    exp_name = f"{train_day} -> {test_day}"
    metrics = evaluate_model(exp_name, model, X_test, y_test)

    # Add experiment metadata
    metrics.update({
        "train_day": train_day,
        "test_day": test_day,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features": len(feature_cols),
        "train_files": [f.name for f in train_files],
        "test_files": [f.name for f in test_files],
    })

    # Save confusion matrix
    cm = np.array([
        [metrics["confusion_matrix"]["tn"], metrics["confusion_matrix"]["fp"]],
        [metrics["confusion_matrix"]["fn"], metrics["confusion_matrix"]["tp"]],
    ])
    safe_name = f"{train_day.lower()}_to_{test_day.lower()}"
    cm_path = CROSS_DAY_DIR / f"{safe_name}_confusion_matrix.png"
    save_confusion_matrix(exp_name, cm, cm_path)
    metrics["confusion_matrix_path"] = str(cm_path)

    return metrics


def run_all_experiments(day_files: dict[str, list[Path]]) -> list[dict]:
    """Run all cross-day experiments based on available data."""
    available_days = [d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"] if d in day_files]

    experiments = []

    # Experiment A: Wednesday -> Thursday
    if "Wednesday" in day_files and "Thursday" in day_files:
        experiments.append(("Wednesday", "Thursday"))

    # Experiment B: Wednesday -> Friday
    if "Wednesday" in day_files and "Friday" in day_files:
        experiments.append(("Wednesday", "Friday"))

    # Experiment C: Thursday -> Friday (optional)
    if "Thursday" in day_files and "Friday" in day_files:
        experiments.append(("Thursday", "Friday"))

    results = []
    for train_day, test_day in experiments:
        try:
            result = run_cross_day_experiment(train_day, test_day, day_files)
            results.append(result)
        except Exception as e:
            logger.error("Experiment %s -> %s failed: %s", train_day, test_day, e)
            results.append({
                "train_day": train_day,
                "test_day": test_day,
                "error": str(e),
            })

    return results
# ---------------------------------------------------------------------------
# Results Storage & Comparison
# ---------------------------------------------------------------------------

def save_results(results: list[dict]) -> None:
    """Save results to JSON and CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CROSS_DAY_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = RESULTS_DIR / "cross_day_validation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to: %s", json_path)

    # CSV - flat table for comparison
    csv_rows = []
    for r in results:
        if "error" in r:
            csv_rows.append({
                "train_day": r["train_day"],
                "test_day": r["test_day"],
                "train_samples": None,
                "test_samples": None,
                "features": None,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "false_positive_rate": None,
                "error": r["error"],
            })
        else:
            csv_rows.append({
                "train_day": r["train_day"],
                "test_day": r["test_day"],
                "train_samples": r["train_samples"],
                "test_samples": r["test_samples"],
                "features": r["features"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1_score"],
                "roc_auc": r["roc_auc"],
                "false_positive_rate": r["false_positive_rate"],
                "error": None,
            })

    df = pd.DataFrame(csv_rows)
    csv_path = RESULTS_DIR / "cross_day_validation.csv"
    df.to_csv(csv_path, index=False)
    logger.info("CSV results saved to: %s", csv_path)


def print_comparison(results: list[dict]) -> None:
    """Print comparison with current (random split) result."""
    print("\n" + "=" * 80)
    print("  CROSS-DAY VALIDATION vs. RANDOM SPLIT COMPARISON")
    print("=" * 80)

    # Current baseline (from Wednesday 80/20 random split)
    print("\n  BASELINE: Random 80/20 Split (Wednesday only)")
    print("  " + "-" * 70)
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'Accuracy':<25} {0.9991:>10.4f}")
    print(f"  {'Precision':<25} {0.9981:>10.4f}")
    print(f"  {'Recall':<25} {0.9992:>10.4f}")
    print(f"  {'F1-Score':<25} {0.9987:>10.4f}")
    print(f"  {'ROC-AUC':<25} {0.9999:>10.4f}")
    print(f"  {'False Positive Rate':<25} {0.0009:>10.4f}")

    # Cross-day results
    print("\n  CROSS-DAY VALIDATION RESULTS")
    print("  " + "-" * 70)
    header = f"  {'Train->Test':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8} {'FPR':>8}"
    print(header)
    print("  " + "-" * len(header))

    for r in results:
        if "error" in r:
            print(f"  {r['train_day'][:3]}->{r['test_day'][:3]:<16} {'ERROR':>8}  {r['error']}")
        else:
            train_short = r["train_day"][:3]
            test_short = r["test_day"][:3]
            auc_val = r["roc_auc"] if r["roc_auc"] is not None else float('nan')
            print(
                f"  {train_short}->{test_short:<16} "
                f"{r['accuracy']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} "
                f"{r['f1_score']:>8.4f} {auc_val:>8.4f} "
                f"{r['false_positive_rate']:>8.4f}"
            )

    # Analysis
    print("\n  ANALYSIS")
    print("  " + "-" * 70)
    successful = [r for r in results if "error" not in r]
    if successful:
        baseline_acc = 0.9991
        baseline_f1 = 0.9987
        for r in successful:
            acc_diff = r["accuracy"] - baseline_acc
            f1_diff = r["f1_score"] - baseline_f1
            print(f"  {r['train_day']}->{r['test_day']}:")
            print(f"    Accuracy Delta: {acc_diff:+.4f} ({'similar' if abs(acc_diff) < 0.01 else 'decreased' if acc_diff < 0 else 'increased'})")
            print(f"    F1 Delta:       {f1_diff:+.4f} ({'similar' if abs(f1_diff) < 0.01 else 'decreased' if f1_diff < 0 else 'increased'})")

            if r["false_positive_rate"] > 0.01:
                print(f"    Warning: Higher FPR ({r['false_positive_rate']:.4f}) suggests distribution shift")
            if r["recall"] < 0.95:
                print(f"    Warning: Lower Recall ({r['recall']:.4f}) suggests unseen attack patterns")

    print("=" * 80 + "\n")
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("  Cross-Day Generalization Validation")
    print("=" * 60)

    # 1. Discover files
    day_files = discover_day_files()
    print_dataset_inventory(day_files)

    # 2. Run experiments
    results = run_all_experiments(day_files)

    # 3. Save results
    save_results(results)

    # 4. Print comparison
    print_comparison(results)

    # 5. Summary
    print("Done. Results saved to:")
    print(f"  - {RESULTS_DIR / 'cross_day_validation.json'}")
    print(f"  - {RESULTS_DIR / 'cross_day_validation.csv'}")
    print(f"  - {CROSS_DAY_DIR}/*.png")


if __name__ == "__main__":
    main()