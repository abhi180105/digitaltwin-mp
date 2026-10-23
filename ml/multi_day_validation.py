"""
ml/multi_day_validation.py
==========================
Multi-Day Training + Held-Out-Day Testing for CICIDS2017.

Research question:
"Does training the Random Forest model on multiple CICIDS2017 capture days
improve generalization to a previously unseen day?"

Experiments:
A: Train = Wednesday + Thursday, Test = Friday
B: Train = Wednesday + Friday, Test = Thursday
C: Train = Thursday + Friday, Test = Wednesday

Usage
-----
    python -m ml.multi_day_validation
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
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
MULTI_DAY_DIR = RESULTS_DIR / "multi_day"
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
    day_files: dict[str, list[Path]] = {}

    for f in csv_files:
        match = DAY_PATTERN.search(f.name)
        if match:
            day = match.group(1).capitalize()
            if day not in day_files:
                day_files[day] = []
            day_files[day].append(f)
        else:
            logger.warning("Could not determine day from filename: %s", f.name)

    # Sort files within each day for reproducibility
    for day in day_files:
        day_files[day].sort()

    return day_files


def print_dataset_inventory(day_files: dict[str, list[Path]]) -> None:
    """Print a summary of available data per day with attack-type breakdown."""
    print("\n" + "=" * 80)
    print("  CICIDS2017 DATASET INVENTORY (with Attack Types)")
    print("=" * 80)

    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        files = day_files.get(day, [])
        if not files:
            print(f"{day:<12} {'0':<3} {'N/A':>12}")
            continue

        total_rows = 0
        all_labels = Counter()
        for f in files:
            try:
                df = pd.read_csv(f, usecols=[" Label"], low_memory=False)
                total_rows += len(df)
                labels = df[" Label"].astype(str).str.strip()
                all_labels.update(labels.value_counts().to_dict())
            except Exception as e:
                logger.warning("Could not read labels from %s: %s", f, e)

        benign_count = sum(v for k, v in all_labels.items() if k.lower() in BENIGN_LABELS)
        attack_count = total_rows - benign_count
        attack_pct = 100 * attack_count / total_rows if total_rows > 0 else 0

        print(f"{day:<12} {len(files):<3} {total_rows:>12,}  Benign={benign_count:,} Attack={attack_count:,} ({attack_pct:.2f}%)")
        attack_labels = {k: v for k, v in all_labels.items() if k.lower() not in BENIGN_LABELS}
        for atk, cnt in sorted(attack_labels.items(), key=lambda x: -x[1]):
            print(f"  {'':15} {atk}: {cnt:,}")

    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Preprocessing (fit on train, apply to test -- NO LEAKAGE)
# ---------------------------------------------------------------------------

def load_and_preprocess_days(
    filepaths: list[Path],
    fit_scaler: StandardScaler = None,
    fit_feature_cols: list[str] = None,
) -> tuple:
    """
    Load and preprocess all files for given days (training or test).

    Parameters
    ----------
    filepaths:
        List of CSV file paths for the day(s).
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
    dfs = []
    for f in filepaths:
        logger.info("Loading: %s", f.name)
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    logger.info("Combined %d rows for %d file(s)", len(df), len(filepaths))

    df = _clean_column_names(df)
    label_col = _find_label_column(df)
    logger.info("Label column: '%s'", label_col)

    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        logger.info("Dropped %d metadata columns", len(drop_cols))

    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info("Removed %d duplicate rows", before - len(df))

    df = _remove_bad_values(df)

    label_col_clean = label_col.strip()
    df["label"] = _normalise_labels(df[label_col_clean])
    if label_col_clean != "label":
        df.drop(columns=[label_col_clean], inplace=True)

    if fit_feature_cols is not None:
        feature_cols = fit_feature_cols
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Test day missing {len(missing)} features expected from train day: {missing[:10]}"
            )
        df = df[feature_cols + ["label"]]
    else:
        feature_cols = _select_numeric_features(df, "label")
        logger.info("Selected %d numeric features", len(feature_cols))

    X = df[feature_cols].astype(np.float32)
    y = df["label"]

    if fit_scaler is not None:
        scaler = fit_scaler
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=feature_cols,
            index=X.index,
        )
        logger.info("Applied pre-fitted scaler to test day")
    else:
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


def get_attack_type_distribution(filepaths: list[Path]) -> dict[str, int]:
    """Get the distribution of original attack labels for the given files."""
    all_labels = Counter()
    for f in filepaths:
        try:
            df = pd.read_csv(f, usecols=[" Label"], low_memory=False)
            labels = df[" Label"].astype(str).str.strip()
            all_labels.update(labels.value_counts().to_dict())
        except Exception as e:
            logger.warning("Could not read labels from %s: %s", f, e)
    return dict(all_labels)
# ---------------------------------------------------------------------------
# Multi-Day Experiment Runner
# ---------------------------------------------------------------------------

def run_multi_day_experiment(
    train_days: list[str],
    test_day: str,
    day_files: dict[str, list[Path]],
) -> dict:
    """
    Run a single multi-day experiment.

    Parameters
    ----------
    train_days: List of day names for training (e.g., ["Wednesday", "Thursday"])
    test_day: Day name for testing (e.g., "Friday")
    day_files: Dict mapping day -> list of file paths

    Returns
    -------
    dict with experiment results
    """
    train_label = "+".join([d[:3] for d in train_days])
    print("\n" + "#" * 60)
    print(f"#  EXPERIMENT: {train_label} -> {test_day[:3]}")
    print("#" * 60)

    train_files = []
    for d in train_days:
        train_files.extend(day_files[d])
    test_files = day_files[test_day]

    print(f"\n[TRAIN] Days: {', '.join(train_days)} ({len(train_files)} file(s))")
    for f in train_files:
        print(f"  - {f.name}")

    start_time = time.time()
    X_train, y_train, scaler, feature_cols = load_and_preprocess_days(train_files)
    train_time = time.time() - start_time

    train_benign = int((y_train == 0).sum())
    train_attack = int((y_train == 1).sum())
    train_attack_pct = 100 * train_attack / len(y_train) if len(y_train) > 0 else 0

    print(f"  Train samples: {len(X_train):,}")
    print(f"  Benign: {train_benign:,} | Attack: {train_attack:,} ({train_attack_pct:.2f}%)")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Training time: {train_time:.2f}s")

    train_attack_dist = get_attack_type_distribution(train_files)
    model = train_random_forest(X_train, y_train)

    print(f"\n[TEST] Day: {test_day} ({len(test_files)} file(s))")
    for f in test_files:
        print(f"  - {f.name}")

    X_test, y_test, _, _ = load_and_preprocess_days(
        test_files, fit_scaler=scaler, fit_feature_cols=feature_cols
    )

    test_benign = int((y_test == 0).sum())
    test_attack = int((y_test == 1).sum())
    test_attack_pct = 100 * test_attack / len(y_test) if len(y_test) > 0 else 0

    print(f"  Test samples: {len(X_test):,}")
    print(f"  Benign: {test_benign:,} | Attack: {test_attack:,} ({test_attack_pct:.2f}%)")

    test_attack_dist = get_attack_type_distribution(test_files)
    exp_name = f"{train_label} -> {test_day}"
    metrics = evaluate_model(exp_name, model, X_test, y_test)

    metrics.update({
        "train_days": train_days,
        "test_day": test_day,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features": len(feature_cols),
        "training_time_seconds": round(train_time, 2),
        "train_files": [f.name for f in train_files],
        "test_files": [f.name for f in test_files],
        "benign_train": train_benign,
        "attack_train": train_attack,
        "benign_test": test_benign,
        "attack_test": test_attack,
        "train_attack_distribution": train_attack_dist,
        "test_attack_distribution": test_attack_dist,
    })

    cm = np.array([
        [metrics["confusion_matrix"]["tn"], metrics["confusion_matrix"]["fp"]],
        [metrics["confusion_matrix"]["fn"], metrics["confusion_matrix"]["tp"]],
    ])
    safe_name = f"{'_'.join([d.lower()[:3] for d in train_days])}_to_{test_day.lower()[:3]}"
    cm_path = MULTI_DAY_DIR / f"{safe_name}_confusion_matrix.png"
    save_confusion_matrix(exp_name, cm, cm_path)
    metrics["confusion_matrix_path"] = str(cm_path)

    return metrics


def run_all_experiments(day_files: dict[str, list[Path]]) -> list[dict]:
    """Run all multi-day experiments based on available data."""
    experiments = [
        (["Wednesday", "Thursday"], "Friday"),
        (["Wednesday", "Friday"], "Thursday"),
        (["Thursday", "Friday"], "Wednesday"),
    ]

    available_days = set(day_files.keys())
    valid_experiments = []
    for train_days, test_day in experiments:
        if all(d in available_days for d in train_days) and test_day in available_days:
            valid_experiments.append((train_days, test_day))
        else:
            missing = [d for d in train_days if d not in available_days]
            if test_day not in available_days:
                missing.append(test_day)
            logger.warning("Skipping experiment (missing days: %s): %s -> %s", missing, train_days, test_day)

    results = []
    for train_days, test_day in valid_experiments:
        try:
            result = run_multi_day_experiment(train_days, test_day, day_files)
            results.append(result)
        except Exception as e:
            logger.error("Experiment %s -> %s failed: %s", train_days, test_day, e)
            results.append({
                "train_days": train_days,
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
    MULTI_DAY_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "multi_day_validation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to: %s", json_path)

    csv_rows = []
    for r in results:
        if "error" in r:
            csv_rows.append({
                "train_days": "+".join(r["train_days"]),
                "test_day": r["test_day"],
                "train_samples": None,
                "test_samples": None,
                "feature_count": None,
                "benign_train": None,
                "attack_train": None,
                "benign_test": None,
                "attack_test": None,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "false_positive_rate": None,
                "training_time_seconds": None,
                "error": r["error"],
            })
        else:
            csv_rows.append({
                "train_days": "+".join(r["train_days"]),
                "test_day": r["test_day"],
                "train_samples": r["train_samples"],
                "test_samples": r["test_samples"],
                "feature_count": r["features"],
                "benign_train": r["benign_train"],
                "attack_train": r["attack_train"],
                "benign_test": r["benign_test"],
                "attack_test": r["attack_test"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1_score"],
                "roc_auc": r["roc_auc"],
                "false_positive_rate": r["false_positive_rate"],
                "training_time_seconds": r["training_time_seconds"],
                "error": None,
            })

    df = pd.DataFrame(csv_rows)
    csv_path = RESULTS_DIR / "multi_day_validation.csv"
    df.to_csv(csv_path, index=False)
    logger.info("CSV results saved to: %s", csv_path)
def print_comparison(results: list[dict]) -> None:
    """Print comparison with existing baselines."""
    print("\n" + "=" * 100)
    print("  MULTI-DAY TRAINING vs. SINGLE-DAY CROSS-DAY vs. RANDOM SPLIT COMPARISON")
    print("=" * 100)

    print("\n  BASELINE: Random 80/20 Split (Wednesday only)")
    print("  " + "-" * 90)
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'Accuracy':<25} {0.9991:>10.4f}")
    print(f"  {'Precision':<25} {0.9981:>10.4f}")
    print(f"  {'Recall':<25} {0.9992:>10.4f}")
    print(f"  {'F1-Score':<25} {0.9987:>10.4f}")
    print(f"  {'ROC-AUC':<25} {0.9999:>10.4f}")
    print(f"  {'False Positive Rate':<25} {0.0009:>10.4f}")

    print("\n  SINGLE-DAY CROSS-DAY VALIDATION (from ml.cross_day_validation)")
    print("  " + "-" * 90)
    single_day_results = {
        "Wed->Thu": {"accuracy": 0.9932, "precision": 0.2762, "recall": 0.0702, "f1_score": 0.1120, "roc_auc": 0.9763, "false_positive_rate": 0.0011},
        "Wed->Fri": {"accuracy": 0.8512, "precision": 0.9956, "recall": 0.4277, "f1_score": 0.5983, "roc_auc": 0.9433, "false_positive_rate": 0.0007},
        "Thu->Fri": {"accuracy": 0.7409, "precision": 0.3333, "recall": 0.0001, "f1_score": 0.0002, "roc_auc": 0.7747, "false_positive_rate": 0.0001},
    }
    header = f"  {'Train->Test':<15} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8} {'FPR':>8}"
    print(header)
    print("  " + "-" * len(header))
    for name, m in single_day_results.items():
        print(f"  {name:<15} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1_score']:>8.4f} {m['roc_auc']:>8.4f} {m['false_positive_rate']:>8.4f}")

    print("\n  MULTI-DAY VALIDATION RESULTS")
    print("  " + "-" * 90)
    header = f"  {'Train->Test':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8} {'FPR':>8}"
    print(header)
    print("  " + "-" * len(header))

    for r in results:
        if "error" in r:
            train_short = "+".join([d[:3] for d in r["train_days"]])
            print(f"  {train_short}->{r['test_day'][:3]:<12} {'ERROR':>8}  {r['error']}")
        else:
            train_short = "+".join([d[:3] for d in r["train_days"]])
            test_short = r["test_day"][:3]
            auc_val = r["roc_auc"] if r["roc_auc"] is not None else float('nan')
            print(
                f"  {train_short}->{test_short:<12} "
                f"{r['accuracy']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} "
                f"{r['f1_score']:>8.4f} {auc_val:>8.4f} "
                f"{r['false_positive_rate']:>8.4f}"
            )

    print("\n  DETAILED COMPARISON TABLE")
    print("  " + "-" * 90)
    print(f"  {'Experiment':<25} {'Train Data':<25} {'Test Data':<12} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("  " + "-" * 90)
    print(f"  {'Baseline (Rand 80/20)':<25} {'Wednesday':<25} {'Wednesday':<12} {0.9991:>7.4f} {0.9981:>7.4f} {0.9992:>7.4f} {0.9987:>7.4f} {0.9999:>7.4f}")
    print(f"  {'Single-day':<25} {'Wednesday':<25} {'Thursday':<12} {0.9932:>7.4f} {0.2762:>7.4f} {0.0702:>7.4f} {0.1120:>7.4f} {0.9763:>7.4f}")
    print(f"  {'Single-day':<25} {'Wednesday':<25} {'Friday':<12} {0.8512:>7.4f} {0.9956:>7.4f} {0.4277:>7.4f} {0.5983:>7.4f} {0.9433:>7.4f}")
    print(f"  {'Single-day':<25} {'Thursday':<25} {'Friday':<12} {0.7409:>7.4f} {0.3333:>7.4f} {0.0001:>7.4f} {0.0002:>7.4f} {0.7747:>7.4f}")
    for r in results:
        if "error" not in r:
            train_short = "+".join(r["train_days"])
            print(f"  {'Multi-day':<25} {train_short:<25} {r['test_day']:<12} {r['accuracy']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1_score']:>7.4f} {r['roc_auc'] if r['roc_auc'] is not None else float('nan'):>7.4f}")

    print("\n  ANALYSIS")
    print("  " + "-" * 90)
    successful = [r for r in results if "error" not in r]
    if successful:
        for r in successful:
            train_short = "+".join([d[:3] for d in r["train_days"]])
            test_short = r["test_day"][:3]

            print(f"  {r['train_days'][0][:3]}+{r['train_days'][1][:3]} -> {r['test_day'][:3]}:")
            print(f"    Recall: {r['recall']:.4f} | F1: {r['f1_score']:.4f} | Precision: {r['precision']:.4f} | FPR: {r['false_positive_rate']:.4f} | AUC: {r['roc_auc'] if r['roc_auc'] is not None else float('nan'):.4f}")
            print(f"    Train: Benign={r['benign_train']:,} Attack={r['attack_train']:,} ({100*r['attack_train']/r['train_samples']:.2f}%)")
            print(f"    Test:  Benign={r['benign_test']:,} Attack={r['attack_test']:,} ({100*r['attack_test']/r['test_samples']:.2f}%)")

            test_atk_dist = r.get("test_attack_distribution", {})
            if test_atk_dist:
                atk_items = [(k, v) for k, v in test_atk_dist.items() if k.lower() not in BENIGN_LABELS]
                if atk_items:
                    print(f"    Test Attack Types: {', '.join([f'{k}({v})' for k, v in sorted(atk_items, key=lambda x: -x[1])])}")

    print("=" * 100 + "\n")
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("  Multi-Day Training + Held-Out-Day Validation")
    print("=" * 60)

    day_files = discover_day_files()
    print_dataset_inventory(day_files)
    results = run_all_experiments(day_files)
    save_results(results)
    print_comparison(results)

    print("Done. Results saved to:")
    print(f"  - {RESULTS_DIR / 'multi_day_validation.json'}")
    print(f"  - {RESULTS_DIR / 'multi_day_validation.csv'}")
    print(f"  - {MULTI_DAY_DIR}/*.png")


if __name__ == "__main__":
    main()

    print("=" * 80 + "\n")