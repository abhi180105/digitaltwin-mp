"""
ml/train.py
===========
Model training pipeline for the AI-Driven Digital Twin Framework.

Trains two models and saves them:
  1. Random Forest (supervised)
  2. Isolation Forest (unsupervised anomaly detection)

Usage
-----
    python -m ml.train
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.preprocess import load_processed

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quick_eval(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute and log key classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = float("nan")

    logger.info(
        "%s | acc=%.4f | prec=%.4f | rec=%.4f | f1=%.4f | auc=%.4f",
        name, acc, prec, rec, f1, auc,
    )
    print(f"\n{name} — Classification Report")
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "ATTACK"]))

    return {
        "model": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
    }


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
) -> dict:
    """
    Train and evaluate a Random Forest classifier.

    Returns
    -------
    dict
        Evaluation metrics from the test set.
    """
    logger.info("Training Random Forest (n_estimators=%d)...", n_estimators)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    metrics = _quick_eval("RandomForest", y_test, y_pred)
    metrics["n_estimators"] = n_estimators
    metrics["type"] = "supervised"

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    logger.info("Random Forest saved to models/random_forest.joblib")

    return metrics


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

def train_isolation_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    contamination: float = 0.1,
) -> dict:
    """
    Train and evaluate an Isolation Forest (unsupervised anomaly detection).

    Note: Isolation Forest is trained WITHOUT labels.
    Predictions are mapped: -1 (anomaly) → 1 (ATTACK), 1 (inlier) → 0 (NORMAL).
    This is an *unsupervised* baseline; its metrics cannot be directly compared
    to the supervised Random Forest.

    Returns
    -------
    dict
        Evaluation metrics from the test set.
    """
    logger.info("Training Isolation Forest (contamination=%.2f)...", contamination)
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    # Unsupervised — trained on full training features (no labels used)
    iso.fit(X_train)

    raw_pred = iso.predict(X_test)
    # Map: -1 → 1 (anomaly/attack), 1 → 0 (normal)
    y_pred = np.where(raw_pred == -1, 1, 0)

    metrics = _quick_eval("IsolationForest (unsupervised)", y_test, y_pred)
    metrics["contamination"] = contamination
    metrics["type"] = "unsupervised"
    metrics["note"] = (
        "Isolation Forest is unsupervised — labels are not used during training. "
        "Metrics are evaluated post-hoc for comparison purposes only."
    )

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, MODELS_DIR / "isolation_forest.joblib")
    logger.info("Isolation Forest saved to models/isolation_forest.joblib")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=" * 60)
    print("  ML Training Pipeline")
    print("=" * 60)

    try:
        X_train, X_test, y_train, y_test = load_processed()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        return

    logger.info(
        "Loaded: X_train=%s, X_test=%s", X_train.shape, X_test.shape
    )

    all_metrics = []

    # --- Random Forest ---
    rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    all_metrics.append(rf_metrics)

    # --- Isolation Forest ---
    iso_metrics = train_isolation_forest(X_train, X_test, y_test)
    all_metrics.append(iso_metrics)

    # Persist results
    results_path = Path("models/training_results.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Training results saved to: %s", results_path)

    print("\nSummary:")
    for m in all_metrics:
        print(
            f"  {m['model']:<35} acc={m['accuracy']:.4f}  "
            f"f1={m['f1_score']:.4f}  [{m['type']}]"
        )

    print("\nTraining complete. Models saved to models/")


if __name__ == "__main__":
    main()
