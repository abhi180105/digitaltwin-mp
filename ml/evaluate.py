"""
ml/evaluate.py
==============
Standalone model evaluation script.

Loads saved models and test data, then produces a comprehensive
evaluation report (metrics + confusion matrix saved as PNG).

Usage
-----
    python -m ml.evaluate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.preprocess import load_processed

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run the training pipeline first:\n"
            "  python -m ml.train"
        )
    return joblib.load(path)


def evaluate_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "supervised",
) -> dict:
    """
    Evaluate a model and return a metrics dict.

    Parameters
    ----------
    name:
        Display name for the model.
    model:
        Trained scikit-learn estimator.
    X_test, y_test:
        Test split.
    model_type:
        'supervised' | 'unsupervised'

    Returns
    -------
    dict
        All computed metrics.
    """
    if model_type == "unsupervised":
        raw_pred = model.predict(X_test)
        y_pred = np.where(raw_pred == -1, 1, 0)
        y_prob = None
    else:
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

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

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall           : {rec:.4f}")
    print(f"  F1-Score         : {f1:.4f}")
    print(f"  False Positive %  : {fpr_val*100:.2f}%")
    if auc is not None:
        print(f"  ROC-AUC          : {auc:.4f}")
    print(f"\n  Confusion Matrix:\n    TN={tn}  FP={fp}\n    FN={fn}  TP={tp}")
    print()
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "ATTACK"]))

    metrics = {
        "model": name,
        "type": model_type,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr_val, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # Save confusion matrix plot
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NORMAL", "ATTACK"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").replace("/", "_")
    cm_path = RESULTS_DIR / f"{safe_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", cm_path)

    # Save ROC curve if possible
    if y_prob is not None and auc is not None:
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr_arr, tpr_arr, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name} — ROC Curve")
        ax.legend()
        plt.tight_layout()
        roc_path = RESULTS_DIR / f"{safe_name}_roc_curve.png"
        plt.savefig(roc_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("ROC curve saved: %s", roc_path)

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
    print("  Model Evaluation")
    print("=" * 60)

    try:
        _, X_test, _, y_test = load_processed()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        return

    all_metrics = []

    # Random Forest
    try:
        rf = _load_model("random_forest.joblib")
        m = evaluate_model("Random Forest", rf, X_test, y_test, model_type="supervised")
        all_metrics.append(m)
    except FileNotFoundError as exc:
        print(f"[SKIP] {exc}")

    # Isolation Forest
    try:
        iso = _load_model("isolation_forest.joblib")
        m = evaluate_model(
            "Isolation Forest (unsupervised)", iso, X_test, y_test, model_type="unsupervised"
        )
        all_metrics.append(m)
    except FileNotFoundError as exc:
        print(f"[SKIP] {exc}")

    if not all_metrics:
        print("No models found. Run: python -m ml.train")
        return

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "evaluation_results.json"
    with open(out, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nEvaluation results saved to: {out}")

    print("\n  Comparison Summary:")
    print(f"  {'Model':<38} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Type'}")
    print("  " + "-" * 70)
    for m in all_metrics:
        auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "  N/A "
        print(
            f"  {m['model']:<38} {m['accuracy']:>7.4f} {m['f1_score']:>7.4f} "
            f"{auc_str:>7} {m['type']}"
        )


if __name__ == "__main__":
    main()
