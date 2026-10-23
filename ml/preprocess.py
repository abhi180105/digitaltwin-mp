"""
ml/preprocess.py
================
Data preprocessing pipeline for the AI-Driven Digital Twin Framework.

Handles CICIDS2017 / UNSW-NB15 CSV datasets:
  1. Column name cleanup
  2. Missing / infinite value handling
  3. Duplicate removal
  4. Label normalisation (binary: NORMAL / ATTACK)
  5. Feature selection (drop non-numeric / irrelevant)
  6. Train/test split (no leakage)
  7. StandardScaler fitted only on train set
  8. Persist processed artifacts to data/processed/

Usage
-----
    python -m ml.preprocess
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns to always drop (IP/port metadata — not useful features, cause leakage)
COLUMNS_TO_DROP = [
    "flow_id", "source_ip", "source_port", "destination_ip",
    "destination_port", "timestamp", "src_ip", "dst_ip",
    "src_port", "dst_port", "protocol",
    # CICIDS2017 specific metadata
    "Flow ID", "Source IP", "Source Port", "Destination IP",
    "Destination Port", "Timestamp", "Protocol",
]

# CICIDS2017 label column candidates
LABEL_COLUMN_CANDIDATES = [
    "Label", "label", "Class", "class", "Attack", "attack",
    "category", "Category", "Label_cat",
]

# Labels that mean "benign / normal" (case-insensitive)
BENIGN_LABELS = {
    "benign", "normal", "legitimate", "0", "background",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_label_column(df: pd.DataFrame) -> str:
    """Identify the label column in the dataframe."""
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    # Fuzzy fallback: column whose name contains 'label'
    for col in df.columns:
        if "label" in col.lower():
            return col
    raise KeyError(
        "Could not identify label column. "
        f"Looked for: {LABEL_COLUMN_CANDIDATES}. "
        f"Columns found: {list(df.columns)}"
    )


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and lowercase all column names."""
    df.columns = [c.strip() for c in df.columns]
    return df


def _normalise_labels(series: pd.Series) -> pd.Series:
    """Map all benign/normal labels to 0, everything else to 1 (ATTACK)."""
    return series.apply(
        lambda x: 0 if str(x).strip().lower() in BENIGN_LABELS else 1
    )


def _remove_bad_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf with NaN, then drop remaining NaN rows."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with NaN/Inf values.", dropped)
    return df


def _select_numeric_features(df: pd.DataFrame, label_col: str) -> list[str]:
    """Return numeric feature columns (excluding the label)."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c != label_col]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_raw_dataset(filepath: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load a raw CSV dataset from disk.

    Parameters
    ----------
    filepath:
        Explicit path to the CSV.  If None, the first CSV found in
        ``data/raw/`` is used.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If no dataset is found.
    """
    if filepath:
        path = Path(filepath)
    else:
        csv_files = list(RAW_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                "Dataset not found.\n"
                "Place a CICIDS2017 or UNSW-NB15 CSV inside data/raw/\n"
                "See docs/dataset.md for download instructions."
            )
        path = csv_files[0]
        logger.info("Auto-detected dataset: %s", path)

    logger.info("Loading dataset: %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows × %d columns.", len(df), len(df.columns))
    return df


def preprocess(
    df: pd.DataFrame,
    sample_size: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, list[str]]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df:
        Raw dataset.
    sample_size:
        Optional row limit (useful for quick experiments).

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # 1. Clean column names
    df = _clean_column_names(df)

    # 2. Identify label column
    label_col = _find_label_column(df)
    logger.info("Label column: '%s'", label_col)

    # 3. Remove metadata columns
    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        logger.info("Dropped %d metadata columns.", len(drop_cols))

    # 4. Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info("Removed %d duplicate rows.", before - len(df))

    # 5. Handle bad values
    df = _remove_bad_values(df)

    # 6. Normalise labels (binary)
    label_col_clean = label_col.strip()
    df["label"] = _normalise_labels(df[label_col_clean])
    if label_col_clean != "label":
        df.drop(columns=[label_col_clean], inplace=True)

    # 7. Feature selection
    feature_cols = _select_numeric_features(df, "label")
    logger.info("Selected %d numeric features.", len(feature_cols))

    # 8. Optional sampling for quick runs
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
        logger.info("Sampled %d rows.", sample_size)

    X = df[feature_cols].astype(np.float32)
    y = df["label"]

    # 9. Train/test split — BEFORE scaling to prevent leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        "Train: %d rows | Test: %d rows | Attack ratio (train): %.2f%%",
        len(X_train), len(X_test), 100 * y_train.mean(),
    )

    # 10. Scale — fit on train only
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def save_processed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
    feature_names: list[str],
) -> None:
    """Persist processed splits and scaler to disk."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")

    logger.info("Processed data saved to: %s", PROCESSED_DIR)


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load previously processed train/test splits from disk."""
    required = [
        PROCESSED_DIR / "X_train.csv",
        PROCESSED_DIR / "X_test.csv",
        PROCESSED_DIR / "y_train.csv",
        PROCESSED_DIR / "y_test.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed data not found. Run preprocessing first:\n"
            "  python -m ml.preprocess\n"
            f"Missing: {missing}"
        )
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        df = load_raw_dataset()
        X_train, X_test, y_train, y_test, scaler, features = preprocess(df)
        save_processed(X_train, X_test, y_train, y_test, scaler, features)
        print("\nPreprocessing complete.")
        print(f"  Train samples : {len(X_train)}")
        print(f"  Test samples  : {len(X_test)}")
        print(f"  Features      : {len(features)}")
        print(f"  Attack % train: {100 * y_train.mean():.2f}%")
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")


if __name__ == "__main__":
    main()
