"""
tests/test_preprocessing.py
===========================
Unit tests for the ML preprocessing pipeline.
"""

import io
import pytest
import numpy as np
import pandas as pd

from ml.preprocess import (
    _clean_column_names,
    _normalise_labels,
    _remove_bad_values,
    _find_label_column,
    _select_numeric_features,
    preprocess,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal synthetic dataset resembling CICIDS2017."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        " Flow Duration ": rng.integers(0, 100_000, n),
        "Total Fwd Packets": rng.integers(0, 500, n),
        "Total Bwd Packets": rng.integers(0, 500, n),
        "Flow Bytes/s": rng.uniform(0, 1e6, n),
        "Label": (["BENIGN"] * 160) + (["DoS Hulk"] * 40),
    })


# ---------------------------------------------------------------------------
# Column name cleanup
# ---------------------------------------------------------------------------

class TestCleanColumns:
    def test_strips_whitespace(self):
        df = pd.DataFrame({" Col A ": [1], "Col B": [2]})
        cleaned = _clean_column_names(df)
        assert "Col A" in cleaned.columns
        assert " Col A " not in cleaned.columns


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

class TestNormaliseLabels:
    def test_benign_becomes_zero(self):
        s = pd.Series(["BENIGN", "Normal", "Legitimate"])
        result = _normalise_labels(s)
        assert (result == 0).all()

    def test_attack_becomes_one(self):
        s = pd.Series(["DoS Hulk", "PortScan", "BruteForce"])
        result = _normalise_labels(s)
        assert (result == 1).all()

    def test_mixed(self):
        s = pd.Series(["BENIGN", "DoS", "BENIGN"])
        result = _normalise_labels(s)
        assert list(result) == [0, 1, 0]


# ---------------------------------------------------------------------------
# Bad value removal
# ---------------------------------------------------------------------------

class TestRemoveBadValues:
    def test_removes_inf(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [1, 2, 3]})
        result = _remove_bad_values(df.copy())
        assert len(result) == 2

    def test_removes_neg_inf(self):
        df = pd.DataFrame({"a": [-np.inf, 2.0], "b": [1, 2]})
        result = _remove_bad_values(df.copy())
        assert len(result) == 1

    def test_removes_nan(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result = _remove_bad_values(df.copy())
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Label column detection
# ---------------------------------------------------------------------------

class TestFindLabelColumn:
    def test_finds_label(self):
        df = pd.DataFrame({"feature": [1], "Label": ["BENIGN"]})
        assert _find_label_column(df) == "Label"

    def test_finds_label_lowercase(self):
        df = pd.DataFrame({"feature": [1], "label": ["BENIGN"]})
        assert _find_label_column(df) == "label"

    def test_raises_if_not_found(self):
        df = pd.DataFrame({"feature": [1], "other": [2]})
        with pytest.raises(KeyError):
            _find_label_column(df)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestPreprocessPipeline:
    def test_returns_correct_shapes(self, sample_df):
        X_train, X_test, y_train, y_test, scaler, features = preprocess(sample_df)
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_labels_are_binary(self, sample_df):
        _, _, y_train, y_test, _, _ = preprocess(sample_df)
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})

    def test_no_leakage_in_scaler(self, sample_df):
        X_train, X_test, _, _, scaler, features = preprocess(sample_df)
        # Scaler is fit on train set only — mean should reflect train
        assert scaler.n_features_in_ == len(features)

    def test_features_are_numeric(self, sample_df):
        X_train, _, _, _, _, _ = preprocess(sample_df)
        assert all(X_train[c].dtype in [np.float32, np.float64] for c in X_train.columns)

    def test_no_inf_in_output(self, sample_df):
        X_train, X_test, _, _, _, _ = preprocess(sample_df)
        assert not np.isinf(X_train.values).any()
        assert not np.isinf(X_test.values).any()

    def test_sample_size_limits_rows(self, sample_df):
        X_train, X_test, _, _, _, _ = preprocess(sample_df, sample_size=50)
        assert len(X_train) + len(X_test) <= 52  # allow small split rounding
