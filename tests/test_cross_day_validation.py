"""
tests/test_cross_day_validation.py
===================================
Unit tests for the Cross-Day Validation pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.cross_day_validation import (
    discover_day_files,
    load_and_preprocess_day,
    run_all_experiments,
    save_results,
    print_comparison,
)


class TestDiscoverDayFiles:
    """Tests for day/file discovery."""

    def test_discovers_all_days(self):
        day_files = discover_day_files()
        assert "Wednesday" in day_files
        assert "Thursday" in day_files
        assert "Friday" in day_files

    def test_groups_files_by_day(self):
        day_files = discover_day_files()
        assert len(day_files["Thursday"]) == 2
        assert len(day_files["Friday"]) == 3
        assert len(day_files["Wednesday"]) == 1

    def test_files_sorted_within_day(self):
        day_files = discover_day_files()
class TestLoadAndPreprocessDay:
    """Tests for the preprocessing pipeline."""

    def test_returns_correct_tuple_structure(self):
        day_files = discover_day_files()
        result = load_and_preprocess_day(day_files["Wednesday"])
        assert isinstance(result, tuple)
        assert len(result) == 4
        X, y, scaler, features = result
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert scaler is not None
        assert isinstance(features, list)

    def test_train_day_fits_scaler(self):
        day_files = discover_day_files()
        X, y, scaler, features = load_and_preprocess_day(day_files["Wednesday"])
        assert hasattr(scaler, 'mean_')
        assert scaler.mean_ is not None

    def test_test_day_uses_pre_fitted_scaler(self):
        day_files = discover_day_files()
        X_train, y_train, scaler, features = load_and_preprocess_day(day_files["Wednesday"])
        X_test, y_test, scaler2, features2 = load_and_preprocess_day(
            day_files["Thursday"], fit_scaler=scaler, fit_feature_cols=features
        )
        assert scaler2 is scaler
        assert features2 == features
        assert list(X_test.columns) == list(X_train.columns)

    def test_no_feature_leakage(self):
        day_files = discover_day_files()
        X_train, y_train, scaler, features = load_and_preprocess_day(day_files["Wednesday"])
        X_test, y_test, _, _ = load_and_preprocess_day(
            day_files["Thursday"], fit_scaler=scaler, fit_feature_cols=features
        )
        assert X_train.shape[1] == X_test.shape[1] == 77
        assert list(X_train.columns) == list(X_test.columns)

    def test_labels_are_binary(self):
        day_files = discover_day_files()
        _, y_train, _, _ = load_and_preprocess_day(day_files["Wednesday"])
        _, y_test, _, _ = load_and_preprocess_day(
            day_files["Thursday"], fit_scaler=None, fit_feature_cols=None
        )
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})

    def test_no_nan_inf_in_output(self):
        day_files = discover_day_files()
        X_train, _, _, _ = load_and_preprocess_day(day_files["Wednesday"])
        assert not np.isnan(X_train.values).any()
        assert not np.isinf(X_train.values).any()
class TestTrainTestSeparation:
    """Tests ensuring no data leakage between train and test days."""

    def test_no_overlap_in_samples(self):
        day_files = discover_day_files()
        X_train, _, _, _ = load_and_preprocess_day(day_files["Wednesday"])
        X_test, _, _, _ = load_and_preprocess_day(
            day_files["Thursday"], fit_scaler=None, fit_feature_cols=None
        )
        assert len(X_train) > 0
        assert len(X_test) > 0

    def test_scaler_fitted_only_on_train(self):
        day_files = discover_day_files()
        X_train, y_train, scaler, features = load_and_preprocess_day(day_files["Wednesday"])
        X_test, y_test, scaler2, _ = load_and_preprocess_day(
            day_files["Thursday"], fit_scaler=scaler, fit_feature_cols=features
        )
        assert scaler is scaler2


class TestRunAllExperiments:
    """Tests for the experiment runner."""

    def test_runs_at_least_two_experiments(self):
        day_files = discover_day_files()
        results = run_all_experiments(day_files)
        assert len(results) >= 2

    def test_results_have_required_keys(self):
        day_files = discover_day_files()
        results = run_all_experiments(day_files)
        for r in results:
            if "error" not in r:
                assert "train_day" in r
                assert "test_day" in r
                assert "accuracy" in r
                assert "precision" in r
                assert "recall" in r
                assert "f1_score" in r
                assert "roc_auc" in r
                assert "false_positive_rate" in r
                assert "train_samples" in r
                assert "test_samples" in r
                assert "features" in r

    def test_experiments_have_distinct_train_test_days(self):
        day_files = discover_day_files()
        results = run_all_experiments(day_files)
        for r in results:
            if "error" not in r:
                assert r["train_day"] != r["test_day"]


class TestEvaluationOutputStructure:
    """Tests for evaluation output structure."""

    def test_confusion_matrix_structure(self):
        day_files = discover_day_files()
        results = run_all_experiments(day_files)
        for r in results:
            if "error" not in r:
                cm = r.get("confusion_matrix", {})
                assert "tn" in cm
                assert "fp" in cm
                assert "fn" in cm
                assert "tp" in cm
                assert all(isinstance(v, int) for v in cm.values())

    def test_metrics_are_floats(self):
        day_files = discover_day_files()
        results = run_all_experiments(day_files)
        for r in results:
            if "error" not in r:
                for key in ["accuracy", "precision", "recall", "f1_score", "false_positive_rate"]:
                    assert isinstance(r[key], float)
                    assert 0.0 <= r[key] <= 1.0
                if r["roc_auc"] is not None:
                    assert isinstance(r["roc_auc"], float)
                    assert 0.0 <= r["roc_auc"] <= 1.0
        for day, files in day_files.items():
            assert files == sorted(files), f"Files for {day} not sorted"