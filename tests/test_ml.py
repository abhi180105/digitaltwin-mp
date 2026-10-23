"""
tests/test_ml.py
================
Unit tests for the ML training and evaluation modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.train import train_random_forest, train_isolation_forest, _quick_eval
from ml.evaluate import evaluate_model
from ml.predict import FlowPredictor


class TestTrainHelpers:
    def test_quick_eval_returns_dict(self):
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        result = _quick_eval("TestModel", y_true, y_pred)
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "f1_score" in result
        assert result["model"] == "TestModel"


class TestTrainRandomForest:
    def test_train_random_forest_returns_metrics(self):
        # Create minimal synthetic data
        X_train = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(5)])
        y_train = pd.Series([0] * 30 + [1] * 20)
        X_test = pd.DataFrame(np.random.rand(20, 5), columns=[f"f{i}" for i in range(5)])
        y_test = pd.Series([0] * 12 + [1] * 8)

        metrics = train_random_forest(X_train, y_train, X_test, y_test, n_estimators=10)
        assert metrics["model"] == "RandomForest"
        assert metrics["type"] == "supervised"
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0


class TestTrainIsolationForest:
    def test_train_isolation_forest_returns_metrics(self):
        X_train = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(5)])
        X_test = pd.DataFrame(np.random.rand(20, 5), columns=[f"f{i}" for i in range(5)])
        y_test = pd.Series([0] * 12 + [1] * 8)

        metrics = train_isolation_forest(X_train, X_test, y_test, contamination=0.1)
        assert metrics["model"] == "IsolationForest (unsupervised)"
        assert metrics["type"] == "unsupervised"
        assert "note" in metrics


class TestEvaluateModel:
    def test_evaluate_supervised_model(self):
        from sklearn.ensemble import RandomForestClassifier
        X_test = pd.DataFrame(np.random.rand(20, 5), columns=[f"f{i}" for i in range(5)])
        y_test = pd.Series([0] * 12 + [1] * 8)

        # Train a quick model
        X_train = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(5)])
        y_train = pd.Series([0] * 30 + [1] * 20)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        metrics = evaluate_model("Test RF", model, X_test, y_test, model_type="supervised")
        assert metrics["model"] == "Test RF"
        assert metrics["type"] == "supervised"
        assert "roc_auc" in metrics

    def test_evaluate_unsupervised_model(self):
        from sklearn.ensemble import IsolationForest
        X_test = pd.DataFrame(np.random.rand(20, 5), columns=[f"f{i}" for i in range(5)])
        y_test = pd.Series([0] * 12 + [1] * 8)

        X_train = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(5)])
        model = IsolationForest(n_estimators=10, contamination=0.1, random_state=42)
        model.fit(X_train)

        metrics = evaluate_model("Test ISO", model, X_test, y_test, model_type="unsupervised")
        assert metrics["model"] == "Test ISO"
        assert metrics["type"] == "unsupervised"
        assert metrics["roc_auc"] is None


class TestFlowPredictor:
    def test_flow_predictor_requires_model(self):
        with pytest.raises(FileNotFoundError):
            FlowPredictor(model_path="nonexistent.joblib")