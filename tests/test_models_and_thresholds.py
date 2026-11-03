"""
Tests for Model Artifacts, Preprocessing Pipelines, and Threshold Optimization.
"""

import os
import pytest
import joblib
import numpy as np
import pandas as pd

from src.threshold_optimizer import calculate_business_cost, optimize_threshold, DEFAULT_COST_CONFIG
from src.train_benchmark import evaluate_predictions, build_preprocessor


def test_business_cost_calculation():
    """Verify financial cost calculation and savings logic."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.8, 0.9])
    
    # At threshold 0.5:
    # y_pred = [0, 0, 1, 1] -> TN=2, FP=0, FN=0, TP=2 (perfect classification)
    cost_info = calculate_business_cost(y_true, y_prob, threshold=0.5, config=DEFAULT_COST_CONFIG)
    
    # Baseline cost: 2 returns * $20 = $40
    assert cost_info["baseline_cost"] == 40.0
    # True positives cost: 2 * ($2.50 + 0.6 * $20) = 2 * ($2.50 + $12) = $29.00
    assert cost_info["total_cost"] == 29.0
    assert cost_info["savings"] == 11.0


def test_threshold_optimizer_validation_only():
    """Verify that optimal threshold is chosen to minimize cost."""
    # Synthetic imbalanced validation probabilities
    np.random.seed(42)
    y_val = np.random.binomial(1, 0.05, 500)
    # Give true returns slightly higher probabilities
    y_prob = np.where(y_val == 1, np.random.uniform(0.4, 0.95, 500), np.random.uniform(0.01, 0.6, 500))

    opt_thresh, cost_df, summary = optimize_threshold(y_val, y_prob, DEFAULT_COST_CONFIG)

    assert 0.01 <= opt_thresh <= 0.99
    assert summary["min_cost_val"] <= summary["baseline_cost_val"]
    assert "cost_config" in summary


def test_artifact_persistence_and_reload():
    """Verify that serialized model artifacts exist and can be reloaded safely."""
    artifact_path = "models/model_artifacts.joblib"
    assert os.path.exists(artifact_path), f"Artifact missing at {artifact_path}. Run main.py first."

    bundle = joblib.load(artifact_path)
    assert "model" in bundle
    assert "preprocessor" in bundle
    assert "feature_cols" in bundle
    assert "optimal_threshold" in bundle
    assert hasattr(bundle["model"], "predict_proba")



def test_multi_model_selection_inference():
    """Verify that multiple models can be dynamically selected and evaluated."""
    from src.inference import load_artifacts, get_available_models, predict

    bundle = load_artifacts()
    assert bundle is not None
    models = get_available_models(bundle)
    assert len(models) >= 2, "Expected at least 2 models saved in artifact bundle."

    sample_tx = {
        "Quantity": 10.0, "UnitPrice": 8.50, "TotalSpend": 85.0,
        "Hour": 14, "DayOfWeek": 2, "Month": 10,
        "IsWeekend": 0, "IsUK": 1, "ProductCategory": "85",
        "cust_prior_orders": 3.0, "cust_prior_items": 20.0,
        "cust_prior_spend": 150.0, "cust_prior_returns": 4.0,
        "cust_prior_return_rate": 0.20, "cust_days_since_last_order": 14.0,
        "prod_prior_purchases": 120.0, "prod_prior_returns": 10.0,
        "prod_prior_return_rate": 0.08
    }

    results = {}
    for m in models:
        prob, label, thresh, name = predict(bundle, sample_tx, model_name=m, use_optimal_threshold=True)
        assert 0.0 <= prob <= 1.0
        assert label in ["High Risk", "Low Risk"]
        assert 0.0 < thresh <= 1.0
        assert name == m
        results[m] = prob

    # Verify that different models produce distinct probability scores
    probs_set = set(round(p, 4) for p in results.values())
    assert len(probs_set) > 1, f"Different models should produce different predictions: {results}"
