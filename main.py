"""
ReturnIQ - Main ML Pipeline Entry Point
Executes the end-to-end Machine Learning pipeline:
  1. Data Loading & Return-Label Construction
  2. Point-in-Time Feature Engineering
  3. Chronological Train / Validation / Test Splitting
  4. Multi-Model Benchmarking (Dummy, Logistic Regression, Random Forest, HistGB, CatBoost)
  5. Financial Threshold Optimization (on Validation Set)
  6. Final Unbiased Test Set Evaluation
  7. Model Explainability (SHAP global summary & local waterfall plots)
  8. Model Artifact Persistence
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np

# Ensure root directory is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import get_prepared_data
from src.feature_engineering import get_featured_data
from src.train_benchmark import chronological_split, benchmark_models, evaluate_predictions, build_preprocessor
from src.threshold_optimizer import optimize_threshold, calculate_business_cost, DEFAULT_COST_CONFIG
from src.explainability import (
    plot_roc_and_pr_curves,
    plot_cost_vs_threshold,
    plot_confusion_matrix,
    generate_shap_explanations
)

MODELS_DIR = "models"
REPORTS_DIR = "reports"
ARTIFACT_PATH = os.path.join(MODELS_DIR, "model_artifacts.joblib")
TEST_METRICS_PATH = os.path.join(REPORTS_DIR, "final_test_metrics.json")


def run_pipeline():
    total_start_time = time.time()
    print("=" * 70)
    print("               ReturnIQ: End-to-End ML Pipeline               ")
    print("=" * 70)

    # Step 1 & 2: Feature Engineering (loads data & matches returns internally)
    print("\n[Step 1 & 2] Loading data & computing point-in-time features...")
    df, feature_cols, target_col = get_featured_data()
    print(f"Dataset successfully prepared: {len(df):,} rows, {len(feature_cols)} features.")

    # Step 3: Chronological Splitting
    print("\n[Step 3] Chronological splitting (60% Train / 20% Val / 20% Test)...")
    train_df, val_df, test_df, split_info = chronological_split(df)
    for k, v in split_info.items():
        print(f"  {k}: {v}")

    # Step 4: Model Benchmarking
    print("\n[Step 4] Benchmarking classification algorithms...")
    benchmark_bundle, comparison_df, preprocessor, best_model = benchmark_models(
        train_df, val_df, feature_cols, target_col
    )
    best_name = benchmark_bundle["best_model_name"]
    print(f"\n--> Selected Model: {best_name}")

    # Step 5: Financial Threshold Optimization on Validation Set for All Models
    print("\n[Step 5] Optimizing decision threshold based on business costs (Validation Set)...")
    y_val = benchmark_bundle["y_val"]
    model_thresholds = {}
    model_opt_summaries = {}

    for name, probs in benchmark_bundle["val_predictions"].items():
        opt_t, c_curve, o_summ = optimize_threshold(y_val, probs, DEFAULT_COST_CONFIG)
        model_thresholds[name] = opt_t
        model_opt_summaries[name] = o_summ
        print(f"  - {name}: Optimal Threshold T* = {opt_t:.2f} | Val Savings: ${o_summ['max_savings_val']:,.2f}")

    optimal_threshold = model_thresholds[best_name]
    cost_curve = c_curve  # For best model visualization

    # Step 6: Final Evaluation on Touched-Once Test Set for All Models
    print("\n[Step 6] Final evaluation on held-out test set...")
    X_test = preprocessor.transform(test_df[feature_cols])
    y_test = test_df[target_col].values

    model_test_metrics = {}
    for name, model in benchmark_bundle["trained_models"].items():
        t_probs = model.predict_proba(X_test)[:, 1]
        m_thresh = model_thresholds[name]
        m_opt = evaluate_predictions(y_test, t_probs, threshold=m_thresh)
        m_opt_cost = calculate_business_cost(y_test, t_probs, threshold=m_thresh, config=DEFAULT_COST_CONFIG)
        m_def = evaluate_predictions(y_test, t_probs, threshold=0.50)
        m_def_cost = calculate_business_cost(y_test, t_probs, threshold=0.50, config=DEFAULT_COST_CONFIG)

        model_test_metrics[name] = {
            "optimal_threshold": m_thresh,
            "metrics_at_optimal_threshold": m_opt,
            "costs_at_optimal_threshold": m_opt_cost,
            "metrics_at_default_0_5": m_def,
            "costs_at_default_0_5": m_def_cost
        }

    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_metrics_opt = model_test_metrics[best_name]["metrics_at_optimal_threshold"]
    test_cost_opt = model_test_metrics[best_name]["costs_at_optimal_threshold"]
    test_metrics_default = model_test_metrics[best_name]["metrics_at_default_0_5"]
    test_cost_default = model_test_metrics[best_name]["costs_at_default_0_5"]

    print("\n" + "-" * 50)
    print(f"     FINAL TEST SET RESULTS: SELECTED BEST ({best_name})     ")
    print("-" * 50)
    print(f"Test Positive Prevalence (Returns): {test_metrics_opt['tp'] + test_metrics_opt['fn']} / {len(y_test)} ({np.mean(y_test):.2%})")
    print(f"ROC-AUC:                           {test_metrics_opt['roc_auc']:.4f}")
    print(f"PR-AUC (Average Precision):        {test_metrics_opt['pr_auc']:.4f}")
    print(f"Optimal Threshold T*:              {optimal_threshold:.2f}")
    print(f"Precision @ T*:                    {test_metrics_opt['precision']:.4f}")
    print(f"Recall @ T*:                       {test_metrics_opt['recall']:.4f}")
    print(f"F1-Score @ T*:                     {test_metrics_opt['f1']:.4f}")
    print(f"Test Net Savings @ T*:             ${test_cost_opt['savings']:,.2f}")
    print(f"Test Net Savings @ 0.50:           ${test_cost_default['savings']:,.2f}")
    print(f"Gain from Threshold Optimization:  ${test_cost_opt['savings'] - test_cost_default['savings']:,.2f}")
    print("-" * 50)

    # Save test metrics JSON
    final_test_summary = {
        "model_name": best_name,
        "optimal_threshold": optimal_threshold,
        "metrics_at_optimal_threshold": test_metrics_opt,
        "costs_at_optimal_threshold": test_cost_opt,
        "metrics_at_default_0_5": test_metrics_default,
        "costs_at_default_0_5": test_cost_default,
        "split_info": split_info,
        "by_model": model_test_metrics,
        "execution_date": pd.Timestamp.now().isoformat()
    }
    with open(TEST_METRICS_PATH, "w") as f:
        json.dump(final_test_summary, f, indent=2)
    print(f"\nSaved final test metrics to {TEST_METRICS_PATH}")

    # Step 7: Visualizations & Explainability
    print("\n[Step 7] Generating visualization figures and SHAP explanations...")
    plot_roc_and_pr_curves(y_val, benchmark_bundle["val_predictions"])
    plot_cost_vs_threshold(cost_curve, optimal_threshold)
    plot_confusion_matrix(y_test, test_probs, optimal_threshold, title=f"Test Confusion Matrix ({best_name})")

    # Generate SHAP
    transformed_feature_names = benchmark_bundle["all_transformed_feature_names"]
    generate_shap_explanations(
        best_model,
        benchmark_bundle["X_train"],
        benchmark_bundle["X_val"],
        transformed_feature_names
    )

    # Step 8: Save Model Artifacts
    print("\n[Step 8] Serializing model artifacts for production inference...")
    artifacts = {
        "model": best_model,
        "model_name": best_name,
        "models": benchmark_bundle["trained_models"],
        "model_thresholds": model_thresholds,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "transformed_feature_names": transformed_feature_names,
        "optimal_threshold": optimal_threshold,
        "cost_config": DEFAULT_COST_CONFIG,
        "metadata": {
            "version": "1.0.0",
            "best_model_name": best_name,
            "available_models": list(benchmark_bundle["trained_models"].keys()),
            "test_roc_auc": test_metrics_opt["roc_auc"],
            "test_pr_auc": test_metrics_opt["pr_auc"],
            "test_f1": test_metrics_opt["f1"],
            "target": "is_returned",
            "date": pd.Timestamp.now().isoformat()
        }
    }
    joblib.dump(artifacts, ARTIFACT_PATH)
    print(f"Model artifacts successfully serialized to {ARTIFACT_PATH}")

    elapsed = time.time() - total_start_time
    print(f"\n[Done] Pipeline completed successfully in {elapsed:.1f} seconds!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()

