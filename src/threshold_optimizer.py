"""
ReturnIQ - Financial Threshold Optimization Module
Determines the cost-optimal decision threshold using a business cost-benefit matrix
evaluated exclusively on validation predictions, then evaluated frozen on the test set.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


DEFAULT_COST_CONFIG = {
    "cost_intervention": 2.50,       # Cost of proactive prompt / sizing assistance friction ($)
    "cost_return": 20.00,            # Reverse logistics, packaging, and restocking loss per return ($)
    "intervention_effectiveness": 0.40 # 40% of returns prevented when flagged early
}


def calculate_business_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    config: Dict[str, float] = DEFAULT_COST_CONFIG
) -> Dict[str, float]:
    """
    Calculate financial cost under a specified probability threshold.
    """
    c_i = config["cost_intervention"]
    c_r = config["cost_return"]
    eff = config["intervention_effectiveness"]

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Financial cost calculation
    cost_fp = fp * c_i
    cost_tp = tp * (c_i + (1.0 - eff) * c_r)
    cost_fn = fn * c_r
    total_cost = cost_fp + cost_tp + cost_fn

    # Baseline cost if we did nothing (never intervene, all returns incur full reverse shipping)
    total_actual_returns = tp + fn
    baseline_cost_no_intervention = total_actual_returns * c_r

    # Savings compared to doing nothing
    savings = baseline_cost_no_intervention - total_cost

    return {
        "threshold": round(threshold, 3),
        "total_cost": float(total_cost),
        "savings": float(savings),
        "baseline_cost": float(baseline_cost_no_intervention),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    }


def optimize_threshold(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    config: Dict[str, float] = DEFAULT_COST_CONFIG
) -> Tuple[float, pd.DataFrame, Dict[str, Any]]:
    """
    Search candidate thresholds in [0.01, 0.99] using validation set predictions.
    Identifies the optimal threshold T* that minimizes financial cost.
    """
    threshold_grid = np.linspace(0.01, 0.99, 99)
    eval_records = []

    for t in threshold_grid:
        res = calculate_business_cost(y_val_true, y_val_prob, t, config)
        eval_records.append(res)

    eval_df = pd.DataFrame(eval_records)
    best_row = eval_df.sort_values(by="total_cost", ascending=True).iloc[0]
    optimal_threshold = float(best_row["threshold"])

    # Also calculate standard 0.5 threshold cost for direct comparison
    cost_at_default = calculate_business_cost(y_val_true, y_val_prob, 0.50, config)

    optimization_summary = {
        "optimal_threshold": optimal_threshold,
        "min_cost_val": float(best_row["total_cost"]),
        "max_savings_val": float(best_row["savings"]),
        "cost_at_default_0_5": float(cost_at_default["total_cost"]),
        "savings_at_default_0_5": float(cost_at_default["savings"]),
        "baseline_cost_val": float(best_row["baseline_cost"]),
        "cost_config": config,
        "precision_at_optimal": float(best_row["precision"]),
        "recall_at_optimal": float(best_row["recall"]),
    }

    return optimal_threshold, eval_df, optimization_summary


if __name__ == "__main__":
    from src.feature_engineering import get_featured_data
    from src.train_benchmark import chronological_split, benchmark_models

    df, feature_cols, target_col = get_featured_data()
    train_df, val_df, test_df, _ = chronological_split(df)
    bundle, _, _, best_model = benchmark_models(train_df, val_df, feature_cols, target_col)

    val_prob = bundle["val_predictions"][bundle["best_model_name"]]
    y_val = bundle["y_val"]

    opt_thresh, cost_curve, summary = optimize_threshold(y_val, val_prob)
    print("\n[ThresholdOptimizer] Optimization Results on Validation Set:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")

