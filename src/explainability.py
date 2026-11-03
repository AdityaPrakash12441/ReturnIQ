"""
ReturnIQ - Model Explainability and Visualization Module
Generates global feature importance, SHAP beeswarm summary plots, local SHAP waterfall plots,
ROC/PR curves, confusion matrices, and cost-vs-threshold figures.
All figures are saved under reports/figures/.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
from typing import List, Dict, Any
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_roc_and_pr_curves(
    y_val: np.ndarray,
    val_predictions: Dict[str, np.ndarray],
    save_path_roc: str = os.path.join(FIGURES_DIR, "roc_curves.png"),
    save_path_pr: str = os.path.join(FIGURES_DIR, "pr_curves.png")
):
    """Plot and save ROC and Precision-Recall curves comparing all benchmarked models."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. ROC Curves
    plt.figure(figsize=(8, 6))
    for name, probs in val_predictions.items():
        fpr, tpr, _ = roc_curve(y_val, probs)
        plt.plot(fpr, tpr, label=name, lw=2)
    plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)", lw=1.5)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) - Validation Set", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path_roc, dpi=300)
    plt.close()
    print(f"[Explainability] Saved ROC curve to {save_path_roc}")

    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    prevalence = float(np.mean(y_val))
    plt.axhline(prevalence, color="k", linestyle="--", label=f"Baseline Prevalence ({prevalence:.1%})", lw=1.5)
    for name, probs in val_predictions.items():
        prec, rec, _ = precision_recall_curve(y_val, probs)
        plt.plot(rec, prec, label=name, lw=2)
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curve (PR-AUC) - Validation Set", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path_pr, dpi=300)
    plt.close()
    print(f"[Explainability] Saved PR curve to {save_path_pr}")


def plot_cost_vs_threshold(
    cost_df: pd.DataFrame,
    optimal_threshold: float,
    save_path: str = os.path.join(FIGURES_DIR, "cost_vs_threshold.png")
):
    """Plot total business cost vs decision threshold curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(cost_df["threshold"], cost_df["total_cost"], color="#1f77b4", lw=2.5, label="Total Expected Cost ($)")
    plt.axvline(optimal_threshold, color="#d62728", linestyle="--", lw=2, label=f"Optimal Threshold T* = {optimal_threshold:.2f}")
    plt.axvline(0.50, color="#7f7f7f", linestyle=":", lw=1.5, label="Default Threshold (0.50)")
    
    baseline_val = cost_df["baseline_cost"].iloc[0]
    plt.axhline(baseline_val, color="#ff7f0e", linestyle="-.", lw=1.5, label=f"No Intervention Baseline (${baseline_val:,.0f})")

    plt.xlabel("Decision Threshold T", fontsize=11)
    plt.ylabel("Total Expected Net Cost ($)", fontsize=11)
    plt.title("Financial Cost vs. Decision Threshold (Validation Set)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Explainability] Saved cost vs threshold curve to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    title: str = "Test Set Confusion Matrix",
    save_path: str = os.path.join(FIGURES_DIR, "confusion_matrix.png")
):
    """Plot confusion matrix heatmap."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted No Return", "Predicted Return"],
        yticklabels=["Actual No Return", "Actual Return"]
    )
    plt.title(f"{title} (Threshold = {threshold:.2f})", fontsize=12, fontweight="bold")
    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Explainability] Saved confusion matrix to {save_path}")


def generate_shap_explanations(
    model: Any,
    X_train_sample: np.ndarray,
    X_val_sample: np.ndarray,
    feature_names: List[str],
    save_summary_path: str = os.path.join(FIGURES_DIR, "shap_summary.png"),
    save_waterfall_path: str = os.path.join(FIGURES_DIR, "shap_waterfall.png"),
    save_importance_path: str = os.path.join(FIGURES_DIR, "feature_importance.png")
) -> Any:
    """
    Generate global SHAP summary (beeswarm), feature importance, and local SHAP waterfall plot.
    """
    print("[Explainability] Generating SHAP explanations...")
    
    # 1. Feature Importance plot
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(9, 6))
        plt.title("Top 15 Feature Importances", fontsize=13, fontweight="bold")
        plt.barh(range(len(indices)), importances[indices][::-1], color="#2ca02c", align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices[::-1]])
        plt.xlabel("Relative Importance", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_importance_path, dpi=300)
        plt.close()
        print(f"[Explainability] Saved feature importance plot to {save_importance_path}")

    # 2. SHAP Explainer
    # Use TreeExplainer for tree models, or standard Explainer with background sample
    try:
        # Use background sample of 100 rows for swift computation
        bg_sample = shap.sample(X_train_sample, 100, random_state=42)
        test_sample = X_val_sample[:200]

        if hasattr(model, "predict_proba"):
            # Model prediction function for positive class
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
            explainer = shap.Explainer(predict_fn, bg_sample)
            shap_values = explainer(test_sample)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(test_sample)

        # Global Beeswarm Summary Plot
        plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(shap_values, max_display=12, show=False)
        plt.title("Global Feature Attribution (SHAP Beeswarm Summary)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Explainability] Saved SHAP summary plot to {save_summary_path}")

        # Local Waterfall Plot (for the highest-risk sample in the test batch)
        highest_risk_idx = int(np.argmax(shap_values.values.sum(axis=1)))
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(shap_values[highest_risk_idx], max_display=10, show=False)
        plt.title("Local Prediction Explanation (SHAP Waterfall Plot)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_waterfall_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Explainability] Saved SHAP waterfall plot to {save_waterfall_path}")

        return explainer

    except Exception as e:
        print(f"[Explainability] SHAP generation encountered fallback note: {e}")
        return None

