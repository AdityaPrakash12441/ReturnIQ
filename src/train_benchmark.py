"""
ReturnIQ - Machine Learning Benchmarking Module
Implements chronological train/validation/test splitting, preprocessing pipeline,
and benchmarking across baseline and candidate classification models:
  1. DummyClassifier (Baseline)
  2. Logistic Regression (Interpretable linear benchmark)
  3. Random Forest (Bagging ensemble)
  4. HistGradientBoosting (LightGBM-equivalent histogram boosting)
  5. CatBoost (Gradient boosting with categorical support)
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix
)
from catboost import CatBoostClassifier

MODELS_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
COMPARISON_CSV = os.path.join(REPORTS_DIR, "model_comparison.csv")
MODEL_ARTIFACTS_PATH = os.path.join(MODELS_DIR, "model_artifacts.joblib")


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split the dataset chronologically into Train, Validation, and Test sets.
    Preserves natural temporal progression of e-commerce transactions.
    """
    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    split_info = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_dates": f"{train_df['InvoiceDate'].min()} to {train_df['InvoiceDate'].max()}",
        "val_dates": f"{val_df['InvoiceDate'].min()} to {val_df['InvoiceDate'].max()}",
        "test_dates": f"{test_df['InvoiceDate'].min()} to {test_df['InvoiceDate'].max()}",
        "train_return_rate": float(train_df["is_returned"].mean()),
        "val_return_rate": float(val_df["is_returned"].mean()),
        "test_return_rate": float(test_df["is_returned"].mean()),
    }

    # Temporal integrity assertion
    assert train_df["InvoiceDate"].max() <= val_df["InvoiceDate"].min(), "Train/Val temporal overlap!"
    assert val_df["InvoiceDate"].max() <= test_df["InvoiceDate"].min(), "Val/Test temporal overlap!"

    return train_df, val_df, test_df, split_info


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer fitted exclusively on training data."""
    num_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def benchmark_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "is_returned"
) -> Tuple[Dict[str, Any], pd.DataFrame, ColumnTransformer, Any]:
    """
    Train and evaluate benchmark models on chronological validation set.
    Selects the best performing model based on PR-AUC / ROC-AUC on validation.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    numeric_cols = [c for c in feature_cols if c != "ProductCategory"]
    categorical_cols = ["ProductCategory"]

    print("[Benchmark] Fitting preprocessing ColumnTransformer on Training set only...")
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train = preprocessor.fit_transform(train_df[feature_cols])
    y_train = train_df[target_col].values

    X_val = preprocessor.transform(val_df[feature_cols])
    y_val = val_df[target_col].values

    # Get transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_transformed_feature_names = numeric_cols + cat_names

    models_to_test = {
        "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=42),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, max_depth=8, class_weight="balanced", random_state=42),
        "CatBoost": CatBoostClassifier(iterations=250, depth=6, auto_class_weights="Balanced", verbose=0, random_seed=42)
    }

    results = []
    trained_models = {}
    val_predictions = {}

    for name, model in models_to_test.items():
        print(f"[Benchmark] Training {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_prob = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val, y_prob, threshold=0.5)
        metrics["model_name"] = name
        metrics["train_time_sec"] = round(train_time, 2)
        
        results.append(metrics)
        trained_models[name] = model
        val_predictions[name] = y_prob
        print(f"  -> {name} | Val ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | F1 (0.5): {metrics['f1']:.4f} | Time: {train_time:.2f}s")

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(COMPARISON_CSV, index=False)
    print(f"[Benchmark] Saved model comparison report to {COMPARISON_CSV}")

    # Select candidate model with highest PR-AUC on validation set
    best_candidate_name = comparison_df.sort_values(by=["pr_auc", "roc_auc"], ascending=False).iloc[0]["model_name"]
    best_model = trained_models[best_candidate_name]
    print(f"\n[Benchmark] Best Model Selected: {best_candidate_name} (Highest Validation PR-AUC)")

    benchmark_bundle = {
        "trained_models": trained_models,
        "val_predictions": val_predictions,
        "comparison_df": comparison_df,
        "best_model_name": best_candidate_name,
        "best_model": best_model,
        "preprocessor": preprocessor,
        "all_transformed_feature_names": all_transformed_feature_names,
        "y_val": y_val,
        "X_val": X_val,
        "X_train": X_train,
        "y_train": y_train
    }

    return benchmark_bundle, comparison_df, preprocessor, best_model


if __name__ == "__main__":
    from src.feature_engineering import get_featured_data

    print("[Benchmark] Loading featured dataset...")
    df, feature_cols, target_col = get_featured_data()
    train_df, val_df, test_df, split_info = chronological_split(df)
    
    print("\n[Benchmark] Chronological Split Info:")
    for k, v in split_info.items():
        print(f"  - {k}: {v}")

    bundle, comparison_df, preproc, best_model = benchmark_models(train_df, val_df, feature_cols, target_col)

