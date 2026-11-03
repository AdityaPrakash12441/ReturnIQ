"""
ReturnIQ — Inference Helper
Loads saved model artifacts and exposes dynamic multi-model selection and prediction.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List

logger = logging.getLogger("returniq.inference")

CANDIDATE_PATHS = [
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "model_artifacts.joblib")),
    os.path.normpath(os.path.join(os.getcwd(), "models", "model_artifacts.joblib")),
    os.path.normpath("models/model_artifacts.joblib"),
    os.path.abspath("models/model_artifacts.joblib"),
]


def get_artifact_path() -> Optional[str]:
    for p in CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    return None


_LAST_LOAD_ERROR: Optional[str] = None


def get_last_load_error() -> Optional[str]:
    return _LAST_LOAD_ERROR


def load_artifacts() -> Optional[Dict[str, Any]]:
    """Load and return the serialised model bundle, or None on failure."""
    global _LAST_LOAD_ERROR
    _LAST_LOAD_ERROR = None

    path = get_artifact_path()
    if not path:
        _LAST_LOAD_ERROR = f"File 'models/model_artifacts.joblib' not found in search paths: {CANDIDATE_PATHS}"
        logger.warning(_LAST_LOAD_ERROR)
        return None

    try:
        bundle = joblib.load(path)
        return bundle
    except Exception as exc:
        _LAST_LOAD_ERROR = f"Failed to unpickle '{path}': {type(exc).__name__}: {exc}"
        logger.error(_LAST_LOAD_ERROR)
        return None


def get_available_models(bundle: Dict[str, Any]) -> List[str]:
    """Return list of trained model names available in the artifact."""
    if "models" in bundle and isinstance(bundle["models"], dict):
        models = list(bundle["models"].keys())
        best_name = bundle.get("model_name")
        # Put best model first for convenience
        if best_name in models:
            models.remove(best_name)
            return [best_name] + models
        return models
    elif "model_name" in bundle:
        return [bundle["model_name"]]
    return []


def get_model_threshold(
    bundle: Dict[str, Any],
    model_name: str,
    use_optimal: bool = True
) -> float:
    """Retrieve the operating threshold (either optimal T* or standard 0.50)."""
    if not use_optimal:
        return 0.50
    thresholds = bundle.get("model_thresholds", {})
    if model_name in thresholds:
        return float(thresholds[model_name])
    return float(bundle.get("optimal_threshold", 0.50))


def predict(
    bundle: Dict[str, Any],
    input_data: Dict[str, Any],
    model_name: Optional[str] = None,
    threshold: Optional[float] = None,
    use_optimal_threshold: bool = True,
) -> Tuple[float, str, float, str]:
    """
    Run inference on a single transaction dict using the requested model.

    Returns
    -------
    (probability, risk_label, threshold_used, resolved_model_name)
    """
    feature_cols = bundle["feature_cols"]
    preprocessor = bundle["preprocessor"]

    # Resolve model
    available_models = bundle.get("models", {})
    if model_name and model_name in available_models:
        model = available_models[model_name]
        resolved_name = model_name
    elif "model" in bundle:
        model = bundle["model"]
        resolved_name = bundle.get("model_name", "Primary Model")
    elif available_models:
        first_key = next(iter(available_models))
        model = available_models[first_key]
        resolved_name = first_key
    else:
        raise ValueError("No valid model found in artifact bundle.")

    # Resolve threshold
    if threshold is not None:
        active_threshold = float(threshold)
    else:
        active_threshold = get_model_threshold(bundle, resolved_name, use_optimal=use_optimal_threshold)

    # Build single-row DataFrame matching the exact feature columns
    row = {col: input_data.get(col, 0.0) for col in feature_cols}
    df = pd.DataFrame([row])[feature_cols]

    X = preprocessor.transform(df)
    prob = float(model.predict_proba(X)[0, 1])
    label = "High Risk" if prob >= active_threshold else "Low Risk"

    return prob, label, active_threshold, resolved_name
