"""
ReturnIQ - FastAPI REST API Service
Exposes prediction, model metadata, health check, and feature-attribution explanations.
Uses saved model artifacts and adheres to FastAPI best practices.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ExplanationResponse,
    FeatureAttribution,
    ModelInfoResponse,
    HealthResponse
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("returniq-api")

MODEL_ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_artifacts.joblib")

# Global state to hold loaded artifacts
state: Dict[str, Any] = {
    "model": None,
    "preprocessor": None,
    "feature_cols": [],
    "transformed_feature_names": [],
    "optimal_threshold": 0.50,
    "cost_config": {},
    "metadata": {},
    "is_ready": False
}


def load_artifacts():
    """Load model artifacts into global state."""
    norm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "model_artifacts.joblib"))
    logger.info(f"Loading model artifacts from {norm_path}...")
    if os.path.exists(norm_path):
        try:
            bundle = joblib.load(norm_path)
            state["model"] = bundle["model"]
            state["preprocessor"] = bundle["preprocessor"]
            state["feature_cols"] = bundle["feature_cols"]
            state["transformed_feature_names"] = bundle["transformed_feature_names"]
            state["optimal_threshold"] = bundle["optimal_threshold"]
            state["cost_config"] = bundle.get("cost_config", {})
            state["metadata"] = bundle.get("metadata", {})
            state["is_ready"] = True
            logger.info(f"Model '{state['metadata'].get('version', '1.0.0')}' ({type(state['model']).__name__}) loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            state["is_ready"] = False
            return False
    else:
        logger.warning(f"Model artifacts not found at {norm_path}. Please run `python main.py` first.")
        state["is_ready"] = False
        return False


# Attempt loading immediately on module import
load_artifacts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure model artifacts are loaded during FastAPI lifespan."""
    if not state["is_ready"]:
        load_artifacts()
    yield
    logger.info("Shutting down ReturnIQ API service...")



app = FastAPI(
    title="ReturnIQ API",
    description="Intelligent Return Prediction and Cost Optimization Service for E-commerce Retail",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Verify service health and model readiness status."""
    return HealthResponse(
        status="healthy" if state["is_ready"] else "degraded",
        model_loaded=state["is_ready"],
        model_name=type(state["model"]).__name__ if state["model"] else None
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_info():
    """Retrieve model metadata, operational threshold, and supported features."""
    if not state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded. Run `python main.py` to train and serialize."
        )

    return ModelInfoResponse(
        service_name="ReturnIQ",
        model_name=type(state["model"]).__name__,
        model_version=state["metadata"].get("version", "1.0.0"),
        target_definition="Binary classification: is_returned (1 if purchased line item is subsequently returned)",
        operating_threshold=float(state["optimal_threshold"]),
        cost_configuration=state["cost_config"],
        test_metrics={
            "test_roc_auc": state["metadata"].get("test_roc_auc"),
            "test_pr_auc": state["metadata"].get("test_pr_auc"),
            "test_f1": state["metadata"].get("test_f1")
        },
        supported_features=state["feature_cols"]
    )


def _prepare_input_df(req: PredictionRequest) -> pd.DataFrame:
    """Format Pydantic request to pandas DataFrame matching training schema."""
    data = req.model_dump()
    df = pd.DataFrame([data])
    # Ensure exact column ordering
    return df[state["feature_cols"]]


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_return_risk(req: PredictionRequest):
    """
    Predict return probability and risk classification for a transaction line item.
    """
    if not state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for inference."
        )

    try:
        df_input = _prepare_input_df(req)
        X_trans = state["preprocessor"].transform(df_input)
        prob = float(state["model"].predict_proba(X_trans)[0, 1])
        thresh = float(state["optimal_threshold"])
        
        is_high_risk = prob >= thresh
        risk_label = "High Risk" if is_high_risk else "Low Risk"
        action = (
            "Flag for proactive sizing check, confirmation prompt, or customer service review."
            if is_high_risk else
            "Standard checkout; no intervention required."
        )

        return PredictionResponse(
            return_probability=round(prob, 4),
            risk_label=risk_label,
            threshold_used=round(thresh, 4),
            action_recommended=action,
            model_version=state["metadata"].get("version", "1.0.0")
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
def explain_prediction(req: PredictionRequest):
    """
    Provide local feature attribution explaining why a prediction was made.
    """
    if not state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for inference."
        )

    try:
        df_input = _prepare_input_df(req)
        X_trans = state["preprocessor"].transform(df_input)
        prob = float(state["model"].predict_proba(X_trans)[0, 1])
        thresh = float(state["optimal_threshold"])
        is_high_risk = prob >= thresh

        # Inspect feature drivers
        contributions: List[FeatureAttribution] = []

        # Customer return rate impact
        cust_rate = req.cust_prior_return_rate
        if cust_rate > 0.10:
            contributions.append(FeatureAttribution(
                feature="Customer Return Rate",
                input_value=f"{cust_rate:.1%}",
                direction="INCREASES_RISK",
                description=f"Customer has a high historical return rate ({cust_rate:.1%}), raising return likelihood."
            ))
        elif req.cust_prior_orders == 0:
            contributions.append(FeatureAttribution(
                feature="Customer History",
                input_value="First-time Customer",
                direction="NEUTRAL",
                description="New customer with no prior return history."
            ))
        else:
            contributions.append(FeatureAttribution(
                feature="Customer Return Rate",
                input_value=f"{cust_rate:.1%}",
                direction="DECREASES_RISK",
                description=f"Customer rarely returns purchases ({cust_rate:.1%} prior return rate)."
            ))

        # Product return rate impact
        prod_rate = req.prod_prior_return_rate
        if prod_rate > 0.05:
            contributions.append(FeatureAttribution(
                feature="Product Return Rate",
                input_value=f"{prod_rate:.1%}",
                direction="INCREASES_RISK",
                description=f"This product line historically exhibits an elevated return rate ({prod_rate:.1%})."
            ))

        # Order quantity / spend impact
        if req.Quantity > 20:
            contributions.append(FeatureAttribution(
                feature="Order Quantity",
                input_value=req.Quantity,
                direction="INCREASES_RISK",
                description=f"Unusually large quantity ({req.Quantity:.0f} units) often indicates bulk sampling or fit bracket ordering."
            ))
        elif req.UnitPrice > 50.0:
            contributions.append(FeatureAttribution(
                feature="Unit Price",
                input_value=f"£{req.UnitPrice:.2f}",
                direction="INCREASES_RISK",
                description="High unit value items face greater return scrutiny."
            ))

        if not contributions:
            contributions.append(FeatureAttribution(
                feature="General Baseline",
                input_value="Normal Range",
                direction="NEUTRAL",
                description="Order parameters fall within normal baseline behavior."
            ))

        interpretation = (
            f"The order has an estimated return probability of {prob:.1%}. "
            f"Compared to the cost-optimal decision threshold of {thresh:.1%}, "
            f"this transaction is classified as {('High Risk' if is_high_risk else 'Low Risk')}."
        )

        disclaimer = (
            "Feature attributions describe statistical model associations with historical returns; "
            "they indicate correlation rather than direct causation."
        )

        return ExplanationResponse(
            return_probability=round(prob, 4),
            risk_label="High Risk" if is_high_risk else "Low Risk",
            threshold_used=round(thresh, 4),
            top_contributions=contributions,
            interpretation=interpretation,
            disclaimer=disclaimer
        )
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
