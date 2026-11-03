"""
ReturnIQ - Pydantic Request and Response Schemas for FastAPI
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


class PredictionRequest(BaseModel):
    Quantity: float = Field(..., gt=0, description="Number of product units purchased in transaction line")
    UnitPrice: float = Field(..., gt=0, description="Unit price of the product (£)")
    TotalSpend: Optional[float] = Field(None, description="Total spend for line item (£). Auto-calculated if omitted.")
    Hour: int = Field(12, ge=0, le=23, description="Hour of checkout (0-23)")
    DayOfWeek: int = Field(2, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    Month: int = Field(6, ge=1, le=12, description="Month of purchase (1-12)")
    IsWeekend: Optional[int] = Field(None, ge=0, le=1, description="1 if weekend, else 0")
    IsUK: int = Field(1, ge=0, le=1, description="1 if customer Country is United Kingdom, else 0")
    ProductCategory: str = Field("OTHER", description="Product prefix/category category code")
    
    # Customer historical point-in-time features
    cust_prior_orders: float = Field(0.0, ge=0, description="Number of prior distinct purchase orders")
    cust_prior_items: float = Field(0.0, ge=0, description="Number of prior items purchased")
    cust_prior_spend: float = Field(0.0, ge=0, description="Total monetary spend prior to this order (£)")
    cust_prior_returns: float = Field(0.0, ge=0, description="Total count of prior returned items")
    cust_prior_return_rate: float = Field(0.0, ge=0.0, le=1.0, description="Prior customer return rate")
    cust_days_since_last_order: float = Field(-1.0, ge=-1.0, description="Days since last order (-1 for new customer)")

    # Product historical point-in-time features
    prod_prior_purchases: float = Field(0.0, ge=0, description="Cumulative prior purchases of this product")
    prod_prior_returns: float = Field(0.0, ge=0, description="Cumulative prior returns of this product")
    prod_prior_return_rate: float = Field(0.0, ge=0.0, le=1.0, description="Historical product return rate")

    @model_validator(mode="after")
    def validate_derived_fields(self):
        if self.TotalSpend is None:
            self.TotalSpend = float(self.Quantity * self.UnitPrice)
        if self.IsWeekend is None:
            self.IsWeekend = 1 if self.DayOfWeek >= 5 else 0
        return self


class PredictionResponse(BaseModel):
    return_probability: float
    risk_label: str
    threshold_used: float
    action_recommended: str
    model_version: str


class FeatureAttribution(BaseModel):
    feature: str
    input_value: Any
    direction: str
    description: str


class ExplanationResponse(BaseModel):
    return_probability: float
    risk_label: str
    threshold_used: float
    top_contributions: List[FeatureAttribution]
    interpretation: str
    disclaimer: str


class ModelInfoResponse(BaseModel):
    service_name: str
    model_name: str
    model_version: str
    target_definition: str
    operating_threshold: float
    cost_configuration: Dict[str, Any]
    test_metrics: Dict[str, Any]
    supported_features: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None

