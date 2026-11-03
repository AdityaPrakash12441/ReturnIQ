"""
Tests for FastAPI Endpoints using HTTPX AsyncClient.
"""

import pytest
import asyncio
from httpx import ASGITransport, AsyncClient
from api.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_endpoint():
    """Verify GET /health returns 200 and model loaded status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "model_loaded" in data


@pytest.mark.anyio
async def test_model_info_endpoint():
    """Verify GET /model/info returns model metadata."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/model/info")
        assert res.status_code == 200
        data = res.json()
        assert "model_name" in data
        assert "operating_threshold" in data
        assert len(data["supported_features"]) > 0


@pytest.mark.anyio
async def test_predict_endpoint_valid():
    """Verify POST /predict returns probability and risk classification."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "Quantity": 10.0,
            "UnitPrice": 5.50,
            "Hour": 15,
            "DayOfWeek": 3,
            "Month": 9,
            "IsUK": 1,
            "ProductCategory": "85",
            "cust_prior_orders": 3.0,
            "cust_prior_items": 15.0,
            "cust_prior_spend": 90.0,
            "cust_prior_returns": 3.0,
            "cust_prior_return_rate": 0.20,
            "cust_days_since_last_order": 10.0,
            "prod_prior_purchases": 100.0,
            "prod_prior_returns": 8.0,
            "prod_prior_return_rate": 0.08
        }
        res = await client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert 0.0 <= data["return_probability"] <= 1.0
        assert data["risk_label"] in ["High Risk", "Low Risk"]
        assert "threshold_used" in data


@pytest.mark.anyio
async def test_predict_endpoint_validation_error():
    """Verify POST /predict rejects invalid negative quantities with HTTP 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        invalid_payload = {
            "Quantity": -5.0,  # Invalid: Quantity must be > 0
            "UnitPrice": 5.50
        }
        res = await client.post("/predict", json=invalid_payload)
        assert res.status_code == 422


@pytest.mark.anyio
async def test_explain_endpoint():
    """Verify POST /explain returns local feature attributions and disclaimer."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "Quantity": 25.0,  # High quantity
            "UnitPrice": 12.0,
            "Hour": 12,
            "DayOfWeek": 1,
            "Month": 5,
            "IsUK": 1,
            "ProductCategory": "22",
            "cust_prior_orders": 5.0,
            "cust_prior_items": 40.0,
            "cust_prior_spend": 300.0,
            "cust_prior_returns": 10.0,
            "cust_prior_return_rate": 0.25,
            "cust_days_since_last_order": 5.0,
            "prod_prior_purchases": 500.0,
            "prod_prior_returns": 45.0,
            "prod_prior_return_rate": 0.09
        }
        res = await client.post("/explain", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "top_contributions" in data
        assert len(data["top_contributions"]) > 0
        assert "disclaimer" in data

