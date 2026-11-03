"""
Tests for Data Loading, Return Label Construction, and Leakage-Safe Feature Engineering.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data_loader import construct_return_labels
from src.feature_engineering import compute_point_in_time_features
from src.train_benchmark import chronological_split


def test_return_matching_logic():
    """Verify that returns match prior purchases correctly without future lookahead."""
    base_time = datetime(2021, 1, 1, 10, 0)
    
    # Customer 101 buys item A on Jan 1
    # Customer 101 returns item A on Jan 5 (4 days later -> match)
    # Customer 102 returns item B on Jan 2 with NO prior purchase -> unmatched
    purchases = pd.DataFrame([
        {
            "InvoiceNo": "1001",
            "StockCode": "ITEM_A",
            "Description": "Product A",
            "Quantity": 2,
            "InvoiceDate": base_time,
            "UnitPrice": 10.0,
            "CustomerID": "101",
            "Country": "United Kingdom"
        },
        {
            "InvoiceNo": "1002",
            "StockCode": "ITEM_C",
            "Description": "Product C",
            "Quantity": 1,
            "InvoiceDate": base_time + timedelta(days=2),
            "UnitPrice": 5.0,
            "CustomerID": "103",
            "Country": "United Kingdom"
        }
    ])

    returns = pd.DataFrame([
        {
            "InvoiceNo": "C1003",
            "StockCode": "ITEM_A",
            "Description": "Product A",
            "Quantity": -1,
            "InvoiceDate": base_time + timedelta(days=4),
            "UnitPrice": 10.0,
            "CustomerID": "101",
            "Country": "United Kingdom"
        },
        {
            "InvoiceNo": "C1004",
            "StockCode": "ITEM_B",
            "Description": "Product B",
            "Quantity": -1,
            "InvoiceDate": base_time + timedelta(days=1),
            "UnitPrice": 15.0,
            "CustomerID": "102",
            "Country": "United Kingdom"
        }
    ])

    labeled_purchases, stats = construct_return_labels(purchases, returns, max_return_window_days=30)

    assert stats["matched_returns"] == 1
    assert stats["unmatched_returns"] == 1
    assert labeled_purchases.loc[0, "is_returned"] == 1
    assert labeled_purchases.loc[1, "is_returned"] == 0
    assert labeled_purchases.loc[0, "return_days"] == pytest.approx(4.0, 0.1)


def test_leakage_prevention_on_historical_features():
    """Verify that customer history features strictly use information before the current row."""
    base_time = datetime(2021, 1, 1, 10, 0)
    
    # 3 sequential purchases for customer 201
    sample_df = pd.DataFrame([
        {
            "InvoiceNo": "2001",
            "StockCode": "ITEM_X",
            "Description": "Item X",
            "Quantity": 2,
            "InvoiceDate": base_time,
            "UnitPrice": 10.0,
            "CustomerID": "201",
            "Country": "United Kingdom",
            "is_returned": 1  # Note: return happened later
        },
        {
            "InvoiceNo": "2002",
            "StockCode": "ITEM_Y",
            "Description": "Item Y",
            "Quantity": 5,
            "InvoiceDate": base_time + timedelta(days=3),
            "UnitPrice": 4.0,
            "CustomerID": "201",
            "Country": "United Kingdom",
            "is_returned": 0
        },
        {
            "InvoiceNo": "2003",
            "StockCode": "ITEM_Z",
            "Description": "Item Z",
            "Quantity": 1,
            "InvoiceDate": base_time + timedelta(days=10),
            "UnitPrice": 50.0,
            "CustomerID": "201",
            "Country": "United Kingdom",
            "is_returned": 0
        }
    ])

    featured_df = compute_point_in_time_features(sample_df)

    # 1. First order MUST have zero prior history
    assert featured_df.loc[0, "cust_prior_orders"] == 0.0
    assert featured_df.loc[0, "cust_prior_items"] == 0.0
    assert featured_df.loc[0, "cust_prior_spend"] == 0.0
    assert featured_df.loc[0, "cust_prior_returns"] == 0.0
    assert featured_df.loc[0, "cust_prior_return_rate"] == 0.0
    assert featured_df.loc[0, "cust_days_since_last_order"] == -1.0

    # 2. Second order must reflect ONLY the 1st order (2 items, £20 spend)
    assert featured_df.loc[1, "cust_prior_orders"] == 1.0
    assert featured_df.loc[1, "cust_prior_items"] == 2.0
    assert featured_df.loc[1, "cust_prior_spend"] == 20.0
    assert featured_df.loc[1, "cust_days_since_last_order"] == pytest.approx(3.0, 0.1)

    # 3. Third order must reflect the prior 2 orders (7 items, £40 spend)
    assert featured_df.loc[2, "cust_prior_orders"] == 2.0
    assert featured_df.loc[2, "cust_prior_items"] == 7.0
    assert featured_df.loc[2, "cust_prior_spend"] == 40.0
    assert featured_df.loc[2, "cust_days_since_last_order"] == pytest.approx(7.0, 0.1)


def test_chronological_split_strictness():
    """Verify that chronological splitting contains zero temporal overlap."""
    n = 100
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "InvoiceDate": dates,
        "is_returned": np.random.binomial(1, 0.05, n)
    })

    train_df, val_df, test_df, info = chronological_split(df, train_frac=0.6, val_frac=0.2)

    assert len(train_df) == 60
    assert len(val_df) == 20
    assert len(test_df) == 20
    assert train_df["InvoiceDate"].max() <= val_df["InvoiceDate"].min()
    assert val_df["InvoiceDate"].max() <= test_df["InvoiceDate"].min()

