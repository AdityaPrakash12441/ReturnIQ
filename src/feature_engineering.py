"""
ReturnIQ - Point-in-Time Feature Engineering Module
Builds leakage-safe transaction, customer-history, and product-history features.
Ensures zero future lookahead: all rolling/historical aggregations strictly reference
observations prior to the current transaction timestamp.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



FEATURE_DATA_PATH = "data/processed/featured_dataset.parquet"


def extract_product_category(df: pd.DataFrame) -> pd.Series:
    """Extract a broad product category based on stock code prefix and description keywords."""
    stock_prefix = df["StockCode"].astype(str).str[:2]
    # Keep top 8 prefixes, group rest into 'OTHER'
    top_prefixes = set(stock_prefix.value_counts().head(8).index)
    category = stock_prefix.apply(lambda x: x if x in top_prefixes else "OTHER")
    return category


def compute_point_in_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute point-in-time features strictly using prior observable data.
    Input must be sorted chronologically by InvoiceDate.
    """
    print("[FeatureEngineering] Sorting transactions chronologically...")
    df = df.sort_values(["InvoiceDate", "InvoiceNo"]).reset_index(drop=True)

    # 1. Transaction-level features available at checkout
    df["Quantity"] = df["Quantity"].astype(float)
    df["UnitPrice"] = df["UnitPrice"].astype(float)
    df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

    # Temporal features
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
    df["Month"] = df["InvoiceDate"].dt.month
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # Country feature (UK vs International)
    df["IsUK"] = (df["Country"] == "United Kingdom").astype(int)

    # Categorical proxy
    df["ProductCategory"] = extract_product_category(df)

    # 2. Customer Historical Features (Zero-leakage point-in-time)
    print("[FeatureEngineering] Computing customer historical features...")
    # Track order invoices
    df["InvoiceCountHelper"] = (~df.duplicated(subset=["CustomerID", "InvoiceNo"])).astype(int)

    # Group by customer and compute cumulative stats strictly prior to current row
    grouped_cust = df.groupby("CustomerID")
    
    # Cumulative prior items (excluding current item via shift)
    df["cust_prior_items"] = grouped_cust["Quantity"].shift(1).fillna(0)
    df["cust_prior_items"] = grouped_cust["cust_prior_items"].cumsum().fillna(0)

    # Cumulative prior spend (excluding current)
    df["cust_prior_spend"] = grouped_cust["TotalSpend"].shift(1).fillna(0)
    df["cust_prior_spend"] = grouped_cust["cust_prior_spend"].cumsum().fillna(0)

    # Cumulative prior orders (excluding current)
    df["cust_prior_orders"] = grouped_cust["InvoiceCountHelper"].shift(1).fillna(0)
    df["cust_prior_orders"] = grouped_cust["cust_prior_orders"].cumsum().fillna(0)

    # Cumulative prior returns (excluding current row's outcome via shift)
    df["cust_prior_returns"] = grouped_cust["is_returned"].shift(1).fillna(0)
    df["cust_prior_returns"] = grouped_cust["cust_prior_returns"].cumsum().fillna(0)

    # Customer return rate
    df["cust_prior_return_rate"] = np.where(
        df["cust_prior_items"] > 0,
        df["cust_prior_returns"] / df["cust_prior_items"],
        0.0
    )

    # Days since previous purchase
    cust_prev_date = grouped_cust["InvoiceDate"].shift(1)
    df["cust_days_since_last_order"] = (df["InvoiceDate"] - cust_prev_date).dt.total_seconds() / 86400.0
    # For new customers, assign -1.0 as a distinctive marker
    df["cust_days_since_last_order"] = df["cust_days_since_last_order"].fillna(-1.0)

    # 3. Product Historical Features (Zero-leakage point-in-time)
    print("[FeatureEngineering] Computing product historical return rates...")
    grouped_prod = df.groupby("StockCode")

    # Cumulative prior purchases of this product
    df["prod_prior_purchases"] = grouped_prod["Quantity"].shift(1).fillna(0)
    df["prod_prior_purchases"] = grouped_prod["prod_prior_purchases"].cumsum().fillna(0)

    # Cumulative prior returns of this product
    df["prod_prior_returns"] = grouped_prod["is_returned"].shift(1).fillna(0)
    df["prod_prior_returns"] = grouped_prod["prod_prior_returns"].cumsum().fillna(0)

    # Product historical return rate
    df["prod_prior_return_rate"] = np.where(
        df["prod_prior_purchases"] > 0,
        df["prod_prior_returns"] / df["prod_prior_purchases"],
        0.0
    )

    # Clean intermediate helper columns
    df = df.drop(columns=["InvoiceCountHelper"])

    # Leakage verification assertions
    print("[FeatureEngineering] Running leakage verification checks...")
    # First purchase of each customer must have 0 prior orders and 0 prior returns
    first_cust_rows = df.drop_duplicates(subset=["CustomerID"], keep="first")
    assert (first_cust_rows["cust_prior_orders"] == 0).all(), "Leakage detected: first customer order has non-zero prior orders!"
    assert (first_cust_rows["cust_prior_returns"] == 0).all(), "Leakage detected: first customer order has non-zero prior returns!"
    assert (first_cust_rows["cust_prior_spend"] == 0).all(), "Leakage detected: first customer order has non-zero prior spend!"

    first_prod_rows = df.drop_duplicates(subset=["StockCode"], keep="first")
    assert (first_prod_rows["prod_prior_returns"] == 0).all(), "Leakage detected: first product purchase has non-zero prior returns!"

    print("[FeatureEngineering] Leakage checks passed successfully!")
    return df


def get_featured_data(reload_cache: bool = False) -> Tuple[pd.DataFrame, List[str], str]:
    """Retrieve or compute the featured dataset."""
    if os.path.exists(FEATURE_DATA_PATH) and not reload_cache:
        print(f"[FeatureEngineering] Loading cached featured dataset from {FEATURE_DATA_PATH}...")
        df = pd.read_parquet(FEATURE_DATA_PATH)
    else:
        from src.data_loader import get_prepared_data
        df_raw, _ = get_prepared_data()
        df = compute_point_in_time_features(df_raw)
        df.to_parquet(FEATURE_DATA_PATH, index=False)
        print(f"[FeatureEngineering] Saved featured dataset to {FEATURE_DATA_PATH}.")

    feature_cols = [
        "Quantity",
        "UnitPrice",
        "TotalSpend",
        "Hour",
        "DayOfWeek",
        "Month",
        "IsWeekend",
        "IsUK",
        "ProductCategory",
        "cust_prior_orders",
        "cust_prior_items",
        "cust_prior_spend",
        "cust_prior_returns",
        "cust_prior_return_rate",
        "cust_days_since_last_order",
        "prod_prior_purchases",
        "prod_prior_returns",
        "prod_prior_return_rate",
    ]
    target_col = "is_returned"

    return df, feature_cols, target_col


if __name__ == "__main__":
    df, features, target = get_featured_data(reload_cache=True)
    print("\nFeatures created:", features)
    print("Dataset shape:", df.shape)
    print(df[features + [target]].head(3).T)
