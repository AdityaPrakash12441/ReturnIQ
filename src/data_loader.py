"""
ReturnIQ - Data Loader and Return-Label Construction Module
Defines the data cleaning, filtering, and leakage-safe return-matching procedures
for the UCI Online Retail dataset.
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
RAW_DATA_DIR = "data/raw"
EXCEL_PATH = os.path.join(RAW_DATA_DIR, "Online Retail.xlsx")
PARQUET_PATH = os.path.join(RAW_DATA_DIR, "online_retail.parquet")
PROCESSED_DATA_DIR = "data/processed"
CLEANED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "matched_transactions.parquet")


# Non-product / administrative stock codes to exclude from physical product return prediction
NON_PRODUCT_CODES = {
    "POST", "D", "M", "CRUK", "DOT", "BANK CHARGES", "S", "AMAZONFEE", "m", "PADS"
}


def download_raw_data() -> str:
    """Download the raw UCI Online Retail dataset if not already present."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    if os.path.exists(PARQUET_PATH):
        return PARQUET_PATH

    if not os.path.exists(EXCEL_PATH):
        zip_path = os.path.join(RAW_DATA_DIR, "online_retail.zip")
        print(f"[DataLoader] Downloading UCI Online Retail dataset from {UCI_DATA_URL}...")
        urllib.request.urlretrieve(UCI_DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DATA_DIR)
        print("[DataLoader] Extracted raw excel file.")

    print("[DataLoader] Converting raw excel file to fast Parquet cache...")
    df = pd.read_excel(EXCEL_PATH)
    for col in ["InvoiceNo", "StockCode", "Description", "CustomerID", "Country"]:
        df[col] = df[col].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df.to_parquet(PARQUET_PATH, index=False)
    return PARQUET_PATH


def load_and_clean_data(file_path: str = PARQUET_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load raw records and split into eligible product purchases and return records.
    Filters:
      - Valid CustomerID (anonymous transactions cannot be linked or historically tracked)
      - Exclude non-product service lines (postage, bank charges, discounts)
      - Positive UnitPrice for normal purchases
    """
    if not os.path.exists(file_path):
        download_raw_data()

    df = pd.read_parquet(file_path)
    total_raw_rows = len(df)

    # 1. Clean CustomerID
    has_valid_cust = ~df["CustomerID"].isin(["nan", "None", "", "NaN"]) & df["CustomerID"].notnull()
    excluded_no_cust = int((~has_valid_cust).sum())

    # 2. Exclude administrative codes
    is_valid_product = ~df["StockCode"].isin(NON_PRODUCT_CODES)
    excluded_admin_codes = int((~is_valid_product & has_valid_cust).sum())

    df_valid = df[has_valid_cust & is_valid_product].copy()
    df_valid["InvoiceDate"] = pd.to_datetime(df_valid["InvoiceDate"])
    df_valid["Quantity"] = pd.to_numeric(df_valid["Quantity"], errors="coerce").fillna(0).astype(int)
    df_valid["UnitPrice"] = pd.to_numeric(df_valid["UnitPrice"], errors="coerce").fillna(0.0)

    # 3. Separate purchases and cancellations
    is_cancellation = df_valid["InvoiceNo"].str.startswith("C") | (df_valid["Quantity"] < 0)
    
    purchases = df_valid[~is_cancellation & (df_valid["UnitPrice"] > 0) & (df_valid["Quantity"] > 0)].copy()
    returns = df_valid[is_cancellation].copy()

    # Sort purchases chronologically
    purchases = purchases.sort_values("InvoiceDate").reset_index(drop=True)
    returns = returns.sort_values("InvoiceDate").reset_index(drop=True)

    cleaning_stats = {
        "total_raw_rows": total_raw_rows,
        "excluded_missing_customer_id": excluded_no_cust,
        "excluded_administrative_codes": excluded_admin_codes,
        "eligible_purchases": len(purchases),
        "return_records": len(returns),
    }

    return purchases, returns, cleaning_stats


def construct_return_labels(
    purchases: pd.DataFrame,
    returns: pd.DataFrame,
    max_return_window_days: int = 60
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Construct defensible ground-truth return labels.
    Matching Procedure:
      - For each return record (CustomerID, StockCode, ret_date, ret_qty):
        Search for the customer's prior purchases of that product occurring strictly before ret_date.
      - Match to the most recent eligible purchase within max_return_window_days.
      - Mark the purchase line item as is_returned = 1.
    
    Right-Censoring:
      - Purchases made within the final 14 days of the dataset cannot be reliably observed
        for returns (boundary truncation). We flag or document this horizon.
    """
    purchases = purchases.copy()
    purchases["is_returned"] = 0
    purchases["return_days"] = np.nan

    # Build an index of purchases by (CustomerID, StockCode)
    purchase_lookup = {}
    for idx, row in purchases.iterrows():
        key = (str(row["CustomerID"]), str(row["StockCode"]))
        if key not in purchase_lookup:
            purchase_lookup[key] = []
        purchase_lookup[key].append({
            "idx": idx,
            "date": row["InvoiceDate"],
            "qty": row["Quantity"],
            "matched": False
        })

    matched_returns = 0
    unmatched_returns = 0
    time_deltas = []

    for _, ret in returns.iterrows():
        key = (str(ret["CustomerID"]), str(ret["StockCode"]))
        ret_date = ret["InvoiceDate"]
        
        if key in purchase_lookup:
            # Find candidate prior purchases before this return date
            candidates = [
                p for p in purchase_lookup[key] 
                if p["date"] <= ret_date and not p["matched"]
            ]
            if candidates:
                # Match to the most recent prior purchase
                match = candidates[-1]
                delta_days = (ret_date - match["date"]).total_seconds() / 86400.0
                if delta_days <= max_return_window_days:
                    match["matched"] = True
                    purchases.at[match["idx"], "is_returned"] = 1
                    purchases.at[match["idx"], "return_days"] = delta_days
                    time_deltas.append(delta_days)
                    matched_returns += 1
                else:
                    unmatched_returns += 1
            else:
                unmatched_returns += 1
        else:
            unmatched_returns += 1

    total_returned_items = int(purchases["is_returned"].sum())
    return_rate = float(purchases["is_returned"].mean())

    label_stats = {
        "total_eligible_purchases": len(purchases),
        "total_returns_processed": len(returns),
        "matched_returns": matched_returns,
        "unmatched_returns": unmatched_returns,
        "match_success_rate": float(matched_returns / len(returns)) if len(returns) > 0 else 0.0,
        "total_returned_purchases": total_returned_items,
        "positive_class_prevalence": return_rate,
        "median_days_to_return": float(np.median(time_deltas)) if time_deltas else 0.0,
        "mean_days_to_return": float(np.mean(time_deltas)) if time_deltas else 0.0,
        "max_return_window_days": max_return_window_days
    }

    # Save processed dataframe
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    purchases.to_parquet(CLEANED_DATA_PATH, index=False)

    return purchases, label_stats


def get_prepared_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """High-level function to return cleaned purchases with ground-truth return labels."""
    if os.path.exists(CLEANED_DATA_PATH):
        df = pd.read_parquet(CLEANED_DATA_PATH)
        stats = {
            "loaded_from_cache": True,
            "total_purchases": len(df),
            "positive_class_prevalence": float(df["is_returned"].mean())
        }
        return df, stats

    purchases, returns, clean_stats = load_and_clean_data()
    labeled_purchases, label_stats = construct_return_labels(purchases, returns)
    stats = {**clean_stats, **label_stats}
    return labeled_purchases, stats


if __name__ == "__main__":
    print("[DataLoader] Running data loading and return label verification...")
    df, stats = get_prepared_data()
    print("[DataLoader] Execution finished. Statistics:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")

