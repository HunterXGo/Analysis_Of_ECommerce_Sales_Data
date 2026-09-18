"""Cleaning and feature engineering functions for ecommerce transactions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_ecommerce_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean fields conservatively and add analysis-ready derived columns.

    A transaction line is retained unless it lacks an essential identifier/date or has
    an impossible non-positive sales/quantity value. Extreme but valid sales values
    are retained because high-value orders can be genuine business events.
    """
    data = frame.copy()
    audit = {"input_rows": len(data), "exact_duplicates_removed": 0, "invalid_rows_removed": 0}
    data = data.drop_duplicates().copy()
    audit["exact_duplicates_removed"] = audit["input_rows"] - len(data)

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype("string").str.strip()
        data.loc[data[col].isin(["", "nan", "None"]), col] = pd.NA

    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    if "ship_date" in data.columns:
        data["ship_date"] = pd.to_datetime(data["ship_date"], errors="coerce")
    for col in ["sales", "profit", "discount", "quantity", "shipping_cost"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    essential = ["order_id", "order_date", "product_name", "category", "sales", "quantity", "profit"]
    valid = data[essential].notna().all(axis=1) & (data["sales"] > 0) & (data["quantity"] > 0)
    audit["invalid_rows_removed"] = int((~valid).sum())
    data = data.loc[valid].copy()
    data["discount"] = data.get("discount", 0).fillna(0).clip(lower=0)
    data["year"] = data["order_date"].dt.year
    data["month"] = data["order_date"].dt.month
    data["month_name"] = data["order_date"].dt.month_name()
    data["quarter"] = "Q" + data["order_date"].dt.quarter.astype(str)
    data["day_of_week"] = data["order_date"].dt.day_name()
    data["year_month"] = data["order_date"].dt.to_period("M").astype(str)
    data["revenue"] = data["sales"]
    data["profit_margin"] = np.where(data["sales"] != 0, data["profit"] / data["sales"], np.nan)
    order_totals = data.groupby("order_id")["revenue"].transform("sum")
    data["average_order_value"] = order_totals
    if "customer_id" in data.columns:
        data["order_frequency"] = data.groupby("customer_id")["order_id"].transform("nunique")
    audit["output_rows"] = len(data)
    audit["date_min"] = str(data["order_date"].min().date())
    audit["date_max"] = str(data["order_date"].max().date())
    return data, audit

