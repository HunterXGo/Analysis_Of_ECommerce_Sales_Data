"""Load an ecommerce CSV and standardize commonly used column names."""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

ALIASES = {
    "customer.id": "customer_id", "customer id": "customer_id",
    "customer.name": "customer_name", "customer name": "customer_name",
    "order.id": "order_id", "order id": "order_id",
    "order.date": "order_date", "order date": "order_date",
    "product.id": "product_id", "product id": "product_id",
    "product.name": "product_name", "product name": "product_name",
    "sub.category": "sub_category", "sub category": "sub_category",
    "ship.date": "ship_date", "ship date": "ship_date",
    "ship.mode": "ship_mode", "ship mode": "ship_mode",
    "shipping.cost": "shipping_cost", "shipping cost": "shipping_cost",
    "row.id": "row_id", "row id": "row_id",
}

REQUIRED = {"order_id", "order_date", "product_name", "category", "sales", "quantity", "profit"}


def _normalise(name: str) -> str:
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read CSV/XLSX data, standardize headers, and validate core analysis fields."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path)
    frame.columns = [_normalise(col) for col in frame.columns]
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(sorted(missing)))
    return frame

