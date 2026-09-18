"""Run EDA, RFM segmentation, a simple monthly-revenue model, and a Markdown report."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:  # Lets the report pipeline run in a minimal environment.
    SKLEARN_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_raw_data
from src.data_cleaning import clean_ecommerce_data
from src.visualization import save_standard_figures


def rfm_segments(data: pd.DataFrame) -> pd.DataFrame:
    """Create transparent quartile-based RFM customer segments."""
    if "customer_id" not in data.columns:
        return pd.DataFrame()
    snapshot = data["order_date"].max() + pd.Timedelta(days=1)
    rfm = data.groupby("customer_id").agg(
        recency_days=("order_date", lambda x: (snapshot - x.max()).days),
        frequency=("order_id", "nunique"), monetary=("revenue", "sum")
    ).reset_index()
    rfm["r_score"] = pd.qcut(rfm["recency_days"].rank(method="first", ascending=False), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    score = rfm[["r_score", "f_score", "m_score"]].sum(axis=1)
    rfm["segment"] = np.select([score >= 10, (rfm.r_score >= 3) & (rfm.f_score <= 2), score <= 6], ["High-value", "Recent", "At-risk"], default="Loyal")
    return rfm


def train_monthly_model(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    monthly = data.groupby("year_month", as_index=False).agg(revenue=("revenue", "sum"))
    monthly["period"] = pd.to_datetime(monthly["year_month"] + "-01")
    monthly["time_index"] = np.arange(len(monthly))
    if len(monthly) < 12:
        return monthly, {"available": False, "reason": "Fewer than 12 monthly observations."}
    split = max(6, int(len(monthly) * .8))
    train, test = monthly.iloc[:split], monthly.iloc[split:]
    if SKLEARN_AVAILABLE:
        model = LinearRegression().fit(train[["time_index"]], train["revenue"])
        prediction = model.predict(test[["time_index"]])
        fitted = model.predict(monthly[["time_index"]])
        implementation = "scikit-learn LinearRegression"
    else:
        slope, intercept = np.polyfit(train["time_index"], train["revenue"], 1)
        prediction = intercept + slope * test["time_index"].to_numpy()
        fitted = intercept + slope * monthly["time_index"].to_numpy()
        implementation = "NumPy least-squares fallback (install scikit-learn for the requested implementation)"
    monthly["predicted_revenue"] = fitted
    residuals = test.revenue.to_numpy() - prediction
    total = ((test.revenue - test.revenue.mean()) ** 2).sum()
    r_squared = 1 - (residuals ** 2).sum() / total if total else np.nan
    return monthly, {"available": True, "implementation": implementation, "train_months": len(train), "test_months": len(test), "mae": float(np.abs(residuals).mean()), "rmse": float(np.sqrt((residuals ** 2).mean())), "r2": float(r_squared)}


def build_metrics(data: pd.DataFrame, rfm: pd.DataFrame) -> dict:
    orders = data.groupby("order_id", as_index=False).agg(revenue=("revenue", "sum"))
    category = data.groupby("category", as_index=False).agg(revenue=("revenue", "sum"), profit=("profit", "sum")).sort_values("revenue", ascending=False)
    product = data.groupby("product_name", as_index=False).agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"), profit=("profit", "sum")).sort_values("revenue", ascending=False)
    geo_col = "country" if "country" in data.columns else "region"
    geography = data.groupby(geo_col, as_index=False).agg(revenue=("revenue", "sum"), profit=("profit", "sum")).sort_values("revenue", ascending=False)
    corr_data = data[["discount", "revenue", "profit", "quantity"]].dropna()
    corr, p_value = pearsonr(corr_data["discount"], corr_data["profit"])
    repeat_share = None
    if "customer_id" in data.columns:
        customer_orders = data.groupby("customer_id")["order_id"].nunique()
        repeat_share = float((customer_orders > 1).mean())
    return {
        "total_revenue": float(data.revenue.sum()), "total_profit": float(data.profit.sum()), "total_orders": int(data.order_id.nunique()),
        "total_customers": int(data.customer_id.nunique()) if "customer_id" in data else None, "total_quantity": int(data.quantity.sum()),
        "average_order_value": float(orders.revenue.mean()), "average_revenue_per_customer": float(data.revenue.sum() / data.customer_id.nunique()) if "customer_id" in data else None,
        "best_category": category.iloc[0].to_dict(), "top_product": product.iloc[0].to_dict(), "top_geography": geography.iloc[0].to_dict(),
        "discount_profit_correlation": float(corr), "discount_profit_pvalue": float(p_value), "repeat_customer_share": repeat_share,
        "high_value_customers": int((rfm.segment == "High-value").sum()) if not rfm.empty else None,
        "category_table": category.round(2).to_dict("records"), "top_products": product.head(10).round(2).to_dict("records"),
    }


def write_report(data: pd.DataFrame, audit: dict, metrics: dict, model: dict, path: Path) -> None:
    cat = metrics["best_category"]; product = metrics["top_product"]; geo = metrics["top_geography"]
    share = cat["revenue"] / metrics["total_revenue"] * 100
    repeat = "n.a." if metrics["repeat_customer_share"] is None else f"{metrics['repeat_customer_share']:.1%}"
    forecast = "The monthly trend model was not run because there were insufficient periods." if not model["available"] else f"A linear regression using a monthly time index was evaluated on {model['test_months']} held-out months (MAE {model['mae']:,.0f}, RMSE {model['rmse']:,.0f}, R² {model['r2']:.3f})."
    text = f"""# Analysis of Ecommerce Sales Data

## Abstract
This project analyzes {len(data):,} ecommerce transaction lines from {audit['date_min']} to {audit['date_max']}. The workflow cleans transaction data, measures sales and profit performance, examines customer and product behavior, and tests a beginner-friendly monthly revenue model.

## Introduction and problem statement
The goal is to turn order-line data into evidence for product, customer, geographic, and pricing decisions. The analysis treats each row as a product line, so revenue and profit are aggregated carefully at order, customer, product, and time levels.

## Objectives
- Identify revenue, order, quantity, and profit patterns.
- Assess product, category, customer, and geographic performance.
- Examine the association between discounts and profit without claiming causation.
- Provide a transparent baseline forecast and actionable, data-backed recommendations.

## Dataset description and collection
**Dataset:** Global Superstore Dataset. **Owner:** Rony Soliman. **Source:** https://www.kaggle.com/datasets/ronysoliman/global-superstore-dataset/data . Kaggle describes it as global retail order-line data with customer, product, geographic, sales, discount, and profit fields. The downloaded source file contains 51,290 rows and 27 columns.

## Data cleaning
Headers were standardized to snake_case. Exact duplicate rows were removed ({audit['exact_duplicates_removed']:,}); rows missing an essential identifier/date or with non-positive sales or quantity were excluded ({audit['invalid_rows_removed']:,}). Dates and numeric fields were typed. Derived features include year, month, quarter, weekday, year-month, revenue, profit margin, average order value, and customer order frequency. Large valid orders were retained rather than automatically removed as outliers.

## Exploratory data analysis
- **Revenue:** {metrics['total_revenue']:,.2f}
- **Profit:** {metrics['total_profit']:,.2f}
- **Orders:** {metrics['total_orders']:,}; **units:** {metrics['total_quantity']:,}
- **Average order value:** {metrics['average_order_value']:,.2f}
- **Customers:** {metrics['total_customers']:,}; **average revenue per customer:** {metrics['average_revenue_per_customer']:,.2f}

## Product analysis
{cat['category']} was the largest revenue category at {cat['revenue']:,.2f}, or {share:.1f}% of total revenue. The highest-revenue product was {product['product_name']} at {product['revenue']:,.2f}. Category and product tables are saved in `reports/summary_metrics.json`.

## Customer analysis
RFM uses the latest transaction date as the reference: recency is days since last purchase, frequency is distinct orders, and monetary value is customer revenue. Quartile scores create High-value, Loyal, Recent, and At-risk segments. The share of customers with more than one order was {repeat}.

## Geographic analysis
The leading {('country' if 'country' in data.columns else 'region')} was {geo[('country' if 'country' in data.columns else 'region')]} with revenue of {geo['revenue']:,.2f} and profit of {geo['profit']:,.2f}.

## Statistical analysis
Pearson correlation between discount and profit is {metrics['discount_profit_correlation']:.3f} (p={metrics['discount_profit_pvalue']:.3g}). This measures linear association in transaction lines; it does not establish that discounts cause profit changes because product mix, geography, and order size can also affect profit.

## Predictive analysis
{forecast} The model is a baseline trend extrapolation, not a guarantee of future revenue. It does not include promotions, product mix, holidays, or external demand drivers.

## Key findings and recommendations
1. **Finding:** {cat['category']} leads revenue at {cat['revenue']:,.2f}. **Interpretation:** It is a core demand driver. **Action:** Protect availability and review its highest-margin sub-categories before allocating incremental marketing.
2. **Finding:** {product['product_name']} leads product revenue at {product['revenue']:,.2f}. **Interpretation:** A small set of products may anchor demand. **Action:** Check replenishment, bundles, and margin before expanding promotions.
3. **Finding:** Discount-profit correlation is {metrics['discount_profit_correlation']:.3f}. **Interpretation:** The relationship warrants segment-level review, not a causal conclusion. **Action:** Set discount guardrails by category and monitor margin alongside revenue.
4. **Finding:** {geo[('country' if 'country' in data.columns else 'region')]} is the largest geography by revenue. **Interpretation:** It is an important commercial area. **Action:** Compare profit margin and repeat purchase rates before prioritizing regional expansion.

## Limitations
The source has no payment-method field and is order-line rather than a complete customer lifecycle record. RFM reflects observed purchase history only. The trend model is intentionally simple and should be enhanced before operational use.

## Conclusion and future scope
The project provides a reproducible starting point for ecommerce performance analysis. Future work could add promotion calendars, returns, acquisition costs, inventory availability, and a seasonality-aware forecasting model.

## References
- Rony Soliman, *Global Superstore Dataset*, Kaggle: https://www.kaggle.com/datasets/ronysoliman/global-superstore-dataset/data
- pandas, scikit-learn, SciPy, Matplotlib, Seaborn, Plotly, and Streamlit documentation.
"""
    path.write_text(text, encoding="utf-8")


def run_pipeline(raw_path: str | Path = ROOT / "data/raw/superstore.csv") -> dict:
    data, audit = clean_ecommerce_data(load_raw_data(raw_path))
    processed = ROOT / "data/processed/cleaned_ecommerce_data.csv"; processed.parent.mkdir(parents=True, exist_ok=True); data.to_csv(processed, index=False)
    rfm = rfm_segments(data); rfm.to_csv(ROOT / "data/processed/rfm_customer_segments.csv", index=False)
    monthly, model = train_monthly_model(data); monthly.to_csv(ROOT / "data/processed/monthly_revenue_model.csv", index=False)
    metrics = build_metrics(data, rfm)
    (ROOT / "reports").mkdir(exist_ok=True); (ROOT / "reports/summary_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ROOT / "reports/model_metrics.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    save_standard_figures(data, ROOT / "reports/figures")
    write_report(data, audit, metrics, model, ROOT / "reports/insights.md")
    return {"audit": audit, "metrics": metrics, "model": model}


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps({"rows": result["audit"]["output_rows"], "revenue": result["metrics"]["total_revenue"], "orders": result["metrics"]["total_orders"]}, indent=2))
