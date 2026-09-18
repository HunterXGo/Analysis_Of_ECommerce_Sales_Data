"""Interactive Streamlit dashboard for the cleaned ecommerce dataset."""
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis import run_pipeline

st.set_page_config(page_title="Ecommerce Sales Analysis", page_icon="🛒", layout="wide")

@st.cache_data
def get_data():
    file = ROOT / "data/processed/cleaned_ecommerce_data.csv"
    if not file.exists(): run_pipeline()
    data = pd.read_csv(file, parse_dates=["order_date"])
    return data

data = get_data()
st.title("Ecommerce Sales Analysis")
st.caption("Interactive exploration of Global Superstore order-line data")
with st.sidebar:
    st.header("Filters")
    dates = st.date_input("Order date", value=(data.order_date.min().date(), data.order_date.max().date()), min_value=data.order_date.min().date(), max_value=data.order_date.max().date())
    def choose(label, col): return st.multiselect(label, sorted(data[col].dropna().unique()), default=sorted(data[col].dropna().unique()))
    categories = choose("Category", "category")
    subcategories = choose("Sub-category", "sub_category") if "sub_category" in data else []
    regions = choose("Region", "region") if "region" in data else []
    segments = choose("Customer segment", "segment") if "segment" in data else []

filtered = data[(data.order_date.dt.date >= dates[0]) & (data.order_date.dt.date <= dates[-1]) & data.category.isin(categories)]
if subcategories: filtered = filtered[filtered.sub_category.isin(subcategories)]
if regions: filtered = filtered[filtered.region.isin(regions)]
if segments: filtered = filtered[filtered.segment.isin(segments)]
orders = filtered.groupby("order_id").revenue.sum()
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Total revenue", f"${filtered.revenue.sum():,.0f}")
c2.metric("Total profit", f"${filtered.profit.sum():,.0f}")
c3.metric("Total orders", f"{filtered.order_id.nunique():,}")
c4.metric("Total customers", f"{filtered.customer_id.nunique():,}")
c5.metric("Average order value", f"${orders.mean():,.0f}" if len(orders) else "n.a.")
monthly = filtered.groupby(filtered.order_date.dt.to_period("M").astype(str), as_index=False).revenue.sum()
category = filtered.groupby("category", as_index=False)[["revenue", "profit"]].sum().sort_values("revenue", ascending=False)
top = filtered.groupby("product_name", as_index=False).revenue.sum().nlargest(10, "revenue")
left,right=st.columns(2)
left.plotly_chart(px.line(monthly, x="order_date", y="revenue", markers=True, title="Revenue trend"), use_container_width=True)
right.plotly_chart(px.bar(category, x="category", y="revenue", title="Revenue by category"), use_container_width=True)
left,right=st.columns(2)
left.plotly_chart(px.bar(category, x="category", y="profit", title="Profit by category"), use_container_width=True)
right.plotly_chart(px.bar(top, x="revenue", y="product_name", orientation="h", title="Top products by revenue"), use_container_width=True)
if "segment" in filtered:
    st.plotly_chart(px.bar(filtered.groupby("segment", as_index=False).revenue.sum(), x="segment", y="revenue", title="Revenue by customer segment"), use_container_width=True)
if "country" in filtered:
    st.plotly_chart(px.bar(filtered.groupby("country", as_index=False)[["revenue", "profit"]].sum().nlargest(15, "revenue"), x="country", y="revenue", title="Top countries by revenue"), use_container_width=True)
st.plotly_chart(px.scatter(filtered.sample(min(5000, len(filtered)), random_state=42), x="discount", y="profit", color="category", hover_data=["product_name"], title="Discount versus profit"), use_container_width=True)

