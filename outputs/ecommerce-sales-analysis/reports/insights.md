# Analysis of Ecommerce Sales Data

## Abstract
This project analyzes 51,289 ecommerce transaction lines from 2011-01-01 to 2014-12-31. The workflow cleans transaction data, measures sales and profit performance, examines customer and product behavior, and tests a beginner-friendly monthly revenue model.

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
Headers were standardized to snake_case. Exact duplicate rows were removed (0); rows missing an essential identifier/date or with non-positive sales or quantity were excluded (1). Dates and numeric fields were typed. Derived features include year, month, quarter, weekday, year-month, revenue, profit margin, average order value, and customer order frequency. Large valid orders were retained rather than automatically removed as outliers.

## Exploratory data analysis
- **Revenue:** 12,642,905.00
- **Profit:** 1,467,458.40
- **Orders:** 25,035; **units:** 178,311
- **Average order value:** 505.01
- **Customers:** 4,873; **average revenue per customer:** 2,594.48

## Product analysis
Technology was the largest revenue category at 4,744,691.00, or 37.5% of total revenue. The highest-revenue product was Apple Smart Phone, Full Size at 86,936.00. Category and product tables are saved in `reports/summary_metrics.json`.

## Customer analysis
RFM uses the latest transaction date as the reference: recency is days since last purchase, frequency is distinct orders, and monetary value is customer revenue. Quartile scores create High-value, Loyal, Recent, and At-risk segments. The share of customers with more than one order was 91.3%.

## Geographic analysis
The leading country was United States with revenue of 2,297,354.00 and profit of 286,398.13.

## Statistical analysis
Pearson correlation between discount and profit is -0.317 (p=0). This measures linear association in transaction lines; it does not establish that discounts cause profit changes because product mix, geography, and order size can also affect profit.

## Predictive analysis
A linear regression using a monthly time index was evaluated on 10 held-out months (MAE 97,409, RMSE 106,566, R² 0.049). The model is a baseline trend extrapolation, not a guarantee of future revenue. It does not include promotions, product mix, holidays, or external demand drivers.

## Key findings and recommendations
1. **Finding:** Technology leads revenue at 4,744,691.00. **Interpretation:** It is a core demand driver. **Action:** Protect availability and review its highest-margin sub-categories before allocating incremental marketing.
2. **Finding:** Apple Smart Phone, Full Size leads product revenue at 86,936.00. **Interpretation:** A small set of products may anchor demand. **Action:** Check replenishment, bundles, and margin before expanding promotions.
3. **Finding:** Discount-profit correlation is -0.317. **Interpretation:** The relationship warrants segment-level review, not a causal conclusion. **Action:** Set discount guardrails by category and monitor margin alongside revenue.
4. **Finding:** United States is the largest geography by revenue. **Interpretation:** It is an important commercial area. **Action:** Compare profit margin and repeat purchase rates before prioritizing regional expansion.

## Limitations
The source has no payment-method field and is order-line rather than a complete customer lifecycle record. RFM reflects observed purchase history only. The trend model is intentionally simple and should be enhanced before operational use.

## Conclusion and future scope
The project provides a reproducible starting point for ecommerce performance analysis. Future work could add promotion calendars, returns, acquisition costs, inventory availability, and a seasonality-aware forecasting model.

## References
- Rony Soliman, *Global Superstore Dataset*, Kaggle: https://www.kaggle.com/datasets/ronysoliman/global-superstore-dataset/data
- pandas, scikit-learn, SciPy, Matplotlib, Seaborn, Plotly, and Streamlit documentation.
