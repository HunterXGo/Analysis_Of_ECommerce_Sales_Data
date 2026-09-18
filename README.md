# Analysis of Ecommerce Sales Data

## Overview
This engineering-college Data Science project analyzes global ecommerce order-line data to identify sales patterns, customer purchasing behavior, revenue trends, product performance, geographic opportunities, and discount-profit relationships. It follows a complete workflow: acquisition, data understanding, careful cleaning, EDA, RFM segmentation, a baseline predictive model, a written report, and a Streamlit dashboard.

## Dataset source
- **Dataset:** Global Superstore Dataset
- **Kaggle owner:** Rony Soliman
- **Source URL:** https://www.kaggle.com/datasets/ronysoliman/global-superstore-dataset/data
- **Description:** Global retail order-line data with customer, product, segment, geography, discount, sales, quantity, and profit fields.
- **Downloaded file:** `data/raw/superstore.csv` (51,290 rows and 27 source columns)

This project uses a real, publicly available dataset. The dataset lacks payment-method data, so that field is not analyzed.

## Objectives
- Measure revenue, order volume, quantity, profit, and average order value.
- Analyze sales over time, by product hierarchy, customer segment, and geography.
- Segment customers with RFM (recency, frequency, monetary value).
- Examine discount and profit association without interpreting correlation as causation.
- Evaluate a simple monthly revenue trend model.
- Present actual, reproducible findings in a report and dashboard.

## Project structure
```
ecommerce-sales-analysis/
├── data/
│   ├── raw/superstore.csv
│   └── processed/
├── notebooks/ecommerce_sales_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── analysis.py
│   └── visualization.py
├── dashboard/app.py
├── reports/insights.md
├── requirements.txt
└── README.md
```

## Technologies
Python, pandas, NumPy, Matplotlib, Seaborn, Plotly, SciPy, scikit-learn, Jupyter Notebook, and Streamlit.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run the project
The raw dataset is included at `data/raw/superstore.csv`.

```bash
python -m src.analysis
jupyter notebook notebooks/ecommerce_sales_analysis.ipynb
streamlit run dashboard/app.py
```

If the raw file is not present, download **Global Superstore Dataset** from the Kaggle URL above and place `superstore.csv` in `data/raw/`. The loader accepts CSV, XLS, and XLSX files, but the source should retain the documented transaction fields.

## Main outputs
- `data/processed/cleaned_ecommerce_data.csv`: cleaned, feature-engineered transaction lines.
- `data/processed/rfm_customer_segments.csv`: customer RFM score and segment.
- `data/processed/monthly_revenue_model.csv`: actual and baseline trend-model revenue.
- `reports/figures/`: EDA graphics.
- `reports/insights.md`: complete college-level final report, populated from actual analysis results.
- `reports/summary_metrics.json` and `reports/model_metrics.json`: reproducible metrics.

## Method notes
Valid high-value transactions are retained; the workflow does not blindly remove outliers. RFM uses the latest observed order date as the snapshot date and quartile scores for recency, frequency, and monetary value. The prediction is a beginner-friendly linear regression on monthly time index; it is a baseline and not a guaranteed forecast.

## Screenshots
After running the dashboard, capture screenshots of its KPI cards and interactive charts for a presentation or project report.

## Future improvements
- Add returns, acquisition costs, inventory, and campaign data.
- Model seasonality and holiday effects.
- Add customer lifetime value and retention cohorts.
- Deploy the dashboard to Streamlit Community Cloud.

## References
- Rony Soliman, *Global Superstore Dataset*, Kaggle: https://www.kaggle.com/datasets/ronysoliman/global-superstore-dataset/data
- pandas, scikit-learn, SciPy, Matplotlib, Seaborn, Plotly, and Streamlit documentation.
