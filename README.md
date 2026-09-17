# Intelligent Demand Forecasting Agent (DemandAI)

Lightweight Streamlit MVP that predicts product demand from historical sales and recommends inventory, presented as a branded BI-style dashboard.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use the sample file at `data/sample_sales.csv` or upload your own CSV.

## Dashboard

The sidebar has 7 sections, all driven by the same underlying calculation:

- **Dashboard** – KPIs, historical sales + forecast chart, insights, inventory panel, weekday/trend charts, AI summary, forecast table
- **Analytics** – trend and seasonality detail
- **Forecast** – forecast chart, table, CSV download
- **Inventory** – recommended stock/order with the formula breakdown
- **Anomalies** – anomaly chart and flagged-day table
- **Data** – data-quality warnings and cleaned data preview/download
- **Settings** – switch anomaly detection method (IQR/Z-score) and safety-stock mode (auto/manual)

Product and forecast horizon are chosen at the top and applied when **Generate Forecast** is clicked.

## Expected CSV columns

- **Date** (required)
- **Product** (required)
- **Sales** (required)
- **Inventory** (optional; used for recommendations)
- **Price** (optional)

The app warns about missing columns, invalid dates, missing/negative sales, duplicates, and missing inventory.

## Pipeline

CSV → cleaning → historical analysis → trend/seasonality → anomaly detection → forecast → inventory recommendation → dashboard

Forecasting uses Holt-Winters exponential smoothing, then Simple Exponential Smoothing, then a moving average if a model fails.

Inventory formulas:

- Recommended stock = forecast demand + safety stock
- Order quantity = max(0, recommended stock − current inventory)

Recommendations are decision support only.
