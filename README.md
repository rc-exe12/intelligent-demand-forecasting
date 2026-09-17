# Intelligent Demand Forecasting Agent

Lightweight Streamlit MVP that predicts product demand from historical sales and recommends inventory.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use the sample file at `data/sample_sales.csv` or upload your own CSV.

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
