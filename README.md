# 🚀 DemandAI — Intelligent Demand Forecasting Agent

> **Turn historical sales into smarter inventory decisions.**

DemandAI is an intelligent demand forecasting and inventory analytics application that analyzes historical sales data, identifies demand patterns, detects unusual activity, forecasts future demand, and recommends inventory replenishment.

The application is designed as a lightweight, explainable AI/analytics solution for businesses that need to make better inventory decisions from historical sales data.

---

## 🎯 Problem Statement

Businesses often struggle to maintain the right amount of inventory.

* **Too much inventory** → increased storage costs and unsold products
* **Too little inventory** → stockouts and missed sales
* **Unpredictable demand** → difficult purchasing and planning decisions
* **Sales anomalies** → unusual spikes or drops can distort forecasts

Traditional reporting systems mainly explain **what happened**.

DemandAI goes one step further by helping answer:

> **"What is likely to happen next, and how much inventory should we maintain?"**

---

## 💡 Our Solution

DemandAI processes historical sales data and converts it into actionable demand intelligence.

```text
Historical Sales Data
        ↓
   Data Processing
        ↓
 ┌──────┼──────────┐
 ↓      ↓          ↓
Trend  Seasonality Anomalies
 └──────┼──────────┘
        ↓
 Demand Forecasting
        ↓
 Inventory Analysis
        ↓
 Recommended Reorder
        ↓
 Actionable Insights
```

---

# ✨ Key Features

## 📊 1. Sales Analytics

Analyze historical sales and demand patterns.

**Provides:**

* Total sales
* Average daily demand
* Daily/weekly/monthly sales analysis
* Sales trends
* Moving averages
* Weekday demand patterns
* Dataset statistics

---

## 🔮 2. Demand Forecasting

Forecast future product demand using historical sales data.

**Provides:**

* 7-day forecast
* 14-day forecast
* 30-day forecast
* Interactive forecast visualization
* Forecast table
* Forecast demand summary
* Confidence/range visualization where supported

The forecasting pipeline uses lightweight time-series techniques such as **Exponential Smoothing / Holt-Winters**, with fallback approaches when the available data is insufficient for the primary model.

---

## 📈 3. Trend Detection

Automatically identifies the direction of demand.

Possible results include:

* Increasing ↗
* Decreasing ↘
* Stable →

The trend is calculated from historical demand rather than manually entered values.

---

## 🔄 4. Seasonality Detection

Identifies recurring demand patterns.

For example:

> Saturday may have consistently higher demand than other days.

The application can analyze:

* Day-of-week demand
* Weekly patterns
* Seasonal behavior where sufficient historical data exists
* Peak demand periods

---

## 🚨 5. Anomaly Detection

Detect unusual demand spikes and drops.

Supported approaches include:

* IQR-based detection
* Z-score detection

Detected anomalies can be highlighted in the demand visualization and analyzed separately.

Example:

```text
Date          Actual Demand     Status
-----------------------------------------
Sep 02        301               Spike
Sep 08         18               Drop
Sep 14        245               Spike
```

---

## 📦 6. Inventory Intelligence

DemandAI combines predicted demand with current inventory to provide a replenishment recommendation.

### Recommended Stock

```text
Recommended Stock
= Forecast Demand + Safety Stock
```

### Recommended Order

```text
Order Quantity
= max(0, Recommended Stock - Current Inventory)
```

This transforms demand forecasting into an actionable inventory decision.

---

## 🤖 7. Demand Intelligence

DemandAI converts analytical results into a concise business summary.

Example:

> Demand is showing an upward trend with stronger weekend activity. Several unusual demand events were detected. Based on projected demand and current inventory, additional stock may be required.

The summary is generated from the application's analytical results and does not require an external LLM API.

---

# 🖥️ Application Modules

DemandAI is organized into the following sections:

| Module       | Purpose                                 |
| ------------ | --------------------------------------- |
| 🏠 Dashboard | Overall demand and inventory overview   |
| 📊 Analytics | Historical sales analysis               |
| 🔮 Forecast  | Future demand prediction                |
| 📦 Inventory | Stock and replenishment recommendations |
| 🚨 Anomalies | Unusual demand detection                |
| 📁 Data      | Dataset upload and validation           |
| ⚙️ Settings  | Forecast and detection configuration    |

---

# 📁 Input Dataset

DemandAI accepts historical sales data in **CSV format**.

A typical dataset can contain:

```csv
date,product,sales,inventory
2025-01-01,Alpha Headphones,82,450
2025-01-02,Alpha Headphones,91,359
2025-01-03,Alpha Headphones,88,271
2025-01-04,Alpha Headphones,105,166
```

### Expected Information

The application requires appropriate columns for:

* Date
* Sales/Demand
* Product, when multiple products are supported
* Current inventory, when inventory recommendations are required

Column names may depend on the project's data-processing implementation.

---

# 🧠 Technology Stack

### Programming

* Python 3.11

### Data Processing

* Pandas
* NumPy

### Forecasting & Statistics

* Statsmodels
* Exponential Smoothing / Holt-Winters
* Statistical anomaly detection

### Visualization

* Plotly

### Application

* Streamlit

### Development

* Git
* GitHub
* VS Code

---

# 🏗️ Project Architecture

```text
DemandAI/
│
├── app.py
│
├── forecast.py
├── data_processing.py
├── trend_detection.py
├── anomaly_detection.py
├── inventory.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

> File names may vary depending on the current implementation.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd intelligent-demand-forecasting-updated
```

---

## 2. Create a Python virtual environment

DemandAI is developed and tested with **Python 3.11**.

### Windows

```powershell
py -3.11 -m venv .venv
```

If Python 3.11 is installed through another Python manager, use its executable directly.

---

## 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Then:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Streamlit application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 🧪 Testing

Run the project's automated tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

If the project uses pytest:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Testing should cover:

* Data loading
* Data validation
* Data processing
* Trend detection
* Seasonality analysis
* Anomaly detection
* Forecast generation
* Inventory calculations

---

# 🔍 How It Works

## Step 1 — Upload Data

The user uploads a historical sales CSV.

## Step 2 — Validate & Process

DemandAI validates the dataset and prepares the data for analysis.

## Step 3 — Analyze Demand

The system calculates:

* Average demand
* Trends
* Seasonal patterns
* Anomalies

## Step 4 — Forecast

The forecasting engine predicts future demand for the selected product and time horizon.

## Step 5 — Analyze Inventory

The forecast is combined with current stock and safety stock.

## Step 6 — Recommend Action

DemandAI calculates the recommended inventory level and replenishment quantity.

## Step 7 — Visualize

The results are presented through an interactive dashboard.

---

# 📈 Example Workflow

```text
User uploads sales.csv
          ↓
Select Product
          ↓
Select Forecast Horizon
          ↓
Generate Forecast
          ↓
Historical Demand ────────► Forecast
          │
          ├──────────────► Trend
          │
          ├──────────────► Seasonality
          │
          └──────────────► Anomalies
                            │
                            ↓
                     Inventory Engine
                            │
                            ↓
                    Recommended Order
```

---

# 🎯 Example Business Scenario

Imagine a retailer selling headphones.

Current inventory:

```text
344 units
```

Expected demand for the next 14 days:

```text
1,330 units
```

Safety stock:

```text
150 units
```

Recommended stock:

```text
1,480 units
```

Therefore:

```text
Recommended Order
= 1,480 - 344
= 1,136 units
```

The dashboard presents this recommendation together with the underlying demand analysis.

---

# 🛡️ Design Principles

DemandAI follows several principles:

### Lightweight

The application avoids unnecessarily complex infrastructure and large deep-learning models for the MVP.

### Explainable

Forecasting and inventory recommendations are based on understandable statistical calculations.

### Action-Oriented

The goal is not only to predict demand but also to help answer:

> **"What should I do with my inventory?"**

### Data-Driven

Dashboard metrics should be calculated from the selected dataset rather than hardcoded values.

### Fault-Tolerant

The application should gracefully handle:

* Missing values
* Invalid CSV files
* Insufficient historical data
* Missing columns
* Unexpected input formats

---

# 🚀 Future Enhancements

Potential future improvements include:

* Real-time demand streaming
* Advanced forecasting models
* Product-level automated forecasting
* Holiday and festival effects
* Weather-based demand signals
* Supplier lead-time integration
* Automated purchase-order generation
* Multi-location inventory optimization
* Mobile application
* Cloud deployment
* Advanced model evaluation
* Automated model selection
* Explainable forecast confidence scoring

---

# 👥 Team

**Project:** DemandAI — Intelligent Demand Forecasting Agent

Developed as a collaborative engineering project.

### Team Members

* Ramcharan Sai

### Institution

**Geetanjali College of Engineering and Technology**

**Department:** Freshman Engineering

**Academic Year:** 2025–2026

---

# 🏆 Project Vision

DemandAI aims to bridge the gap between **historical data, demand prediction, and inventory decisions**.

Instead of simply asking:

> *"What happened?"*

DemandAI helps businesses ask:

> **"What is likely to happen next, and what should we stock?"**

---

## 📄 License

This project is developed for educational, demonstration, and hackathon purposes.

---

## ⭐ DemandAI

**Analyze. Predict. Decide.**
