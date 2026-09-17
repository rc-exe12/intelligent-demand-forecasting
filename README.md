# 🚀 KARIGAR X — Intelligent Demand Forecasting Agent

> **AI-powered demand forecasting and inventory intelligence for smarter business decisions.**

KARIGAR X is an intelligent demand forecasting system designed to analyze historical sales data, identify demand patterns, detect anomalies, forecast future demand, and provide inventory recommendations.

Built as a lightweight and practical solution for businesses that need faster and data-driven inventory decisions.

---

## 🎯 Problem Statement

Businesses often struggle with:

* Unpredictable customer demand
* Overstocking and excess inventory
* Stockouts and lost sales
* Seasonal demand fluctuations
* Sudden sales anomalies
* Manual and time-consuming forecasting

KARIGAR X addresses these challenges by transforming historical sales data into actionable demand and inventory insights.

---

## 💡 Our Solution

KARIGAR X provides an interactive dashboard that allows users to upload sales data and analyze:

**Historical Sales → Demand Patterns → Forecast → Inventory Recommendations**

The system combines statistical forecasting, trend analysis, seasonality detection, anomaly detection, and inventory intelligence into a single dashboard.

---

## ✨ Key Features

### 📊 Demand Analytics

* Analyze historical sales performance
* Identify demand trends
* Detect seasonal patterns
* Compare product performance

### 🔮 Demand Forecasting

* Forecast future product demand
* Generate product-level predictions
* Visualize historical vs predicted demand
* Support data-driven planning

### 📦 Inventory Intelligence

* Monitor inventory levels
* Identify potential stockout risks
* Detect overstock situations
* Generate inventory recommendations

### 🚨 Anomaly Detection

* Detect unusual sales patterns
* Identify sudden demand spikes
* Identify unexpected demand drops
* Highlight potentially abnormal business activity

### 📁 Smart Data Upload

Supports CSV datasets with common column names such as:

```text
Date / order_date / transaction_date
Product / item / product_name
Sales / quantity / units_sold / demand
Inventory / stock / stock_level
```

KARIGAR X automatically maps supported column names into the required format.

---

## 🖥️ Dashboard

The application provides an interactive dashboard containing:

* Dashboard
* Analytics
* Forecast
* Inventory
* Anomalies
* Data
* Settings

The interface is designed with a modern dark, premium-style UI for easy visualization and presentation.

---

## 🛠️ Technology Stack

| Technology   | Purpose                    |
| ------------ | -------------------------- |
| Python       | Core application           |
| Streamlit    | Interactive web dashboard  |
| Pandas       | Data processing            |
| NumPy        | Numerical computation      |
| Statsmodels  | Statistical forecasting    |
| Plotly       | Interactive visualizations |
| Git & GitHub | Version control            |

---

## 📂 Project Structure

```text
intelligent-demand-forecasting/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── sample datasets
│
├── src/
│   ├── anomaly_detection.py
│   ├── data_processing.py
│   ├── inventory.py
│   └── trend_detection.py
│
└── tests/
    └── test files
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/rc-exe12/intelligent-demand-forecasting.git
```

### 2. Open the project

```bash
cd intelligent-demand-forecasting
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows CMD:**

```cmd
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## 📄 Input Data Format

The recommended CSV format is:

| Date       | Product   | Sales | Inventory |
| ---------- | --------- | ----: | --------: |
| 2026-01-01 | Product A |   120 |       500 |
| 2026-01-02 | Product A |   135 |       365 |
| 2026-01-03 | Product A |   128 |       237 |

### Required Columns

```text
Date
Product
Sales
```

### Optional Column

```text
Inventory
```

---

## 🔄 How KARIGAR X Works

```text
          CSV DATA
              │
              ▼
      ┌─────────────────┐
      │ Data Processing  │
      └────────┬────────┘
    │
               ▼
      ┌─────────────────┐
      │ Trend Detection │
      └────────┬────────┘
               │
               ▼
      ┌─────────────────┐
      │ Forecast Engine │
      └────────┬────────┘
               │
        ┌──────┴───────┐
        ▼              ▼
   Anomaly         Inventory
   Detection      Intelligence
        │              │
        └──────┬───────┘
               ▼
       ┌────────────────┐
       │   Dashboard    │
       └────────────────┘
🔮 Forecasting Approach

KARIGAR X uses statistical time-series analysis to identify demand patterns from historical sales data.

The forecasting pipeline considers:

Historical demand
Trend
Seasonality
Recent demand behavior
Anomalies
Product-level sales patterns

This approach keeps the system lightweight and suitable for rapid deployment without requiring GPU-based deep learning models.

📈 Business Impact

KARIGAR X can help businesses:

Reduce unnecessary inventory
Identify potential stockouts
Improve demand planning
Understand customer demand patterns
Make faster inventory decisions
Detect unusual sales behavior
🚀 Future Scope

Future versions can include:

Real-time demand forecasting
Live POS integration
Automated purchase-order recommendations
Multi-store forecasting
Advanced machine-learning models
Deep-learning forecasting
Mobile application
Multilingual support
Cloud-based data synchronization
👥 Team

Codeathon Project — Intelligent Demand Forecasting

Built by the KARIGAR X team.

📜 License

This project is developed for educational and demonstration purposes.
