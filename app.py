"""Streamlit dashboard for the Intelligent Demand Forecasting Agent."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_detection import detect_anomalies, winsorize_for_forecast
from src.data_processing import (
    daily_series,
    filter_product,
    historical_metrics,
    latest_inventory,
    products,
    validate_and_clean,
)
from src.forecasting import forecast_demand
from src.inventory import recommend_inventory
from src.seasonality import WEEKDAY_ORDER, detect_seasonality
from src.trend_detection import detect_trend

ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = ROOT / "data" / "sample_sales.csv"

st.set_page_config(
    page_title="Intelligent Demand Forecasting Agent",
    page_icon="📦",
    layout="wide",
)


def build_insights(trend: dict, seasonality: dict, anomalies: dict, inventory: dict) -> list[str]:
    lines: list[str] = []
    if trend["label"] == "Increasing":
        lines.append(f"Demand is increasing by {abs(trend['change_pct']):.0f}%.")
    elif trend["label"] == "Decreasing":
        lines.append(f"Demand is decreasing by {abs(trend['change_pct']):.0f}%.")
    else:
        lines.append("Demand is relatively stable versus the previous period.")

    if seasonality.get("weekly_detected") and seasonality.get("peak_day"):
        peak_avg = seasonality["weekday_averages"].get(seasonality["peak_day"], 0)
        lines.append(
            f"{seasonality['peak_day']} has the highest average demand ({peak_avg:.0f} units)."
        )
    else:
        lines.append("No strong weekly seasonality was found.")

    count = anomalies["count"]
    if count == 0:
        lines.append("No sales anomalies were detected.")
    elif count == 1:
        lines.append("1 sales anomaly was detected.")
    else:
        lines.append(f"{count} sales anomalies were detected.")

    qty = inventory["order_quantity"]
    if qty > 0:
        lines.append(f"Approximately {qty:.0f} additional units are recommended.")
    else:
        lines.append("No additional order is recommended based on current stock.")
    return lines


def sales_forecast_chart(
    history: pd.Series,
    forecast: dict,
    anomalies: dict,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Historical sales",
            line=dict(color="#1f77b4", width=2),
        )
    )
    points = anomalies["points"]
    if not points.empty:
        fig.add_trace(
            go.Scatter(
                x=points["Date"],
                y=points["Sales"],
                mode="markers",
                name="Anomalies",
                marker=dict(color="#d62728", size=10, symbol="x"),
            )
        )
    frame = forecast["frame"]
    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Upper"],
            mode="lines",
            name="Upper interval",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Lower"],
            mode="lines",
            name="Confidence interval",
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.18)",
            line=dict(width=0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Predicted"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=420,
        xaxis_title="Date",
        yaxis_title="Units",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    return fig


def weekday_chart(seasonality: dict) -> go.Figure:
    avgs = seasonality.get("weekday_averages") or {}
    days = [d for d in WEEKDAY_ORDER if d in avgs]
    values = [avgs[d] for d in days]
    fig = go.Figure(go.Bar(x=days, y=values, marker_color="#2ca02c", name="Avg sales"))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        xaxis_title="Weekday",
        yaxis_title="Average units",
        showlegend=False,
    )
    return fig


def main() -> None:
    st.title("Intelligent Demand Forecasting Agent")
    st.caption(
        "Predict future product demand from historical sales and get inventory decision support."
    )

    st.sidebar.header("Controls")
    uploaded = st.sidebar.file_uploader("Upload sales CSV", type=["csv"])
    use_sample = st.sidebar.checkbox("Use sample dataset", value=uploaded is None)

    raw = None
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception:
            st.error("Could not read the CSV. Please upload a valid comma-separated file.")
            return
    elif use_sample and SAMPLE_CSV.exists():
        raw = pd.read_csv(SAMPLE_CSV)
        st.sidebar.caption(f"Loaded `{SAMPLE_CSV.name}`.")
    else:
        st.info("Upload a CSV with Date, Product, Sales, Inventory, and Price, or enable the sample dataset.")
        return

    clean, warnings, errors = validate_and_clean(raw)
    for msg in warnings:
        st.warning(msg)
    if errors:
        for msg in errors:
            st.error(msg)
        return

    product_list = products(clean)
    product = st.sidebar.selectbox("Product", product_list)
    horizon = st.sidebar.selectbox("Forecast horizon", [7, 14, 30], index=1, format_func=lambda d: f"{d}-day forecast")

    product_df = filter_product(clean, product)
    history = daily_series(product_df)
    metrics = historical_metrics(history)
    current_inv = latest_inventory(product_df)

    anomalies = detect_anomalies(history)
    train = winsorize_for_forecast(history, anomalies["lower"], anomalies["upper"])
    forecast = forecast_demand(train, horizon=horizon)
    trend = detect_trend(history)
    seasonality = detect_seasonality(history)
    inventory = recommend_inventory(forecast["total"], current_inv, history)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sales", f"{metrics['total_sales']:.0f}")
    m2.metric("Average Sales", f"{metrics['average_sales']:.1f}")
    m3.metric("Current Inventory", f"{current_inv:.0f}")
    m4.metric("Forecast Demand", f"{forecast['total']:.0f}")

    st.subheader("Historical Sales, Forecast, and Anomalies")
    st.plotly_chart(sales_forecast_chart(history, forecast, anomalies), use_container_width=True)
    st.caption(
        f"Forecast method: {forecast['method']}. "
        f"Total predicted demand: {forecast['total']:.0f} units. "
        f"Average predicted daily demand: {forecast['average']:.1f} units."
    )
    if forecast.get("fallback_note"):
        st.info(forecast["fallback_note"])

    left, right = st.columns(2)
    with left:
        st.subheader("Insights")
        st.markdown(f"**Trend:** {trend['label']}")
        st.write(trend["explanation"])
        if seasonality["weekly_detected"]:
            st.markdown("**Weekly Seasonality Detected**")
            st.markdown(f"**Peak Day: {seasonality['peak_day']}**")
        else:
            st.markdown("**Seasonality:** No strong weekly pattern")
        st.write(seasonality["explanation"])
        st.markdown(f"**Anomalies:** {anomalies['count']} unusual day(s)")
        st.write(
            f"Typical demand range: {anomalies['normal_low']:.0f}–{anomalies['normal_high']:.0f} units "
            f"(IQR bounds {anomalies['lower']:.0f}–{anomalies['upper']:.0f})."
        )
        if not anomalies["points"].empty:
            st.dataframe(anomalies["points"], use_container_width=True, hide_index=True)

    with right:
        st.subheader("Weekly Demand Pattern")
        st.plotly_chart(weekday_chart(seasonality), use_container_width=True)
        st.subheader("Forecast table")
        table = forecast["frame"].copy()
        table["Predicted"] = table["Predicted"].round(1)
        table["Lower"] = table["Lower"].round(1)
        table["Upper"] = table["Upper"].round(1)
        st.dataframe(table, use_container_width=True, hide_index=True, height=240)

    st.subheader("Inventory Recommendation")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Current stock", f"{inventory['current_inventory']:.0f}")
    i2.metric("Forecast requirement", f"{inventory['forecast_demand']:.0f}")
    i3.metric("Safety stock", f"{inventory['safety_stock']:.0f} ({inventory['safety_pct']*100:.0f}%)")
    i4.metric("Recommended order", f"{inventory['order_quantity']:.0f}")
    if inventory["status"] == "stockout":
        st.error(inventory["warning"])
    elif inventory["status"] == "overstock":
        st.warning(inventory["warning"])
    else:
        st.success(inventory["warning"])
    st.caption(inventory["disclaimer"])

    st.subheader("AI Insights")
    for line in build_insights(trend, seasonality, anomalies, inventory):
        st.write(f"• {line}")

    with st.expander("Cleaned data preview"):
        st.dataframe(product_df.tail(20), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
