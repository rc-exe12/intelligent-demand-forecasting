"""Streamlit dashboard for the Intelligent Demand Forecasting Agent.

This file is presentation only: every number on screen comes from the
calculation modules in ``src/`` (data cleaning, forecasting, trend,
seasonality, anomaly detection, inventory). Nothing is hardcoded.
"""

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

NAV_ITEMS = [
    ("Dashboard", "🏠"),
    ("Analytics", "📊"),
    ("Forecast", "📈"),
    ("Inventory", "📦"),
    ("Anomalies", "⚠️"),
    ("Data", "🗂️"),
    ("Settings", "⚙️"),
]

ACCENT = "#1f9d55"
ACCENT_DARK = "#0f7a3d"
FORECAST_COLOR = "#e8823c"
LINE_COLOR = "#1c3d2e"
DANGER = "#d64545"
WARNING = "#c98a1f"

st.set_page_config(
    page_title="DemandAI | Intelligent Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp { font-family: 'Inter', -apple-system, sans-serif; }

        .stApp { background: #f4f7f5; }
        .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1300px; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0e2a1e 0%, #0a2018 100%);
            min-width: 245px !important;
        }
        section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
        section[data-testid="stSidebar"] * { color: #d9e8df !important; }
        .sidebar-brand {
            display: flex; align-items: center; gap: 10px;
            padding: 0 0.6rem 1.1rem 0.6rem;
            margin-bottom: 0.6rem;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        .sidebar-brand .logo-badge {
            width: 34px; height: 34px; border-radius: 9px;
            background: linear-gradient(135deg, #22b25c, #0f7a3d);
            display: flex; align-items: center; justify-content: center;
            font-size: 17px;
        }
        .sidebar-brand .brand-text { font-weight: 800; font-size: 1.18rem; letter-spacing: 0.2px; color: #f3fbf6 !important; }
        .sidebar-caption { padding: 0 0.6rem 0.9rem 0.6rem; font-size: 0.76rem; color: #86a596 !important; }

        section[data-testid="stSidebar"] .stButton { margin-bottom: 3px; }
        section[data-testid="stSidebar"] .stButton>button {
            width: 100%; text-align: left; justify-content: flex-start;
            background: transparent; border: 1px solid transparent;
            color: #cfe3d8 !important; font-weight: 500; font-size: 0.92rem;
            padding: 0.5rem 0.75rem; border-radius: 8px;
            box-shadow: none;
        }
        section[data-testid="stSidebar"] .stButton>button:hover {
            background: rgba(255,255,255,0.07); border-color: rgba(255,255,255,0.08);
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stButton>button[kind="primary"] {
            background: #1f9d55 !important; color: #ffffff !important;
            font-weight: 600; border: none;
        }
        section[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover { background: #1c8c4c !important; }

        /* Header */
        .app-title { font-size: 1.65rem; font-weight: 800; color: #10231b; margin-bottom: 0.1rem; }
        .app-subtitle { color: #63796f; font-size: 0.98rem; margin-bottom: 1.1rem; }

        /* Generic card */
        .card {
            background: #ffffff; border: 1px solid #e6ebe8; border-radius: 14px;
            padding: 1.1rem 1.2rem; box-shadow: 0 1px 2px rgba(16,35,27,0.04);
        }
        .card h4 { margin-top: 0; }

        /* KPI cards */
        .kpi-card {
            background: #ffffff; border: 1px solid #e6ebe8; border-radius: 14px;
            padding: 1rem 1.15rem; box-shadow: 0 1px 2px rgba(16,35,27,0.04);
            height: 100%;
        }
        .kpi-label { font-size: 0.8rem; color: #6b7d74; font-weight: 500; margin-bottom: 0.35rem; }
        .kpi-value { font-size: 1.55rem; font-weight: 800; color: #10231b; line-height: 1.1; }
        .kpi-sub { font-size: 0.78rem; color: #8b9c93; margin-top: 0.3rem; }
        .kpi-icon {
            display: inline-flex; align-items: center; justify-content: center;
            width: 30px; height: 30px; border-radius: 8px; background: #e8f6ee;
            font-size: 15px; margin-bottom: 0.5rem;
        }

        /* Section title */
        .section-title { font-weight: 700; font-size: 1.05rem; color: #10231b; margin: 1.4rem 0 0.6rem 0; }

        /* Insight rows */
        .insight-row { display: flex; justify-content: space-between; padding: 0.55rem 0; border-bottom: 1px solid #eef2f0; }
        .insight-row:last-child { border-bottom: none; }
        .insight-label { color: #6b7d74; font-size: 0.85rem; }
        .insight-value { font-weight: 700; font-size: 0.88rem; color: #10231b; }
        .badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.76rem; font-weight: 700; }
        .badge-up { background: #e8f6ee; color: #0f7a3d; }
        .badge-down { background: #fdecec; color: #b83232; }
        .badge-flat { background: #f1f3f2; color: #62766c; }

        /* AI summary */
        .ai-summary {
            background: linear-gradient(135deg, #103a27 0%, #0d2a1c 100%);
            border-radius: 14px; padding: 1.2rem 1.35rem; color: #eafbf1;
        }
        .ai-summary .ai-title { font-weight: 700; font-size: 0.95rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px; }
        .ai-summary p { color: #d9f2e4; font-size: 0.93rem; line-height: 1.55; margin: 0; }

        div[data-testid="stMetricValue"] { color: #10231b; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }

        [data-testid="stForm"] { border: none; padding: 0; background: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(icon: str, label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card"><div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{sub_html}</div>'
    )


def trend_badge(label: str) -> str:
    cls = {"Increasing": "badge-up", "Decreasing": "badge-down"}.get(label, "badge-flat")
    return f'<span class="badge {cls}">{label}</span>'


# --------------------------------------------------------------------------
# Charts (built only from calculated data passed in)
# --------------------------------------------------------------------------
def sales_forecast_chart(history: pd.Series, forecast: dict, anomalies: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index, y=history.values, mode="lines", name="Historical sales",
            line=dict(color=LINE_COLOR, width=2),
        )
    )
    points = anomalies["points"]
    if not points.empty:
        fig.add_trace(
            go.Scatter(
                x=points["Date"], y=points["Sales"], mode="markers", name="Anomalies",
                marker=dict(color=DANGER, size=9, symbol="x"),
            )
        )
    frame = forecast["frame"]
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["Upper"], mode="lines", name="Upper interval",
                              line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["Lower"], mode="lines", name="Confidence interval",
                              fill="tonexty", fillcolor="rgba(232,130,60,0.16)", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["Predicted"], mode="lines+markers", name="Forecast",
                              line=dict(color=FORECAST_COLOR, width=2, dash="dash")))
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10), height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Date", yaxis_title="Units",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#eef2f0")
    fig.update_yaxes(gridcolor="#eef2f0")
    return fig


def anomaly_chart(history: pd.Series, anomalies: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="Daily sales",
                              line=dict(color=LINE_COLOR, width=2)))
    fig.add_hrect(y0=max(anomalies["lower"], 0), y1=anomalies["upper"],
                  fillcolor="rgba(31,157,85,0.08)", line_width=0)
    points = anomalies["points"]
    if not points.empty:
        colors = ["#d64545" if t == "High" else "#c98a1f" for t in points["Type"]]
        fig.add_trace(go.Scatter(x=points["Date"], y=points["Sales"], mode="markers", name="Anomaly",
                                  marker=dict(color=colors, size=11, symbol="x", line=dict(width=2))))
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=380, plot_bgcolor="white",
                       paper_bgcolor="white", xaxis_title="Date", yaxis_title="Units",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_xaxes(gridcolor="#eef2f0")
    fig.update_yaxes(gridcolor="#eef2f0")
    return fig


def weekday_chart(seasonality: dict) -> go.Figure:
    avgs = seasonality.get("weekday_averages") or {}
    days = [d for d in WEEKDAY_ORDER if d in avgs]
    values = [avgs[d] for d in days]
    fig = go.Figure(go.Bar(x=days, y=values, marker_color=ACCENT, name="Avg sales"))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300, plot_bgcolor="white",
                       paper_bgcolor="white", xaxis_title="Weekday", yaxis_title="Average units",
                       showlegend=False)
    fig.update_xaxes(gridcolor="#eef2f0")
    fig.update_yaxes(gridcolor="#eef2f0")
    return fig


def trend_seasonality_chart(history: pd.Series) -> go.Figure:
    rolling = history.rolling(window=min(7, max(1, len(history))), min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="Daily sales",
                              line=dict(color="#c7d6cd", width=1.4)))
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values, mode="lines", name="7-day trend",
                              line=dict(color=ACCENT_DARK, width=2.6)))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300, plot_bgcolor="white",
                       paper_bgcolor="white", xaxis_title="Date", yaxis_title="Units",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_xaxes(gridcolor="#eef2f0")
    fig.update_yaxes(gridcolor="#eef2f0")
    return fig


# --------------------------------------------------------------------------
# AI summary (template-based, no external LLM)
# --------------------------------------------------------------------------
def generate_ai_summary(trend: dict, seasonality: dict, anomalies: dict, forecast: dict, inventory: dict) -> str:
    if trend["label"] == "Increasing":
        trend_sentence = f"Demand is currently increasing by approximately {abs(trend['change_pct']):.0f}%."
    elif trend["label"] == "Decreasing":
        trend_sentence = f"Demand is currently decreasing by approximately {abs(trend['change_pct']):.0f}%."
    else:
        trend_sentence = "Demand is currently stable versus the previous period."

    if seasonality.get("weekly_detected") and seasonality.get("peak_day"):
        season_sentence = f"Weekly seasonality is present, with peak demand on {seasonality['peak_day']}."
    else:
        season_sentence = "No strong weekly seasonality was detected."

    count = anomalies["count"]
    anomaly_sentence = (
        "No sales anomalies were detected in the historical data."
        if count == 0
        else f"{count} sales anomal{'y' if count == 1 else 'ies'} {'was' if count == 1 else 'were'} detected."
    )

    forecast_sentence = (
        f"Forecast demand for the next {forecast['horizon']} days is {forecast['total']:.0f} units "
        f"(method: {forecast['method']})."
    )

    if inventory["status"] == "stockout":
        inventory_sentence = (
            f"Current inventory is below projected demand, so approximately "
            f"{inventory['order_quantity']:.0f} additional units are recommended."
        )
    elif inventory["status"] == "overstock":
        inventory_sentence = (
            f"Current inventory is well above the recommended stock level of "
            f"{inventory['recommended_stock']:.0f} units, so no additional order is needed right now."
        )
    else:
        inventory_sentence = (
            f"Current inventory is in line with projected demand; the recommended order is "
            f"{inventory['order_quantity']:.0f} units."
        )

    return " ".join([trend_sentence, season_sentence, anomaly_sentence, forecast_sentence, inventory_sentence])


# --------------------------------------------------------------------------
# App state & data loading
# --------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "page": "Dashboard",
        "anomaly_method": "iqr",
        "safety_mode": "auto",
        "safety_override_pct": 15,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="logo-badge">📦</div>
            <div class="brand-text">DemandAI</div>
        </div>
        <div class="sidebar-caption">Demand forecasting &amp; inventory</div>
        """,
        unsafe_allow_html=True,
    )
    for name, icon in NAV_ITEMS:
        is_active = st.session_state["page"] == name
        if st.sidebar.button(
            f"{icon}   {name}",
            key=f"nav_{name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state["page"] = name
            st.rerun()


def main() -> None:
    inject_css()
    init_state()
    render_sidebar()

    st.markdown('<div class="app-title">Intelligent Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Turn historical sales into smarter inventory decisions.</div>',
        unsafe_allow_html=True,
    )

    # --- Data source controls (apply immediately) ---
    top_l, top_r = st.columns([3, 1])
    with top_l:
        uploaded = st.file_uploader("Upload sales CSV", type=["csv"], label_visibility="collapsed")
    with top_r:
        use_sample = st.checkbox("Use sample dataset", value=uploaded is None)

    raw = None
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception:
            st.error("Could not read the CSV. Please upload a valid comma-separated file.")
            return
    elif use_sample and SAMPLE_CSV.exists():
        raw = pd.read_csv(SAMPLE_CSV)
    else:
        st.info("Upload a CSV with Date, Product, Sales, Inventory, and Price, or enable the sample dataset.")
        return

    clean, warnings, errors = validate_and_clean(raw)
    if errors:
        for msg in errors:
            st.error(msg)
        return

    product_list = products(clean)
    if not product_list:
        st.warning("No valid products found in the data.")
        return

    # --- Forecast controls (gated behind Generate Forecast, via st.form) ---
    with st.form("controls_form"):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            product = st.selectbox("Product", product_list)
        with c2:
            horizon = st.selectbox("Forecast horizon", [7, 14, 30], index=1,
                                    format_func=lambda d: f"{d}-day forecast")
        with c3:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            st.form_submit_button("Generate Forecast", use_container_width=True, type="primary")

    # --- Shared calculations (single source of truth for every page) ---
    product_df = filter_product(clean, product)
    history = daily_series(product_df)
    metrics = historical_metrics(history)
    current_inv = latest_inventory(product_df)

    anomalies = detect_anomalies(history, method=st.session_state["anomaly_method"])
    train = winsorize_for_forecast(history, anomalies["lower"], anomalies["upper"])
    forecast = forecast_demand(train, horizon=horizon)
    trend = detect_trend(history)
    seasonality = detect_seasonality(history)
    override = (
        st.session_state["safety_override_pct"] / 100.0
        if st.session_state["safety_mode"] == "manual"
        else None
    )
    inventory = recommend_inventory(forecast["total"], current_inv, history, override_safety_pct=override)

    for msg in warnings:
        st.warning(msg)
    if forecast.get("fallback_note"):
        st.info(forecast["fallback_note"])

    page = st.session_state["page"]
    if page == "Dashboard":
        render_dashboard(metrics, current_inv, forecast, history, anomalies, trend, seasonality, inventory)
    elif page == "Analytics":
        render_analytics(trend, seasonality, history)
    elif page == "Forecast":
        render_forecast(history, forecast, anomalies)
    elif page == "Inventory":
        render_inventory(inventory)
    elif page == "Anomalies":
        render_anomalies(history, anomalies)
    elif page == "Data":
        render_data(raw, clean, product_df, warnings)
    elif page == "Settings":
        render_settings()


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def render_kpis(metrics: dict, current_inv: float, forecast: dict) -> None:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card("💰", "Total Sales", f"{metrics['total_sales']:.0f}", "units, full history"),
                     unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("📅", "Average Daily Demand", f"{metrics['average_sales']:.1f}", "units/day"),
                     unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("🏬", "Current Inventory", f"{current_inv:.0f}", "units on hand"),
                     unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card("📈", "Forecast Demand", f"{forecast['total']:.0f}",
                              f"next {forecast['horizon']} days"), unsafe_allow_html=True)


def render_dashboard(metrics, current_inv, forecast, history, anomalies, trend, seasonality, inventory) -> None:
    render_kpis(metrics, current_inv, forecast)

    st.markdown('<div class="section-title">Historical Sales + Forecast</div>', unsafe_allow_html=True)
    left, right = st.columns([2.2, 1])
    with left:
        with st.container(border=True):
            st.plotly_chart(sales_forecast_chart(history, forecast, anomalies), use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown("#### Demand Insights")
            st.markdown(
                f'<div class="insight-row"><span class="insight-label">Trend</span>'
                f'<span class="insight-value">{trend_badge(trend["label"])}</span></div>',
                unsafe_allow_html=True,
            )
            peak = seasonality.get("peak_day") or "—"
            season_val = f"Peak: {peak}" if seasonality.get("weekly_detected") else "No strong pattern"
            st.markdown(
                f'<div class="insight-row"><span class="insight-label">Seasonality</span>'
                f'<span class="insight-value">{season_val}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="insight-row"><span class="insight-label">Anomalies</span>'
                f'<span class="insight-value">{anomalies["count"]} detected</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(trend["explanation"])

    st.markdown('<div class="section-title">Inventory Recommendation</div>', unsafe_allow_html=True)
    render_inventory_cards(inventory)

    lc, rc = st.columns(2)
    with lc:
        st.markdown('<div class="section-title">Demand by Day of Week</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.plotly_chart(weekday_chart(seasonality), use_container_width=True)
    with rc:
        st.markdown('<div class="section-title">Trend &amp; Seasonality</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.plotly_chart(trend_seasonality_chart(history), use_container_width=True)

    st.markdown('<div class="section-title">AI Demand Summary</div>', unsafe_allow_html=True)
    summary = generate_ai_summary(trend, seasonality, anomalies, forecast, inventory)
    st.markdown(
        f'<div class="ai-summary"><div class="ai-title">🤖 AI Demand Summary</div><p>{summary}</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Forecast Details</div>', unsafe_allow_html=True)
    render_forecast_table(forecast)


def render_inventory_cards(inventory: dict) -> None:
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.markdown(kpi_card("🏬", "Current Inventory", f"{inventory['current_inventory']:.0f}"),
                     unsafe_allow_html=True)
    with i2:
        st.markdown(kpi_card("📈", "Forecast Demand", f"{inventory['forecast_demand']:.0f}"),
                     unsafe_allow_html=True)
    with i3:
        st.markdown(kpi_card("🛡️", "Safety Stock",
                              f"{inventory['safety_stock']:.0f}", f"{inventory['safety_pct']*100:.0f}% buffer"),
                     unsafe_allow_html=True)
    with i4:
        st.markdown(kpi_card("🛒", "Recommended Order", f"{inventory['order_quantity']:.0f}"),
                     unsafe_allow_html=True)
    status_fn = {"stockout": st.error, "overstock": st.warning, "ok": st.success}[inventory["status"]]
    status_fn(inventory["warning"])
    st.caption(inventory["disclaimer"])


def render_forecast_table(forecast: dict) -> None:
    table = forecast["frame"].copy()
    table["Predicted"] = table["Predicted"].round(1)
    table["Lower"] = table["Lower"].round(1)
    table["Upper"] = table["Upper"].round(1)
    with st.container(border=True):
        st.dataframe(table, use_container_width=True, hide_index=True, height=260)
        st.download_button(
            "Download forecast CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="forecast.csv",
            mime="text/csv",
        )


def render_analytics(trend: dict, seasonality: dict, history: pd.Series) -> None:
    st.markdown('<div class="section-title">Trend Detail</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Trend", trend["label"])
        c2.metric("Recent avg / day", f"{trend['recent_avg']:.1f}")
        c3.metric("Previous avg / day", f"{trend['previous_avg']:.1f}")
        st.write(trend["explanation"])

    st.markdown('<div class="section-title">Seasonality Detail</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.write(seasonality["explanation"])
        if seasonality.get("monthly_detected"):
            st.write(f"Monthly peak month: **{seasonality['peak_month']}**")

    lc, rc = st.columns(2)
    with lc:
        st.markdown('<div class="section-title">Demand by Day of Week</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.plotly_chart(weekday_chart(seasonality), use_container_width=True)
    with rc:
        st.markdown('<div class="section-title">Trend &amp; Seasonality</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.plotly_chart(trend_seasonality_chart(history), use_container_width=True)


def render_forecast(history: pd.Series, forecast: dict, anomalies: dict) -> None:
    st.markdown('<div class="section-title">Historical Sales + Forecast</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.plotly_chart(sales_forecast_chart(history, forecast, anomalies), use_container_width=True)
        st.caption(
            f"Forecast method: {forecast['method']}. Total predicted demand: {forecast['total']:.0f} units. "
            f"Average predicted daily demand: {forecast['average']:.1f} units."
        )
    st.markdown('<div class="section-title">Forecast Details</div>', unsafe_allow_html=True)
    render_forecast_table(forecast)


def render_inventory(inventory: dict) -> None:
    st.markdown('<div class="section-title">Inventory Recommendation</div>', unsafe_allow_html=True)
    render_inventory_cards(inventory)
    with st.container(border=True):
        st.markdown("#### How this is calculated")
        st.write(
            f"Recommended stock = forecast demand ({inventory['forecast_demand']:.0f}) "
            f"+ safety stock ({inventory['safety_stock']:.0f}) = **{inventory['recommended_stock']:.0f} units**."
        )
        st.write(
            f"Recommended order = max(0, recommended stock − current inventory) = "
            f"max(0, {inventory['recommended_stock']:.0f} − {inventory['current_inventory']:.0f}) = "
            f"**{inventory['order_quantity']:.0f} units**."
        )


def render_anomalies(history: pd.Series, anomalies: dict) -> None:
    st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.plotly_chart(anomaly_chart(history, anomalies), use_container_width=True)
        st.caption(
            f"Method: {anomalies['method'].upper()}. Typical demand range: "
            f"{anomalies['normal_low']:.0f}–{anomalies['normal_high']:.0f} units "
            f"(bounds {anomalies['lower']:.0f}–{anomalies['upper']:.0f})."
        )
    st.markdown(f'<div class="section-title">Flagged Days ({anomalies["count"]})</div>', unsafe_allow_html=True)
    with st.container(border=True):
        if anomalies["points"].empty:
            st.write("No anomalies detected in the current history.")
        else:
            st.dataframe(anomalies["points"], use_container_width=True, hide_index=True)


def render_data(raw: pd.DataFrame, clean: pd.DataFrame, product_df: pd.DataFrame, warnings: list[str]) -> None:
    st.markdown('<div class="section-title">Data Quality</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.metric("Raw rows", len(raw) if raw is not None else 0)
        c2.metric("Clean rows used", len(clean))
        if warnings:
            for msg in warnings:
                st.write(f"• {msg}")
        else:
            st.write("No data quality issues found.")

    st.markdown('<div class="section-title">Cleaned Data (selected product)</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.dataframe(product_df, use_container_width=True, hide_index=True, height=380)
        st.download_button(
            "Download cleaned data CSV",
            data=clean.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_sales_data.csv",
            mime="text/csv",
        )


def render_settings() -> None:
    st.markdown('<div class="section-title">Anomaly Detection Method</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.radio(
            "Method",
            options=["iqr", "zscore"],
            format_func=lambda v: "IQR (interquartile range)" if v == "iqr" else "Z-score",
            key="anomaly_method",
            horizontal=True,
        )
        st.caption("Controls how unusual sales days are flagged across every page.")

    st.markdown('<div class="section-title">Safety Stock</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.radio(
            "Safety stock mode",
            options=["auto", "manual"],
            format_func=lambda v: "Automatic (based on demand variability)" if v == "auto" else "Manual override",
            key="safety_mode",
            horizontal=True,
        )
        st.slider(
            "Manual safety stock percentage",
            min_value=5, max_value=50, key="safety_override_pct",
            disabled=st.session_state["safety_mode"] != "manual",
        )
        st.caption("Automatic mode derives the safety-stock percentage from the coefficient of variation of demand.")


if __name__ == "__main__":
    main()
