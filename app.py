"""KARIGAR X - Intelligent Demand Forecasting Agent.

Presentation layer for the existing forecasting backend.
Backend modules under src/ remain the single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from html import escape

import pandas as pd
import textwrap
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SAMPLE_CSV_CANDIDATES = [
    ROOT / "demandai_sample_sales.csv",
    ROOT / "data" / "demandai_sample_sales.csv",
    ROOT / "data" / "sample_sales.csv",
]

NAV_ITEMS = [
    ("Dashboard", "⌂"),
    ("Analytics", "◒"),
    ("Forecast", "↗"),
    ("Inventory", "▣"),
    ("Anomalies", "!"),
    ("Data", "▤"),
    ("Settings", "⚙"),
]

THEMES = {
    "Dark": {
        "bg": "#07110D",
        "surface": "#0D1A15",
        "surface2": "#12231B",
        "text": "#F3F7F4",
        "muted": "#9AAEA4",
        "border": "#20362B",
        "accent": "#29C477",
        "accent_dark": "#159A59",
        "accent_soft": "#143C29",
        "grid": "#1D3127",
        "chart_history": "#E8F2EC",
        "chart_forecast": "#FFB45B",
        "chart_anomaly": "#FF5E67",
        "plot_bg": "#0D1A15",
    },
    "Light": {
        "bg": "#F4F7F5",
        "surface": "#FFFFFF",
        "surface2": "#EDF4F0",
        "text": "#10231B",
        "muted": "#63766D",
        "border": "#DCE7E1",
        "accent": "#159957",
        "accent_dark": "#0E713F",
        "accent_soft": "#E5F5EC",
        "grid": "#E7EEE9",
        "chart_history": "#163F2D",
        "chart_forecast": "#D9782F",
        "chart_anomaly": "#D9434D",
        "plot_bg": "#FFFFFF",
    },
    "Glass": {
        "bg": "#06110D",
        "surface": "rgba(255,255,255,0.075)",
        "surface2": "rgba(255,255,255,0.105)",
        "text": "#F4FFFA",
        "muted": "#A8BCB2",
        "border": "rgba(255,255,255,0.14)",
        "accent": "#52E6A0",
        "accent_dark": "#28BE78",
        "accent_soft": "rgba(82,230,160,0.12)",
        "grid": "rgba(255,255,255,0.10)",
        "chart_history": "#E9FFF3",
        "chart_forecast": "#FFC26B",
        "chart_anomaly": "#FF6670",
        "plot_bg": "rgba(255,255,255,0.025)",
    },
}


st.set_page_config(
    page_title="KARIGAR X | Intelligent Demand Forecasting",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "page": "Dashboard",
        "theme": "Dark",
        "anomaly_method": "iqr",
        "safety_mode": "auto",
        "safety_override_pct": 15,
        "selected_product": None,
        "forecast_horizon": 14,
        "forecast_ready": False,
        "use_sample": False,
        "uploaded_file": None,
        "data_source": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def inject_css(theme: dict) -> None:
    st.markdown(
        textwrap.dedent(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --bg: {theme["bg"]};
            --surface: {theme["surface"]};
            --surface2: {theme["surface2"]};
            --text: {theme["text"]};
            --muted: {theme["muted"]};
            --border: {theme["border"]};
            --accent: {theme["accent"]};
            --accent-dark: {theme["accent_dark"]};
            --accent-soft: {theme["accent_soft"]};
        }}

        html, body, [class*="css"], .stApp {{
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .stApp {{
            background: var(--bg);
            color: var(--text);
        }}

        .block-container {{
            max-width: 1450px;
            padding-top: 1.15rem;
            padding-bottom: 3rem;
        }}

        /* Remove the old sidebar completely. */
        section[data-testid="stSidebar"] {{
            display: none !important;
        }}

        #MainMenu, footer {{
            visibility: hidden;
        }}

        header[data-testid="stHeader"] {{
            background: transparent !important;
        }}

        /* Make native text/widgets follow the selected theme. */
        .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
        .stApp h4, .stApp h5, .stApp h6, .stApp span,
        .stApp [data-testid="stCaptionContainer"] {{
            color: var(--text);
        }}

        .stApp [data-testid="stCaptionContainer"] {{
            color: var(--muted) !important;
        }}

        .stTextInput input, .stNumberInput input,
        .stDateInput input, .stTextArea textarea {{
            background: var(--surface) !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
        }}

        div[data-baseweb="select"] > div {{
            background: var(--surface) !important;
            border-color: var(--border) !important;
            color: var(--text) !important;
        }}

        div[data-baseweb="select"] span {{
            color: var(--text) !important;
        }}

        [data-testid="stFileUploader"] {{
            background: var(--surface) !important;
            border: 1px dashed var(--accent) !important;
            border-radius: 16px !important;
        }}

        [data-testid="stFileUploader"] section {{
            background: transparent !important;
        }}

        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span {{
            color: var(--muted) !important;
        }}

        .stButton > button, .stDownloadButton > button {{
            background: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 11px !important;
            font-weight: 650 !important;
            min-height: 42px;
            transition: 0.18s ease;
        }}

        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: var(--accent) !important;
            color: var(--accent) !important;
            transform: translateY(-1px);
        }}

        .stButton > button[kind="primary"] {{
            background: var(--accent) !important;
            color: #06120C !important;
            border-color: var(--accent) !important;
        }}

        .stButton > button[kind="primary"]:hover {{
            background: var(--accent-dark) !important;
            color: #ffffff !important;
        }}

        .topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 12px 16px;
            margin-bottom: 10px;
            box-shadow: 0 12px 35px rgba(0,0,0,0.08);
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 11px;
        }}

        .logo {{
            width: 42px;
            height: 42px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 13px;
            background: linear-gradient(135deg, var(--accent), var(--accent-dark));
            color: #06120C !important;
            font-size: 21px;
            font-weight: 900;
        }}

        .brand-name {{
            color: var(--text) !important;
            font-size: 20px;
            font-weight: 800;
            letter-spacing: -0.4px;
        }}

        .brand-tag {{
            color: var(--muted) !important;
            font-size: 10px;
            margin-top: 1px;
        }}

        .nav-wrap {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 6px;
            margin-bottom: 28px;
        }}

        .hero {{
            padding: 4px 4px 22px 4px;
        }}

        .eyebrow {{
            color: var(--accent) !important;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1.4px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}

        .hero-title {{
            color: var(--text) !important;
            font-size: clamp(30px, 4vw, 48px);
            line-height: 1.03;
            font-weight: 800;
            letter-spacing: -1.8px;
            margin: 0;
        }}

        .hero-copy {{
            color: var(--muted) !important;
            font-size: 15px;
            line-height: 1.65;
            max-width: 720px;
            margin-top: 11px;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            margin-top: 15px;
            padding: 7px 11px;
            border-radius: 999px;
            background: var(--accent-soft);
            border: 1px solid var(--border);
            color: var(--text) !important;
            font-size: 11px;
            font-weight: 650;
        }}

        .status-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 0 4px rgba(41,196,119,0.10);
        }}

        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.05);
        }}

        .metric-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 19px;
            min-height: 145px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.04);
        }}

        .metric-icon {{
            width: 34px;
            height: 34px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            background: var(--accent-soft);
            color: var(--accent) !important;
            font-size: 15px;
            margin-bottom: 13px;
        }}

        .metric-label {{
            color: var(--muted) !important;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .55px;
            text-transform: uppercase;
        }}

        .metric-value {{
            color: var(--text) !important;
            font-size: 29px;
            font-weight: 800;
            letter-spacing: -0.8px;
            margin-top: 4px;
        }}

        .metric-sub {{
            color: var(--muted) !important;
            font-size: 11px;
            margin-top: 5px;
        }}

        .section-title {{
            color: var(--text) !important;
            font-size: 18px;
            font-weight: 750;
            margin: 29px 0 11px 2px;
        }}

        .section-copy {{
            color: var(--muted) !important;
            font-size: 13px;
            line-height: 1.6;
            margin: -4px 0 12px 2px;
        }}

        .insight-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 21px;
            min-height: 245px;
        }}

        .insight-title {{
            color: var(--text) !important;
            font-size: 16px;
            font-weight: 750;
            margin-bottom: 12px;
        }}

        .insight-row {{
            display: flex;
            justify-content: space-between;
            gap: 20px;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }}

        .insight-row:last-child {{
            border-bottom: 0;
        }}

        .insight-label {{
            color: var(--muted) !important;
            font-size: 12px;
        }}

        .insight-value {{
            color: var(--text) !important;
            font-size: 12px;
            font-weight: 750;
            text-align: right;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 9px;
            border-radius: 999px;
            font-size: 10px;
            font-weight: 800;
        }}

        .badge-up {{
            background: rgba(41,196,119,0.14);
            color: var(--accent) !important;
        }}

        .badge-down {{
            background: rgba(217,67,77,0.13);
            color: #E75C65 !important;
        }}

        .badge-flat {{
            background: rgba(127,145,136,0.14);
            color: var(--muted) !important;
        }}

        .ai-card {{
            background: linear-gradient(135deg, #123A28 0%, #0A2519 100%);
            border: 1px solid rgba(82,230,160,0.22);
            border-radius: 20px;
            padding: 23px;
        }}

        .ai-title {{
            color: #F2FFF8 !important;
            font-size: 16px;
            font-weight: 750;
            margin-bottom: 9px;
        }}

        .ai-copy {{
            color: #CFE8DB !important;
            font-size: 13px;
            line-height: 1.7;
            margin: 0;
        }}

        .recommend-card {{
            background: linear-gradient(135deg, var(--surface), var(--surface2));
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 23px;
        }}

        .recommend-label {{
            color: var(--muted) !important;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: .8px;
            font-weight: 750;
        }}

        .recommend-number {{
            color: var(--accent) !important;
            font-size: 42px;
            line-height: 1;
            font-weight: 850;
            letter-spacing: -1.4px;
            margin: 9px 0;
        }}

        .recommend-sub {{
            color: var(--muted) !important;
            font-size: 12px;
        }}

        .upload-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 15px 18px 5px 18px;
            margin-bottom: 10px;
        }}

        .data-chip {{
            display: inline-flex;
            padding: 5px 9px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent) !important;
            font-size: 10px;
            font-weight: 750;
        }}

        .empty-state {{
            background: var(--surface);
            border: 1px dashed var(--border);
            border-radius: 18px;
            padding: 55px 25px;
            text-align: center;
        }}

        .empty-title {{
            color: var(--text) !important;
            font-size: 18px;
            font-weight: 750;
        }}

        .empty-copy {{
            color: var(--muted) !important;
            font-size: 13px;
            max-width: 560px;
            margin: 8px auto 0 auto;
            line-height: 1.6;
        }}

        /* Streamlit metric / dataframe surfaces */
        div[data-testid="stMetric"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 15px;
            padding: 14px;
        }}

        div[data-testid="stMetricLabel"] {{
            color: var(--muted) !important;
        }}

        div[data-testid="stMetricValue"] {{
            color: var(--text) !important;
        }}

        .stDataFrame, [data-testid="stDataFrame"] {{
            border-radius: 14px;
            overflow: hidden;
        }}

        /* Alerts */
        .stAlert {{
            border-radius: 13px !important;
        }}
        </style>
        """),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def metric_card(icon: str, label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{escape(icon)}</div>
        <div class="metric-label">{escape(label)}</div>
        <div class="metric-value">{escape(value)}</div>
        <div class="metric-sub">{escape(sub)}</div>
    </div>
    """


def trend_badge(label: str) -> str:
    cls = {
        "Increasing": "badge-up",
        "Decreasing": "badge-down",
    }.get(label, "badge-flat")
    return f'<span class="badge {cls}">{escape(label)}</span>'


def page_title(title: str, subtitle: str) -> None:
    st.markdown(
        textwrap.dedent(f"""
        <div class="hero">
            <div class="eyebrow">KARIGAR X / {escape(title)}</div>
            <div class="hero-title">{escape(title)}</div>
            <div class="hero-copy">{escape(subtitle)}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def safe_number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def configure_plot(fig: go.Figure, theme: dict, height: int = 400) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=20, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=theme["plot_bg"],
        font=dict(color=theme["muted"], family="Inter, sans-serif"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        gridcolor=theme["grid"],
        zeroline=False,
        linecolor=theme["border"],
    )
    fig.update_yaxes(
        gridcolor=theme["grid"],
        zeroline=False,
        linecolor=theme["border"],
    )
    return fig


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def sales_forecast_chart(
    history: pd.Series,
    forecast: dict,
    anomalies: dict,
    theme: dict,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Historical",
            line=dict(color=theme["chart_history"], width=2.3),
        )
    )

    points = anomalies.get("points", pd.DataFrame())
    if not points.empty:
        fig.add_trace(
            go.Scatter(
                x=points["Date"],
                y=points["Sales"],
                mode="markers",
                name="Anomalies",
                marker=dict(
                    color=theme["chart_anomaly"],
                    size=9,
                    symbol="x",
                    line=dict(width=1.5),
                ),
            )
        )

    frame = forecast["frame"]

    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Lower"],
            mode="lines",
            name="Forecast range",
            fill="tonexty",
            fillcolor="rgba(255,180,91,0.12)",
            line=dict(width=0),
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Predicted"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=theme["chart_forecast"], width=2.5, dash="dash"),
            marker=dict(size=4),
        )
    )

    fig.add_vline(
        x=history.index[-1],
        line_width=1,
        line_dash="dot",
        line_color=theme["accent"],
    )

    fig.update_layout(xaxis_title="Date", yaxis_title="Units")
    return configure_plot(fig, theme, 430)


def anomaly_chart(history: pd.Series, anomalies: dict, theme: dict) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Daily demand",
            line=dict(color=theme["chart_history"], width=2.2),
        )
    )

    fig.add_hrect(
        y0=max(safe_number(anomalies.get("lower")), 0),
        y1=safe_number(anomalies.get("upper")),
        fillcolor="rgba(41,196,119,0.08)",
        line_width=0,
        annotation_text="Typical range",
        annotation_position="top left",
        annotation_font=dict(color=theme["muted"], size=10),
    )

    points = anomalies.get("points", pd.DataFrame())
    if not points.empty:
        colors = [
            theme["chart_anomaly"] if str(t).lower() == "high"
            else theme["chart_forecast"]
            for t in points["Type"]
        ]
        fig.add_trace(
            go.Scatter(
                x=points["Date"],
                y=points["Sales"],
                mode="markers",
                name="Flagged days",
                marker=dict(
                    color=colors,
                    size=11,
                    symbol="x",
                    line=dict(width=1.5),
                ),
            )
        )

    fig.update_layout(xaxis_title="Date", yaxis_title="Units")
    return configure_plot(fig, theme, 390)


def weekday_chart(seasonality: dict, theme: dict) -> go.Figure:
    avgs = seasonality.get("weekday_averages") or {}
    days = [d for d in WEEKDAY_ORDER if d in avgs]
    values = [avgs[d] for d in days]

    fig = go.Figure(
        go.Bar(
            x=days,
            y=values,
            marker_color=theme["accent"],
            name="Average demand",
            hovertemplate="%{x}: %{y:.1f} units<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Weekday", yaxis_title="Average units", showlegend=False)
    return configure_plot(fig, theme, 310)


def trend_seasonality_chart(history: pd.Series, theme: dict) -> go.Figure:
    window = min(7, max(1, len(history)))
    rolling = history.rolling(window=window, min_periods=1).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Daily demand",
            line=dict(color=theme["muted"], width=1.4),
            opacity=0.55,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rolling.index,
            y=rolling.values,
            mode="lines",
            name="7-day trend",
            line=dict(color=theme["accent"], width=2.7),
        )
    )

    fig.update_layout(xaxis_title="Date", yaxis_title="Units")
    return configure_plot(fig, theme, 310)


# ---------------------------------------------------------------------------
# Analytical summary
# ---------------------------------------------------------------------------

def generate_ai_summary(
    trend: dict,
    seasonality: dict,
    anomalies: dict,
    forecast: dict,
    inventory: dict,
) -> str:
    if trend["label"] == "Increasing":
        trend_sentence = (
            f"Demand is increasing by approximately "
            f"{abs(safe_number(trend.get('change_pct'))):.0f}%."
        )
    elif trend["label"] == "Decreasing":
        trend_sentence = (
            f"Demand is decreasing by approximately "
            f"{abs(safe_number(trend.get('change_pct'))):.0f}%."
        )
    else:
        trend_sentence = "Demand is stable versus the previous period."

    if seasonality.get("weekly_detected") and seasonality.get("peak_day"):
        season_sentence = (
            f"Weekly seasonality is present, with peak demand on "
            f"{seasonality['peak_day']}."
        )
    else:
        season_sentence = "No strong weekly seasonality was detected."

    count = int(anomalies.get("count", 0))
    if count == 0:
        anomaly_sentence = "No sales anomalies were detected."
    elif count == 1:
        anomaly_sentence = "One unusual sales day was detected."
    else:
        anomaly_sentence = f"{count} unusual sales days were detected."

    forecast_sentence = (
        f"The {forecast['horizon']}-day forecast is "
        f"{safe_number(forecast['total']):.0f} units using {forecast['method']}."
    )

    status = inventory.get("status")
    if status == "stockout":
        inventory_sentence = (
            f"Current inventory is below projected demand; "
            f"approximately {safe_number(inventory['order_quantity']):.0f} "
            f"additional units are recommended."
        )
    elif status == "overstock":
        inventory_sentence = (
            f"Current inventory is above the recommended stock level of "
            f"{safe_number(inventory['recommended_stock']):.0f} units; "
            f"no additional order is needed right now."
        )
    else:
        inventory_sentence = (
            f"Inventory is aligned with projected demand; the recommended "
            f"order is {safe_number(inventory['order_quantity']):.0f} units."
        )

    return " ".join(
        [
            trend_sentence,
            season_sentence,
            anomaly_sentence,
            forecast_sentence,
            inventory_sentence,
        ]
    )


# ---------------------------------------------------------------------------
# Top navigation
# ---------------------------------------------------------------------------

def render_topbar() -> None:
    c1, c2 = st.columns([5.5, 1.2])

    with c1:
        st.markdown(
            textwrap.dedent("""
            <div class="topbar">
                <div class="brand">
                    <div class="logo">◈</div>
                    <div>
                        <div class="brand-name">KARIGAR X</div>
                        <div class="brand-tag">INTELLIGENT DEMAND FORECASTING</div>
                    </div>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    with c2:
        theme = st.selectbox(
            "Theme",
            list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state["theme"]),
            key="theme_selector",
            label_visibility="collapsed",
        )
        if theme != st.session_state["theme"]:
            st.session_state["theme"] = theme
            st.rerun()


def render_navigation() -> None:
    cols = st.columns(7)

    for col, (name, icon) in zip(cols, NAV_ITEMS):
        with col:
            active = st.session_state["page"] == name
            if st.button(
                f"{icon}  {name}",
                key=f"topnav_{name}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state["page"] = name
                st.rerun()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _sample_path() -> Path | None:
    for candidate in SAMPLE_CSV_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _normalize_uploaded_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Accept common real-world column names and map them to the backend schema:
    Date, Product, Sales, Inventory.
    """
    df = raw.copy()

    # Normalize only for matching; preserve data values.
    normalized = {
        str(col).strip().lower().replace(" ", "_").replace("-", "_"): col
        for col in df.columns
    }

    aliases = {
        "Date": [
            "date", "datetime", "timestamp", "order_date",
            "sales_date", "transaction_date", "day"
        ],
        "Product": [
            "product", "product_name", "item", "item_name",
            "sku", "product_id", "name"
        ],
        "Sales": [
            "sales", "sale", "quantity", "qty", "units",
            "units_sold", "demand", "demand_quantity", "sales_quantity"
        ],
        "Inventory": [
            "inventory", "stock", "stock_level", "current_stock",
            "available_stock", "inventory_level", "on_hand", "stock_on_hand"
        ],
    }

    rename = {}
    used = set()

    for standard, candidates in aliases.items():
        for candidate in candidates:
            if candidate in normalized and normalized[candidate] not in used:
                original = normalized[candidate]
                rename[original] = standard
                used.add(original)
                break

    df = df.rename(columns=rename)

    return df


def _generate_builtin_sample() -> pd.DataFrame:
    """
    Fallback sample generator so the app still works even if the sample CSV
    was not copied into the repository.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=180, freq="D")

    configs = {
        "Pepsi": (360, 0.10, 1.18, 22, 900),
        "Alpha Headphones": (120, 0.08, 1.10, 10, 450),
        "Beta Keyboard": (85, 0.05, 1.08, 8, 320),
    }

    rows = []

    for product_name, (base, trend_rate, weekend_factor, noise, starting_stock) in configs.items():
        inventory = starting_stock

        for i, date in enumerate(dates):
            dow = date.dayofweek
            weekly = 1.0

            if dow == 4:
                weekly = 1.12
            elif dow == 5:
                weekly = weekend_factor
            elif dow == 6:
                weekly = 1.10

            seasonal = 1 + 0.07 * np.sin(2 * np.pi * i / 30)
            trend = 1 + trend_rate * i / len(dates)

            sales = base * weekly * seasonal * trend
            sales += rng.normal(0, noise)

            if product_name == "Pepsi" and date in [
                pd.Timestamp("2026-02-14"),
                pd.Timestamp("2026-04-18"),
                pd.Timestamp("2026-05-16"),
            ]:
                sales *= 1.75

            if product_name == "Alpha Headphones" and date == pd.Timestamp("2026-03-20"):
                sales *= 2.0

            if product_name == "Beta Keyboard" and date == pd.Timestamp("2026-04-07"):
                sales *= 0.35

            sales = max(5, int(round(sales)))
            inventory -= sales

            if inventory < 150:
                inventory += starting_stock

            rows.append({
                "Date": date,
                "Product": product_name,
                "Sales": sales,
                "Inventory": inventory,
            })

    return pd.DataFrame(rows)


def load_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[str], list[str]]:
    """
    Data priority:
      1. User-uploaded CSV
      2. Explicitly selected sample dataset
      3. No data

    The uploaded CSV ALWAYS wins over sample data.
    """
    uploaded = st.session_state.get("uploaded_file")

    # ---------------------------------------------------------
    # USER UPLOAD — ALWAYS PRIORITY
    # ---------------------------------------------------------
    if uploaded is not None:
        try:
            # Reset pointer in case Streamlit/runtime has already read it.
            if hasattr(uploaded, "seek"):
                uploaded.seek(0)

            raw = pd.read_csv(uploaded)

            if raw.empty:
                return None, None, [], ["The uploaded CSV is empty."]

            raw = _normalize_uploaded_columns(raw)

            required = ["Date", "Product", "Sales"]
            missing = [col for col in required if col not in raw.columns]

            if missing:
                detected = ", ".join(map(str, raw.columns))
                return raw, None, [], [
                    "Could not identify required column(s): "
                    + ", ".join(missing)
                    + ". Your CSV must contain Date, Product and Sales. "
                    + f"Detected columns: {detected}"
                ]

            clean, warnings, errors = validate_and_clean(raw)

            st.session_state["data_source"] = "Uploaded CSV"

            return raw, clean, warnings, errors

        except Exception as exc:
            return None, None, [], [
                f"Could not read the uploaded CSV: {exc}"
            ]

    # ---------------------------------------------------------
    # SAMPLE — ONLY WHEN USER ENABLES IT
    # ---------------------------------------------------------
    if st.session_state.get("use_sample", False):
        sample_path = _sample_path()

        try:
            raw = (
                pd.read_csv(sample_path)
                if sample_path is not None
                else _generate_builtin_sample()
            )

            raw = _normalize_uploaded_columns(raw)
            clean, warnings, errors = validate_and_clean(raw)

            st.session_state["data_source"] = "Sample Dataset"

            return raw, clean, warnings, errors

        except Exception as exc:
            return None, None, [], [
                f"Could not load the sample dataset: {exc}"
            ]

    st.session_state["data_source"] = None
    return None, None, [], []


def render_data_source() -> None:
    st.markdown(
        '<div class="section-title">Data source</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3.5, 1.2])

    with c1:
        uploaded = st.file_uploader(
            "Upload historical sales CSV",
            type=["csv"],
            key="uploaded_file_widget",
            help="Required columns: Date, Product and Sales. Inventory is optional.",
        )

        # Explicitly copy the widget result into our application state.
        if uploaded is not None:
            st.session_state["uploaded_file"] = uploaded
            st.session_state["use_sample"] = False

    with c2:
        st.write("")
        st.write("")
        sample = st.checkbox(
            "Use sample dataset",
            value=st.session_state.get("use_sample", False),
            disabled=uploaded is not None,
            key="sample_dataset_checkbox",
        )

        if uploaded is None:
            st.session_state["use_sample"] = sample

    source = st.session_state.get("data_source")

    if source == "Uploaded CSV":
        st.success("✓ Using your uploaded CSV")

    elif source == "Sample Dataset":
        st.info("Using KARIGAR X sample dataset")

    elif uploaded is not None:
        st.info("Uploaded file received — validating columns...")

    else:
        st.info("Upload a CSV or select the sample dataset to begin.")

    # Main action: generate the forecast from the currently selected dataset.
    st.markdown('<div style="margin-top:18px"></div>', unsafe_allow_html=True)
    if st.button("✦ Generate Forecast", type="primary", use_container_width=True, key="generate_forecast_main"):
        with st.spinner("Analyzing demand and generating forecast..."):
            raw_check, clean_check, warnings_check, errors_check = load_data()

            if errors_check:
                for err in errors_check:
                    st.error(err)
                st.session_state["forecast_ready"] = False
            elif clean_check is None or clean_check.empty:
                st.warning("Please upload a valid sales CSV or select the sample dataset first.")
                st.session_state["forecast_ready"] = False
            else:
                st.session_state["forecast_ready"] = True
                st.session_state["data_source"] = (
                    "Uploaded CSV" if st.session_state.get("uploaded_file") is not None
                    else "Sample Dataset"
                )
                st.success("✓ Forecast generated successfully from the active dataset.")
                st.rerun()

# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def render_kpis(metrics: dict, current_inv: float, forecast: dict) -> None:
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            metric_card(
                "Σ",
                "Total Sales",
                f"{safe_number(metrics['total_sales']):,.0f}",
                "full historical period",
            ),
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            metric_card(
                "◷",
                "Average Demand",
                f"{safe_number(metrics['average_sales']):,.1f}",
                "units per day",
            ),
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            metric_card(
                "▣",
                "Current Inventory",
                f"{safe_number(current_inv):,.0f}",
                "units on hand",
            ),
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            metric_card(
                "↗",
                "Forecast Demand",
                f"{safe_number(forecast['total']):,.0f}",
                f"next {forecast['horizon']} days",
            ),
            unsafe_allow_html=True,
        )


def render_forecast_controls(product_list: list[str]) -> tuple[str, int, bool]:
    if (
        st.session_state.get("selected_product") not in product_list
        or st.session_state.get("selected_product") is None
    ):
        st.session_state["selected_product"] = product_list[0]

    st.markdown('<div class="section-title">Forecast controls</div>', unsafe_allow_html=True)

    with st.form("forecast_controls", clear_on_submit=False):
        c1, c2, c3 = st.columns([2.5, 1.2, 1.1])

        with c1:
            product = st.selectbox(
                "Product",
                product_list,
                index=product_list.index(st.session_state["selected_product"]),
                key="product_selector",
            )

        with c2:
            horizon = st.selectbox(
                "Forecast horizon",
                [7, 14, 30],
                index=[7, 14, 30].index(st.session_state["forecast_horizon"]),
                format_func=lambda x: f"{x} days",
                key="horizon_selector",
            )

        with c3:
            st.markdown("<div style='height: 27px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Generate Forecast",
                use_container_width=True,
                type="primary",
            )

    if submitted or st.session_state.get("selected_product") is None:
        st.session_state["selected_product"] = product
        st.session_state["forecast_horizon"] = horizon
        st.session_state["forecast_ready"] = True
    else:
        product = st.session_state["selected_product"]
        horizon = st.session_state["forecast_horizon"]

    return product, horizon, submitted


def render_dashboard(
    metrics: dict,
    current_inv: float,
    forecast: dict,
    history: pd.Series,
    anomalies: dict,
    trend: dict,
    seasonality: dict,
    inventory: dict,
    theme: dict,
    product: str,
) -> None:
    page_title(
        "Intelligent Demand Forecasting",
        "Understand what happened, predict what comes next, and turn the forecast into an inventory decision.",
    )

    st.markdown(
        f'<span class="data-chip">● {escape(product)} is active</span>',
        unsafe_allow_html=True,
    )

    render_kpis(metrics, current_inv, forecast)

    st.markdown('<div class="section-title">Demand performance</div>', unsafe_allow_html=True)
    left, right = st.columns([2.25, 1])

    with left:
        st.plotly_chart(
            sales_forecast_chart(history, forecast, anomalies, theme),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with right:
        peak = seasonality.get("peak_day") or "No strong pattern"
        st.markdown(
            textwrap.dedent(f"""
            <div class="insight-card">
                <div class="insight-title">Demand intelligence</div>
                <div class="insight-row">
                    <span class="insight-label">Trend</span>
                    <span class="insight-value">{trend_badge(trend["label"])}</span>
                </div>
                <div class="insight-row">
                    <span class="insight-label">Seasonality</span>
                    <span class="insight-value">{escape(str(peak))}</span>
                </div>
                <div class="insight-row">
                    <span class="insight-label">Anomalies</span>
                    <span class="insight-value">{int(anomalies["count"])} detected</span>
                </div>
                <div class="insight-row">
                    <span class="insight-label">Forecast method</span>
                    <span class="insight-value">{escape(str(forecast["method"]))}</span>
                </div>
                <div style="margin-top:14px;color:{theme["muted"]};font-size:11px;line-height:1.55;">
                    {escape(str(trend["explanation"]))}
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Inventory decision</div>', unsafe_allow_html=True)
    render_inventory_cards(inventory, theme)

    lc, rc = st.columns(2)

    with lc:
        st.markdown('<div class="section-title">Demand by weekday</div>', unsafe_allow_html=True)
        st.plotly_chart(weekday_chart(seasonality, theme), use_container_width=True)
    with rc:
        st.markdown('<div class="section-title">Trend & seasonality</div>', unsafe_allow_html=True)
        st.plotly_chart(trend_seasonality_chart(history, theme), use_container_width=True)
    st.markdown('<div class="section-title">AI demand summary</div>', unsafe_allow_html=True)
    summary = generate_ai_summary(trend, seasonality, anomalies, forecast, inventory)
    st.markdown(
        textwrap.dedent(f"""
        <div class="ai-card">
            <div class="ai-title">✦ KARIGAR X Intelligence</div>
            <p class="ai-copy">{escape(summary)}</p>
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Forecast details</div>', unsafe_allow_html=True)
    render_forecast_table(forecast)


def render_inventory_cards(inventory: dict, theme: dict) -> None:
    i1, i2, i3 = st.columns(3)

    with i1:
        st.markdown(
            metric_card(
                "▣",
                "Current Inventory",
                f"{safe_number(inventory['current_inventory']):,.0f}",
                "units on hand",
            ),
            unsafe_allow_html=True,
        )

    with i2:
        st.markdown(
            metric_card(
                "↗",
                "Recommended Stock",
                f"{safe_number(inventory['recommended_stock']):,.0f}",
                f"{safe_number(inventory['safety_pct']) * 100:.0f}% safety buffer",
            ),
            unsafe_allow_html=True,
        )

    with i3:
        st.markdown(
            metric_card(
                "↻",
                "Forecast Demand",
                f"{safe_number(inventory['forecast_demand']):,.0f}",
                "selected forecast horizon",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    order = safe_number(inventory["order_quantity"])
    status = inventory.get("status", "ok")

    st.markdown(
        textwrap.dedent(f"""
        <div class="recommend-card">
            <div class="recommend-label">Recommended replenishment</div>
            <div class="recommend-number">{order:,.0f}</div>
            <div class="recommend-sub">
                {"units to order now" if order > 0 else "no additional order required right now"}
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    if status == "stockout":
        st.error(inventory["warning"])
    elif status == "overstock":
        st.warning(inventory["warning"])
    else:
        st.success(inventory["warning"])

    st.caption(inventory["disclaimer"])


def render_forecast_table(forecast: dict) -> None:
    table = forecast["frame"].copy()
    for col in ["Predicted", "Lower", "Upper"]:
        if col in table.columns:
            table[col] = table[col].round(1)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=280,
    )

    st.download_button(
        "Download forecast CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="demandai_forecast.csv",
        mime="text/csv",
        use_container_width=False,
    )


def render_analytics(
    trend: dict,
    seasonality: dict,
    history: pd.Series,
    theme: dict,
) -> None:
    page_title(
        "Sales Analytics",
        "Explore historical demand, growth, weekday behavior and recurring patterns.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Trend", trend["label"])
    with c2:
        st.metric("Recent average / day", f"{safe_number(trend['recent_avg']):.1f}")
    with c3:
        st.metric("Previous average / day", f"{safe_number(trend['previous_avg']):.1f}")

    st.markdown('<div class="section-title">Trend analysis</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card"><div style="color:{theme["muted"]};font-size:13px;line-height:1.7;">'
        f'{escape(str(trend["explanation"]))}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Seasonality analysis</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card"><div style="color:{theme["muted"]};font-size:13px;line-height:1.7;">'
        f'{escape(str(seasonality["explanation"]))}</div></div>',
        unsafe_allow_html=True,
    )

    lc, rc = st.columns(2)

    with lc:
        st.markdown('<div class="section-title">Demand by weekday</div>', unsafe_allow_html=True)
        st.plotly_chart(weekday_chart(seasonality, theme), use_container_width=True)
    with rc:
        st.markdown('<div class="section-title">Rolling demand trend</div>', unsafe_allow_html=True)
        st.plotly_chart(trend_seasonality_chart(history, theme), use_container_width=True)
def render_forecast_page(
    history: pd.Series,
    forecast: dict,
    anomalies: dict,
    theme: dict,
) -> None:
    page_title(
        "Demand Forecast",
        "Generate and inspect future demand predictions for the selected product.",
    )

    st.markdown(
        textwrap.dedent(f"""
        <div class="card">
            <div class="insight-row">
                <span class="insight-label">Model</span>
                <span class="insight-value">{escape(str(forecast["method"]))}</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Forecast horizon</span>
                <span class="insight-value">{forecast["horizon"]} days</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Total predicted demand</span>
                <span class="insight-value">{safe_number(forecast["total"]):,.0f} units</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Average predicted demand</span>
                <span class="insight-value">{safe_number(forecast["average"]):,.1f} units/day</span>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    if forecast.get("fallback_note"):
        st.info(forecast["fallback_note"])

    st.markdown('<div class="section-title">Forecast curve</div>', unsafe_allow_html=True)
    st.plotly_chart(
        sales_forecast_chart(history, forecast, anomalies, theme),
        use_container_width=True,
    )
    st.markdown('<div class="section-title">Forecast table</div>', unsafe_allow_html=True)
    render_forecast_table(forecast)


def render_inventory(
    inventory: dict,
    history: pd.Series,
    forecast: dict,
    theme: dict,
) -> None:
    page_title(
        "Inventory Intelligence",
        "Translate predicted demand into a transparent replenishment recommendation.",
    )

    render_inventory_cards(inventory, theme)

    st.markdown('<div class="section-title">How the recommendation is calculated</div>', unsafe_allow_html=True)
    st.markdown(
        textwrap.dedent(f"""
        <div class="card">
            <div class="insight-row">
                <span class="insight-label">Forecast demand</span>
                <span class="insight-value">{safe_number(inventory["forecast_demand"]):,.0f} units</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Safety stock</span>
                <span class="insight-value">{safe_number(inventory["safety_stock"]):,.0f} units</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Recommended stock</span>
                <span class="insight-value">{safe_number(inventory["recommended_stock"]):,.0f} units</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Current inventory</span>
                <span class="insight-value">{safe_number(inventory["current_inventory"]):,.0f} units</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">Recommended order</span>
                <span class="insight-value">{safe_number(inventory["order_quantity"]):,.0f} units</span>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Demand versus inventory</div>', unsafe_allow_html=True)

    frame = forecast["frame"].copy()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["Predicted"],
            mode="lines+markers",
            name="Forecast demand",
            line=dict(color=theme["accent"], width=2.5),
        )
    )

    fig.add_hline(
        y=safe_number(inventory["current_inventory"]),
        line_dash="dot",
        line_color=theme["chart_forecast"],
        annotation_text="Current inventory",
        annotation_font=dict(color=theme["muted"]),
    )

    fig.add_hline(
        y=safe_number(inventory["recommended_stock"]),
        line_dash="dash",
        line_color=theme["accent"],
        annotation_text="Recommended stock",
        annotation_font=dict(color=theme["muted"]),
    )

    fig.update_layout(xaxis_title="Forecast date", yaxis_title="Units")
    st.plotly_chart(configure_plot(fig, theme, 390), use_container_width=True)
def render_anomalies(
    history: pd.Series,
    anomalies: dict,
    theme: dict,
) -> None:
    page_title(
        "Anomaly Detection",
        "Find unusual demand spikes and drops without hiding them from the forecast.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Anomalies detected", int(anomalies["count"]))
    with c2:
        st.metric("Detection method", str(anomalies["method"]).upper())
    with c3:
        st.metric(
            "Typical demand range",
            f"{safe_number(anomalies['normal_low']):.0f}–{safe_number(anomalies['normal_high']):.0f}",
        )

    st.markdown('<div class="section-title">Anomaly map</div>', unsafe_allow_html=True)
    st.plotly_chart(
        anomaly_chart(history, anomalies, theme),
        use_container_width=True,
    )
    st.markdown(
        f'<div class="section-title">Flagged days ({int(anomalies["count"])})</div>',
        unsafe_allow_html=True,
    )

    points = anomalies.get("points", pd.DataFrame())
    if points.empty:
        st.success("No anomalies detected in the current historical demand.")
    else:
        st.dataframe(points, use_container_width=True, hide_index=True)


def render_data(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    product_df: pd.DataFrame,
    warnings: list[str],
) -> None:
    page_title(
        "Dataset & Quality",
        "Validate the input data, inspect cleaned records and export the processed dataset.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Raw rows", len(raw))
    with c2:
        st.metric("Clean rows", len(clean))
    with c3:
        st.metric("Selected product rows", len(product_df))

    st.markdown('<div class="section-title">Data quality</div>', unsafe_allow_html=True)
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No data quality warnings were reported by the validation pipeline.")

    st.markdown('<div class="section-title">Cleaned data</div>', unsafe_allow_html=True)
    st.dataframe(
        product_df,
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    st.download_button(
        "Download cleaned data CSV",
        data=clean.to_csv(index=False).encode("utf-8"),
        file_name="demandai_cleaned_data.csv",
        mime="text/csv",
    )


def render_settings(theme: dict) -> None:
    page_title(
        "Settings",
        "Control anomaly detection and safety-stock behavior used across KARIGAR X.",
    )

    st.markdown('<div class="section-title">Appearance</div>', unsafe_allow_html=True)
    st.markdown(
        textwrap.dedent(f"""
        <div class="card">
            <div class="insight-row">
                <span class="insight-label">Active theme</span>
                <span class="insight-value">{escape(st.session_state["theme"])}</span>
            </div>
            <div style="color:{theme["muted"]};font-size:12px;line-height:1.6;margin-top:10px;">
                Use the Theme selector in the top-right corner to switch between
                Dark, Light and Glass modes.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Anomaly detection</div>', unsafe_allow_html=True)
    with st.container():
        st.radio(
            "Detection method",
            options=["iqr", "zscore"],
            format_func=lambda x: "IQR (Interquartile Range)" if x == "iqr" else "Z-score",
            key="anomaly_method",
            horizontal=True,
        )
        st.caption("This setting affects anomaly detection and the forecast preprocessing pipeline.")

    st.markdown('<div class="section-title">Safety stock</div>', unsafe_allow_html=True)
    with st.container():
        st.radio(
            "Safety-stock mode",
            options=["auto", "manual"],
            format_func=lambda x: (
                "Automatic — demand variability"
                if x == "auto"
                else "Manual — percentage override"
            ),
            key="safety_mode",
            horizontal=True,
        )

        st.slider(
            "Manual safety-stock percentage",
            min_value=5,
            max_value=50,
            key="safety_override_pct",
            disabled=st.session_state["safety_mode"] != "manual",
        )

        st.caption(
            "Automatic mode uses the backend's demand-variability calculation. "
            "Manual mode supplies the selected percentage to the inventory engine."
        )


def render_no_data() -> None:
    st.markdown(
        textwrap.dedent("""
        <div class="empty-state">
            <div class="empty-title">Connect a sales dataset</div>
            <div class="empty-copy">
                Upload a CSV above or enable the sample dataset.
                Once valid data is available, every KARIGAR X feature will use the
                same processed dataset and backend calculations.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    init_state()

    theme = THEMES[st.session_state["theme"]]
    inject_css(theme)

    render_topbar()
    render_navigation()
    render_data_source()

    raw, clean, warnings, errors = load_data()

    if errors:
        for error in errors:
            st.error(error)
        return

    if raw is None or clean is None:
        render_no_data()
        return

    if clean.empty:
        st.warning("The cleaned dataset contains no usable rows.")
        return

    for warning in warnings:
        st.warning(warning)

    product_list = products(clean)
    if not product_list:
        st.error("No valid products were found in the dataset.")
        return

    product, horizon, _ = render_forecast_controls(product_list)

    product_df = filter_product(clean, product)
    history = daily_series(product_df)

    if history.empty:
        st.error("No valid daily demand history is available for the selected product.")
        return

    # Single source of truth: all pages use these backend calculations.
    metrics = historical_metrics(history)
    current_inv = latest_inventory(product_df)

    anomalies = detect_anomalies(
        history,
        method=st.session_state["anomaly_method"],
    )

    train = winsorize_for_forecast(
        history,
        anomalies["lower"],
        anomalies["upper"],
    )

    forecast = forecast_demand(train, horizon=horizon)
    trend = detect_trend(history)
    seasonality = detect_seasonality(history)

    override = (
        st.session_state["safety_override_pct"] / 100.0
        if st.session_state["safety_mode"] == "manual"
        else None
    )

    inventory = recommend_inventory(
        forecast["total"],
        current_inv,
        history,
        override_safety_pct=override,
    )

    page = st.session_state["page"]

    if page == "Dashboard":
        render_dashboard(
            metrics,
            current_inv,
            forecast,
            history,
            anomalies,
            trend,
            seasonality,
            inventory,
            theme,
            product,
        )

    elif page == "Analytics":
        render_analytics(trend, seasonality, history, theme)

    elif page == "Forecast":
        render_forecast_page(history, forecast, anomalies, theme)

    elif page == "Inventory":
        render_inventory(inventory, history, forecast, theme)

    elif page == "Anomalies":
        render_anomalies(history, anomalies, theme)

    elif page == "Data":
        render_data(raw, clean, product_df, warnings)

    elif page == "Settings":
        render_settings(theme)


if __name__ == "__main__":
    main()
