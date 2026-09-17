import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DemandAI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# SESSION STATE
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# ============================================================
# THEME
# ============================================================

themes = {
    "Dark": {
        "bg": "#07110D",
        "surface": "#0D1B15",
        "surface2": "#12241C",
        "text": "#F4F8F5",
        "muted": "#9BAEA4",
        "accent": "#27C477",
        "accent2": "#1FA463",
        "border": "#1D3328",
        "chart": "#27C477"
    },

    "Light": {
        "bg": "#F5F8F6",
        "surface": "#FFFFFF",
        "surface2": "#EEF4F0",
        "text": "#10231A",
        "muted": "#66756E",
        "accent": "#159957",
        "accent2": "#0F7D45",
        "border": "#DDE7E1",
        "chart": "#159957"
    },

    "Glass": {
        "bg": "#071411",
        "surface": "rgba(255,255,255,0.07)",
        "surface2": "rgba(255,255,255,0.10)",
        "text": "#F7FFFA",
        "muted": "#A7BDB3",
        "accent": "#45E39A",
        "accent2": "#25B977",
        "border": "rgba(255,255,255,0.12)",
        "chart": "#45E39A"
    }
}

theme = themes[st.session_state.theme]


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {{
        background: {theme["bg"]};
        color: {theme["text"]};
    }}

    .main .block-container {{
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }}

    /* Hide Streamlit sidebar */

    section[data-testid="stSidebar"] {{
        display: none;
    }}

    /* Hide default menu */

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    header {{
        background: transparent !important;
    }}

    /* ---------- TOP NAV ---------- */

    .top-nav {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 20px;
        margin-bottom: 35px;

        background: {theme["surface"]};
        border: 1px solid {theme["border"]};
        border-radius: 18px;
    }}

    .brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 21px;
        font-weight: 800;
        color: {theme["text"]};
    }}

    .brand-icon {{
        width: 38px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 11px;

        background: {theme["accent"]};
        color: #06120C;

        font-weight: 900;
        font-size: 20px;
    }}

    .brand-sub {{
        color: {theme["muted"]};
        font-size: 11px;
        font-weight: 500;
    }}

    /* ---------- HERO ---------- */

    .hero {{
        padding: 15px 0 30px 0;
    }}

    .hero-title {{
        font-size: 42px;
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -1.5px;
        color: {theme["text"]};
        margin-bottom: 10px;
    }}

    .hero-description {{
        color: {theme["muted"]};
        font-size: 16px;
        max-width: 650px;
        line-height: 1.6;
    }}

    .status {{
        display: inline-flex;
        align-items: center;
        gap: 7px;

        margin-top: 18px;
        padding: 7px 12px;

        border-radius: 999px;

        background: {theme["surface2"]};
        border: 1px solid {theme["border"]};

        color: {theme["muted"]};
        font-size: 12px;
    }}

    .status-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: {theme["accent"]};
    }}

    /* ---------- CARDS ---------- */

    .metric-card {{
        background: {theme["surface"]};
        border: 1px solid {theme["border"]};

        border-radius: 18px;
        padding: 22px;

        min-height: 145px;

        transition: all 0.2s ease;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        border-color: {theme["accent"]};
    }}

    .metric-label {{
        color: {theme["muted"]};
        font-size: 13px;
        margin-bottom: 12px;
    }}

    .metric-value {{
        color: {theme["text"]};
        font-size: 29px;
        font-weight: 800;
        letter-spacing: -0.8px;
    }}

    .metric-description {{
        color: {theme["muted"]};
        font-size: 12px;
        margin-top: 8px;
    }}

    .positive {{
        color: {theme["accent"]};
    }}

    /* ---------- SECTION TITLE ---------- */

    .section-title {{
        font-size: 19px;
        font-weight: 750;
        color: {theme["text"]};
        margin-top: 32px;
        margin-bottom: 14px;
    }}

    /* ---------- INSIGHT CARD ---------- */

    .insight-card {{
        background: {theme["surface"]};
        border: 1px solid {theme["border"]};
        border-radius: 18px;
        padding: 24px;
        min-height: 230px;
    }}

    .insight-heading {{
        color: {theme["text"]};
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 18px;
    }}

    .insight-row {{
        display: flex;
        justify-content: space-between;
        padding: 13px 0;

        border-bottom: 1px solid {theme["border"]};

        color: {theme["muted"]};
        font-size: 13px;
    }}

    .insight-value {{
        color: {theme["text"]};
        font-weight: 700;
    }}

    /* ---------- INVENTORY ---------- */

    .inventory-card {{
        background: linear-gradient(
            135deg,
            {theme["surface"]},
            {theme["surface2"]}
        );

        border: 1px solid {theme["border"]};

        border-radius: 18px;
        padding: 24px;
    }}

    .inventory-number {{
        font-size: 40px;
        font-weight: 850;
        color: {theme["accent"]};
        margin: 10px 0;
    }}

    .inventory-label {{
        color: {theme["muted"]};
        font-size: 13px;
    }}

    /* ---------- STREAMLIT BUTTONS ---------- */

    .stButton > button {{
        border-radius: 10px !important;
        border: 1px solid {theme["border"]} !important;

        background: {theme["surface"]} !important;
        color: {theme["text"]} !important;

        font-weight: 650 !important;
    }}

    .stButton > button:hover {{
        border-color: {theme["accent"]} !important;
        color: {theme["accent"]} !important;
    }}

    /* ---------- SELECTBOX ---------- */

    div[data-baseweb="select"] > div {{
        background: {theme["surface"]} !important;
        border-color: {theme["border"]} !important;
        color: {theme["text"]} !important;
        border-radius: 10px !important;
    }}

    /* ---------- FILE UPLOADER ---------- */

    [data-testid="stFileUploader"] {{
        background: {theme["surface"]};
        border: 1px dashed {theme["accent"]};
        border-radius: 16px;
        padding: 8px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TOP NAVIGATION
# ============================================================

nav1, nav2, nav3, nav4, nav5, nav6, nav7, nav8 = st.columns(
    [2.4, 1, 1, 1, 1, 1, 1, 0.9]
)

with nav1:
    st.markdown(
        f"""
        <div class="brand">
            <div class="brand-icon">◈</div>
            <div>
                DemandAI
                <div class="brand-sub">Demand Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

pages = [
    ("Dashboard", nav2),
    ("Analytics", nav3),
    ("Forecast", nav4),
    ("Inventory", nav5),
    ("Anomalies", nav6),
    ("Data", nav7),
]

for page_name, column in pages:
    with column:
        if st.button(
            page_name,
            key=f"nav_{page_name}",
            use_container_width=True
        ):
            st.session_state.page = page_name
            st.rerun()

with nav8:
    theme_choice = st.selectbox(
        "Theme",
        ["Dark", "Light", "Glass"],
        index=["Dark", "Light", "Glass"].index(
            st.session_state.theme
        ),
        label_visibility="collapsed"
    )

    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()


# ============================================================
# DEMO DATA
# Replace this section with your actual backend data
# ============================================================

dates = pd.date_range(
    end=pd.Timestamp.today(),
    periods=90
)

np.random.seed(42)

sales = (
    80
    + np.linspace(0, 15, 90)
    + 10 * np.sin(np.arange(90) * 2 * np.pi / 7)
    + np.random.normal(0, 6, 90)
)

sales = np.maximum(sales, 5)

forecast_dates = pd.date_range(
    start=dates[-1] + pd.Timedelta(days=1),
    periods=14
)

forecast = np.linspace(
    sales[-7:].mean(),
    sales[-7:].mean() + 5,
    14
)

current_inventory = 344
forecast_demand = round(forecast.sum())
safety_stock = 150
recommended_stock = forecast_demand + safety_stock
order_quantity = max(
    0,
    recommended_stock - current_inventory
)


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Intelligent Demand Forecasting
            </div>

            <div class="hero-description">
                Turn historical sales into smarter inventory decisions
                using demand analytics, forecasting and intelligent
                replenishment recommendations.
            </div>

            <div class="status">
                <span class="status-dot"></span>
                Forecast engine ready
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Controls

    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        product = st.selectbox(
            "Product",
            ["Alpha Headphones", "Beta Keyboard", "Gamma Mouse"]
        )

    with c2:
        horizon = st.selectbox(
            "Forecast",
            ["7 Days", "14 Days", "30 Days"]
        )

    with c3:
        st.write("")
        st.write("")
        st.button(
            "✦ Generate Forecast",
            use_container_width=True
        )

    # KPI cards

    st.markdown(
        '<div class="section-title">Overview</div>',
        unsafe_allow_html=True
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">TOTAL SALES</div>
                <div class="metric-value">33,385</div>
                <div class="metric-description positive">
                    ↑ 8.4% from previous period
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">AVERAGE DAILY DEMAND</div>
                <div class="metric-value">91.7</div>
                <div class="metric-description">
                    units / day
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">FORECAST DEMAND</div>
                <div class="metric-value">{forecast_demand:,}</div>
                <div class="metric-description">
                    next 14 days
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">CURRENT INVENTORY</div>
                <div class="metric-value">{current_inventory:,}</div>
                <div class="metric-description">
                    units on hand
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Main chart

    st.markdown(
        '<div class="section-title">Demand Performance</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sales,
            name="Historical Demand",
            mode="lines",
            line=dict(
                color=theme["chart"],
                width=2.5
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast,
            name="Forecast",
            mode="lines",
            line=dict(
                color="#FFB454",
                width=3,
                dash="dot"
            )
        )
    )

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color=theme["muted"]
        ),

        xaxis=dict(
            showgrid=False
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor=theme["border"]
        ),

        legend=dict(
            orientation="h",
            y=1.08,
            x=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Bottom cards

    left, right = st.columns(2)

    with left:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-heading">
                    ✦ Demand Intelligence
                </div>

                <div class="insight-row">
                    <span>Trend</span>
                    <span class="insight-value positive">
                        Increasing ↗
                    </span>
                </div>

                <div class="insight-row">
                    <span>Seasonality</span>
                    <span class="insight-value">
                        Saturday Peak
                    </span>
                </div>

                <div class="insight-row">
                    <span>Anomalies</span>
                    <span class="insight-value">
                        21 detected
                    </span>
                </div>

                <div class="insight-row">
                    <span>Forecast</span>
                    <span class="insight-value">
                        Elevated demand
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            f"""
            <div class="inventory-card">

                <div class="insight-heading">
                    📦 Inventory Recommendation
                </div>

                <div class="insight-row">
                    <span>Current Stock</span>
                    <span class="insight-value">
                        {current_inventory:,}
                    </span>
                </div>

                <div class="insight-row">
                    <span>Forecast Demand</span>
                    <span class="insight-value">
                        {forecast_demand:,}
                    </span>
                </div>

                <div class="insight-row">
                    <span>Safety Stock</span>
                    <span class="insight-value">
                        {safety_stock:,}
                    </span>
                </div>

                <div class="inventory-label">
                    Recommended Order
                </div>

                <div class="inventory-number">
                    {order_quantity:,} units
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# OTHER PAGES
# ============================================================

elif st.session_state.page == "Analytics":

    st.markdown(
        '<div class="hero-title">Sales Analytics</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore historical demand, trends and recurring patterns."
    )

    st.info(
        "Connect this page to your existing analytics backend."
    )


elif st.session_state.page == "Forecast":

    st.markdown(
        '<div class="hero-title">Demand Forecast</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Generate future demand predictions for your selected product."
    )

    st.info(
        "Connect this page to your existing forecasting function."
    )


elif st.session_state.page == "Inventory":

    st.markdown(
        '<div class="hero-title">Inventory Intelligence</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Turn demand predictions into replenishment recommendations."
    )

    st.info(
        "Connect this page to your existing inventory calculation."
    )


elif st.session_state.page == "Anomalies":

    st.markdown(
        '<div class="hero-title">Anomaly Detection</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Identify unusual demand spikes and drops."
    )


elif st.session_state.page == "Data":

    st.markdown(
        '<div class="hero-title">Dataset</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload historical sales CSV",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.success(
            f"Dataset loaded: {len(df):,} rows"
        )

        st.dataframe(
            df.head(20),
            use_container_width=True
        )