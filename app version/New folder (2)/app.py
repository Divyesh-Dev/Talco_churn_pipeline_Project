# ============================================================
# app.py  —  Telco Churn Analytics  |  Figma Dark Theme
# ============================================================

import os, sys, json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from src.config import DB_URL, TABLE_CLEANED, PROCESSED_DATA
from src.auth   import (is_logged_in, is_admin, current_user, logout,
                         get_all_users, add_user, delete_user,
                         change_password, get_access_log, log_event)

MODELS_DIR   = os.path.join(_ROOT, "models")
SCORED_CSV   = os.path.join(_ROOT, "data", "processed", "churn_scored.csv")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'/>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
# GLOBAL CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── base ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
.main, .block-container {
    background: #0d1117 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.block-container { padding:0 !important; max-width:100% !important; }

/* hide streamlit chrome but KEEP the sidebar collapse arrow */
header[data-testid="stHeader"] { display:none !important; }
#MainMenu, footer               { display:none !important; }

/* show the sidebar toggle button */
[data-testid="collapsedControl"] {
    display:flex !important;
    background:#161b27 !important;
    color:#8892a4 !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    border-radius:0 8px 8px 0 !important;
    top:20px !important;
}
[data-testid="collapsedControl"]:hover { color:#fff !important; background:#1a2035 !important; }

/* ── sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] > div:first-child { padding:0 !important; }
[data-testid="stSidebarContent"]            { padding:0 !important; }

/* sidebar labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #8892a4 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}

/* ── inputs ───────────────────────────────────────────── */
input, textarea,
[data-testid="stTextInput"]   input,
[data-testid="stNumberInput"] input {
    background: #1e2a3a !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    transition: border-color .2s, box-shadow .2s !important;
}
input::placeholder { color: #4a5568 !important; }
input:focus {
    border-color: #1a6ef5 !important;
    box-shadow: 0 0 0 3px rgba(26,110,245,0.18) !important;
    outline: none !important;
}

/* selectbox */
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div {
    background: #1e2a3a !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
[data-baseweb="popover"] li {
    background: #1a2035 !important;
    color: #e2e8f0 !important;
}
[data-baseweb="popover"] li:hover { background: #1e2a3a !important; }

/* slider */
[data-testid="stSlider"] > div { color:#8892a4 !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#1a6ef5 !important; }

/* multiselect */
[data-testid="stMultiSelect"] > div > div {
    background: #1e2a3a !important;
    border-color: rgba(255,255,255,0.1) !important;
}
span[data-baseweb="tag"] {
    background: rgba(26,110,245,0.2) !important;
    color: #60a5fa !important;
}

/* number input arrows */
[data-testid="stNumberInput"] button {
    background: #1e2a3a !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
}

/* ── buttons ──────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg,#1a6ef5,#00d4ff) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 11px 20px !important;
    letter-spacing: 0.01em !important;
    transition: opacity .2s, transform .15s, box-shadow .2s !important;
    box-shadow: 0 4px 14px rgba(26,110,245,0.25) !important;
}
.stButton > button:hover {
    opacity: .92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(26,110,245,0.35) !important;
}
.stButton > button:active { transform:translateY(0) !important; }

/* ── metrics ──────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #1a2035 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 20px 22px !important;
    transition: border-color .2s !important;
}
[data-testid="stMetric"]:hover { border-color: rgba(26,110,245,0.3) !important; }
[data-testid="stMetricLabel"] p { color:#8892a4 !important; font-size:0.8rem !important; }
[data-testid="stMetricValue"]   { color:#fff !important; font-size:1.75rem !important; font-weight:700 !important; }

/* ── dataframe ────────────────────────────────────────── */
[data-testid="stDataFrame"],
.dvn-scroller { background:#1a2035 !important; border-radius:10px !important; }

/* ── tabs ─────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #1a2035 !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 6px 6px 0 !important;
    gap: 4px !important;
    border-bottom: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #8892a4 !important;
    border-radius: 7px 7px 0 0 !important;
    font-size: 0.84rem !important;
    padding: 9px 18px !important;
    border-bottom: 2px solid transparent !important;
    transition: color .15s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(26,110,245,0.12) !important;
    color: #60a5fa !important;
    border-bottom: 2px solid #1a6ef5 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none !important; }
[data-testid="stTabPanel"] {
    background: #1a2035 !important;
    border-radius: 0 10px 10px 10px !important;
    padding: 24px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── hr ───────────────────────────────────────────────── */
hr { border-color:rgba(255,255,255,0.06) !important; margin:20px 0 !important; }

/* ── alerts ───────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── component classes ────────────────────────────────── */
.page-header   { padding:28px 32px 0; }
.page-title    { font-size:1.55rem; font-weight:700; color:#fff; margin-bottom:5px; }
.page-subtitle { font-size:0.84rem; color:#8892a4; margin-bottom:20px; }
.content-wrap  { padding:20px 32px 40px; }
.section-title { font-size:0.95rem; font-weight:600; color:#fff; margin:22px 0 14px; }

.kpi-card {
    background:#1a2035;
    border:1px solid rgba(255,255,255,0.06);
    border-radius:12px; padding:20px 22px;
    display:flex; justify-content:space-between; align-items:flex-start;
    transition:border-color .2s, transform .2s;
}
.kpi-card:hover { border-color:rgba(26,110,245,0.25); transform:translateY(-1px); }
.kpi-label     { font-size:0.78rem; color:#8892a4; margin-bottom:8px; font-weight:500; }
.kpi-value     { font-size:1.65rem; font-weight:700; color:#fff; line-height:1; }
.kpi-delta-pos { color:#00e676; font-size:0.75rem; margin-top:6px; }
.kpi-delta-neg { color:#ff5252; font-size:0.75rem; margin-top:6px; }
.kpi-icon {
    width:42px; height:42px; border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-size:1rem; font-weight:700; flex-shrink:0;
    font-family:'Inter',sans-serif;
}
.icon-blue  { background:rgba(26,110,245,0.18); color:#60a5fa; }
.icon-red   { background:rgba(255,82,82,0.18);  color:#ff8a80; }
.icon-green { background:rgba(0,230,118,0.15);  color:#00e676; }
.icon-amber { background:rgba(255,171,0,0.15);  color:#ffab00; }

.chart-card {
    background:#1a2035;
    border:1px solid rgba(255,255,255,0.06);
    border-radius:12px; padding:20px 22px;
    margin-bottom:16px;
}
.chart-title { font-size:0.9rem; font-weight:600; color:#fff; margin-bottom:14px; }

.offer-card {
    background:#1a2035;
    border:1px solid rgba(255,255,255,0.07);
    border-radius:12px; padding:22px;
    height:100%;
    transition:border-color .2s, transform .2s;
}
.offer-card:hover { border-color:rgba(26,110,245,0.25); transform:translateY(-2px); }
.offer-icon-box {
    width:44px; height:44px; border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-size:0.85rem; font-weight:700; margin-bottom:14px;
}
.offer-title    { font-size:0.9rem; font-weight:600; color:#fff; margin-bottom:4px; }
.offer-subtitle { font-size:0.75rem; color:#8892a4; margin-bottom:16px; }
.offer-row      { display:flex; justify-content:space-between; font-size:0.78rem; margin-bottom:7px; }
.offer-key      { color:#8892a4; }
.offer-val-pos  { color:#00e676; font-weight:600; }
.offer-val      { color:#e2e8f0; }

.badge-active   { background:rgba(0,230,118,0.15); color:#00e676; font-size:0.68rem; font-weight:700; padding:3px 9px; border-radius:5px; letter-spacing:.03em; }
.badge-proposed { background:rgba(26,110,245,0.15); color:#60a5fa; font-size:0.68rem; font-weight:700; padding:3px 9px; border-radius:5px; letter-spacing:.03em; }
.badge-admin    { background:rgba(26,110,245,0.18); color:#60a5fa; font-size:0.7rem; font-weight:600; padding:2px 9px; border-radius:5px; }
.badge-user     { background:rgba(0,230,118,0.15); color:#00e676; font-size:0.7rem; font-weight:600; padding:2px 9px; border-radius:5px; }
.badge-high     { background:rgba(255,82,82,0.15); color:#ff5252; font-size:0.75rem; font-weight:600; padding:3px 10px; border-radius:6px; }
.badge-medium   { background:rgba(255,171,0,0.15); color:#ffab00; font-size:0.75rem; font-weight:600; padding:3px 10px; border-radius:6px; }
.badge-low      { background:rgba(0,230,118,0.15); color:#00e676; font-size:0.75rem; font-weight:600; padding:3px 10px; border-radius:6px; }

.progress-wrap      { background:rgba(255,255,255,0.08); border-radius:4px; height:5px; margin-top:12px; }
.progress-green     { height:5px; border-radius:4px; background:linear-gradient(90deg,#00e676,#00b894); }
.progress-blue      { height:5px; border-radius:4px; background:linear-gradient(90deg,#1a6ef5,#00d4ff); }
.progress-gray      { height:5px; border-radius:4px; background:rgba(255,255,255,0.22); }

.avatar {
    width:34px; height:34px; border-radius:50%;
    background:linear-gradient(135deg,#1a6ef5,#00d4ff);
    display:inline-flex; align-items:center; justify-content:center;
    color:#fff; font-size:0.72rem; font-weight:700;
    flex-shrink:0; letter-spacing:.02em;
}

.user-row {
    display:grid; grid-template-columns:2fr 1fr 1fr;
    padding:14px 20px; align-items:center;
    border-bottom:1px solid rgba(255,255,255,0.04);
    transition:background .15s;
}
.user-row:hover { background:rgba(255,255,255,0.02); }

.footer-bar {
    margin-top:40px;
    border-top:1px solid rgba(255,255,255,0.06);
    padding:18px 32px;
    text-align:center;
}
.footer-title { font-size:0.73rem; font-weight:600; color:#8892a4; margin-bottom:4px; letter-spacing:.01em; }
.footer-sub   { font-size:0.68rem; color:#4a5568; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# AUTH GATE  —  login page
# ════════════════════════════════════════════════════════════
if not is_logged_in():
    st.markdown("""
    <style>
    /* Full-page login background */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    .main { background: radial-gradient(ellipse at 30% 50%, #0c1e5c 0%, #0d1117 65%) !important; }
    .block-container { padding:0 !important; }

    /* tighten default column gap */
    [data-testid="column"] { padding:0 !important; }

    /* hide sidebar on login */
    [data-testid="stSidebar"]        { display:none !important; }
    [data-testid="collapsedControl"] { display:none !important; }

    /* login card */
    .login-card {
        background: #161b27;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 40px 40px 36px;
    }
    .login-logo {
        width:60px; height:60px; border-radius:14px;
        background:linear-gradient(135deg,#1a6ef5,#00d4ff);
        display:flex; align-items:center; justify-content:center;
        margin:0 auto 18px;
        font-size:1.4rem; font-family:'Inter',sans-serif;
        font-weight:700; color:#fff; letter-spacing:-.02em;
    }
    .login-title { font-size:1.35rem; font-weight:700; color:#fff; text-align:center; margin-bottom:5px; }
    .login-sub   { font-size:0.8rem; color:#8892a4; text-align:center; margin-bottom:28px; }
    .login-label { font-size:0.8rem; font-weight:500; color:#c8d0dc; margin-bottom:6px; display:block; }
    .login-footer { font-size:0.72rem; color:#4a5568; text-align:center; margin-top:20px; }
    .login-footer a { color:#1a6ef5; text-decoration:none; }
    .login-footer a:hover { text-decoration:underline; }
    </style>
    """, unsafe_allow_html=True)

    # Centred layout: 1 - 1.6 - 1
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        # Top spacing
        st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="login-card">
          <div class="login-logo">TA</div>
          <div class="login-title">Telco Churn Analytics</div>
          <div class="login-sub">ETL Dashboard &amp; Prediction Platform</div>
        </div>
        """, unsafe_allow_html=True)

        # Fields rendered INSIDE the card area using st.container
        with st.container():
            st.markdown("""
            <div style="background:#161b27;border:1px solid rgba(255,255,255,0.08);
                        border-top:none;border-radius:0 0 16px 16px;padding:0 40px 36px;
                        margin-top:-2px">
            </div>
            """, unsafe_allow_html=True)

            # Overlay inputs on top of card bottom half
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.8rem;font-weight:500;color:#c8d0dc;margin-bottom:4px">Username</p>', unsafe_allow_html=True)
            username = st.text_input("u", placeholder="Enter your username",
                                     label_visibility="collapsed", key="li_u")
            st.markdown('<p style="font-size:0.8rem;font-weight:500;color:#c8d0dc;margin:12px 0 4px">Password</p>', unsafe_allow_html=True)
            password = st.text_input("p", type="password", placeholder="Enter your password",
                                     label_visibility="collapsed", key="li_p")
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            if st.button("Sign In", use_container_width=True, key="signin_btn"):
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    from src.auth import login
                    ok, msg = login(username, password)
                    if ok:
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown("""
            <div class="login-footer">
                Don't have an account?
                <a href="#">Contact Administrator</a>
            </div>
            <div style="font-size:0.68rem;color:#4a5568;text-align:center;margin-top:28px">
                &copy; 2026 Telco Analytics Platform. All rights reserved.
            </div>
            """, unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════════════════════════
user = current_user()

@st.cache_data(ttl=300)
def load_scored():
    if os.path.exists(SCORED_CSV):
        return pd.read_csv(SCORED_CSV), "Pipeline output"
    if os.path.exists(PROCESSED_DATA):
        return pd.read_csv(PROCESSED_DATA), "Processed CSV"
    try:
        from sqlalchemy import create_engine
        df = pd.read_sql(f"SELECT * FROM {TABLE_CLEANED}", create_engine(DB_URL))
        return df, "PostgreSQL"
    except Exception:
        pass
    return _demo(), "Demo data"

def _demo():
    np.random.seed(42); n = 1000
    contracts = np.random.choice(["Month-to-month","One year","Two year"], n, p=[0.55,0.25,0.20])
    internet  = np.random.choice(["Fiber optic","DSL","No"], n, p=[0.44,0.34,0.22])
    payment   = np.random.choice(["Electronic check","Mailed check",
                                   "Bank transfer (automatic)","Credit card (automatic)"],
                                  n, p=[0.34,0.23,0.22,0.21])
    tenure  = np.random.randint(1, 72, n)
    monthly = np.round(np.random.uniform(20, 110, n), 2)
    senior  = np.random.binomial(1, 0.16, n)
    churn   = np.where((contracts=="Month-to-month")&(monthly>65),
                        np.random.binomial(1,0.50,n), np.random.binomial(1,0.12,n))
    proba   = np.where(churn==1, np.random.uniform(0.55,0.99,n),
                                  np.random.uniform(0.05,0.45,n))
    risk    = np.where(proba>=0.65,"High", np.where(proba>=0.35,"Medium","Low"))
    rc      = np.where(monthly<35,"Low", np.where(monthly<70,"Medium","High"))
    tg      = pd.cut(tenure,[0,12,24,48,72],
                     labels=["0-1 Year","1-2 Years","2-4 Years","4-6 Years"]).astype(str)
    def _offer(i):
        if risk[i]=="High" and "Month-to-month" in contracts[i]:
            return ("CONTRACT_UPGRADE","Contract Upgrade Discount","20%",
                    "Switch to 1-year contract, 20% off for 6 months.","Sales team call")
        if risk[i]=="High" and "Fiber" in internet[i]:
            return ("TECHSUPPORT_BUNDLE","Free Tech Support Bundle","Free add-on",
                    "Free TechSupport for 3 months.","Auto-activate bundle")
        if "Electronic check" in payment[i]:
            return ("AUTOPAY_CASHBACK","Auto-Pay Cashback","Rs.200 cashback",
                    "Switch to auto-pay, get Rs.200 cashback.","Send SMS link")
        return ("STANDARD_LOYALTY","Loyalty Reward","Rs.100 credit",
                "Rs.100 bill credit this month.","Apply automatically")
    offers = list(map(_offer, range(n)))
    return pd.DataFrame({
        "customerID":     [f"C{i:04d}" for i in range(n)],
        "gender":         np.random.choice(["Male","Female"], n),
        "SeniorCitizen":  senior,
        "Partner":        np.random.choice(["Yes","No"], n),
        "Dependents":     np.random.choice(["Yes","No"], n),
        "tenure":         tenure, "MonthlyCharges": monthly,
        "TotalCharges":   np.round(monthly*tenure, 2),
        "AnnualRevenue":  np.round(monthly*12, 2),
        "LoyaltyScore":   np.round(tenure/72, 3),
        "Churn":          churn, "Contract": contracts,
        "InternetService": internet, "PaymentMethod": payment,
        "tenure_group":   tg, "RevenueCategory": rc,
        "ChurnProbability": np.round(proba, 4),
        "ChurnRisk":      risk, "WillChurn_Predicted": churn,
        "OfferID":        [o[0] for o in offers],
        "OfferTitle":     [o[1] for o in offers],
        "OfferDiscount":  [o[2] for o in offers],
        "OfferDescription":[o[3] for o in offers],
        "OfferAction":    [o[4] for o in offers],
    })

@st.cache_data(ttl=600)
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f: return json.load(f)
    return {}

df_all, source = load_scored()
metrics_data   = load_metrics()
HAS_ML = "ChurnProbability" in df_all.columns

def _reconstruct(df, col):
    if col in df.columns: return df
    cands = [c for c in df.columns if c.startswith(f"{col}_")]
    if cands:
        df = df.copy()
        df[col] = df[cands].idxmax(axis=1).str.replace(f"{col}_","",regex=False)
    return df

for _c in ["Contract","InternetService","PaymentMethod"]:
    df_all = _reconstruct(df_all, _c)


# ── Plotly dark layout helper ─────────────────────────────
def dfig(fig, h=320):
    fig.update_layout(
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892a4", family="Inter", size=12),
        margin=dict(t=10,b=10,l=10,r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(0,0,0,0)",
                   tickfont=dict(color="#8892a4")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(0,0,0,0)",
                   tickfont=dict(color="#8892a4")),
    )
    return fig

GRAD = ["#1a6ef5","#00d4ff","#00e676","#ffab00","#ff5252","#b388ff","#40c4ff"]


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

all_pages   = ["Dashboard","ML Predictions","Recommendations","Admin Panel"]
admin_only  = {"Admin Panel"}
visible     = [p for p in all_pages if p not in admin_only or is_admin()]

nav_icons = {
    "Dashboard":       "DB",
    "ML Predictions":  "ML",
    "Recommendations": "AI",
    "Admin Panel":     "AD",
}

initials = "".join([w[0].upper() for w in
                    (user["full_name"] or user["username"]).split()[:2]])

with st.sidebar:
    # Brand header
    st.markdown(f"""
    <div style="padding:20px 20px 16px;
                border-bottom:1px solid rgba(255,255,255,0.06);
                margin-bottom:10px">
        <div style="display:flex;align-items:center;gap:12px">
            <div style="width:38px;height:38px;border-radius:9px;
                        background:linear-gradient(135deg,#1a6ef5,#00d4ff);
                        display:flex;align-items:center;justify-content:center;
                        font-size:0.72rem;font-weight:800;color:#fff;
                        letter-spacing:-.01em">TA</div>
            <div>
                <div style="font-size:0.9rem;font-weight:700;color:#fff;
                            letter-spacing:-.01em">Telco ETL</div>
                <div style="font-size:0.7rem;color:#8892a4">Analytics</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Nav
    for p in visible:
        is_active = (st.session_state["page"] == p)
        if is_active:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a6ef5,#00d4ff);
                        border-radius:9px;padding:10px 14px;margin:2px 10px;
                        display:flex;align-items:center;gap:10px;
                        font-size:0.85rem;font-weight:600;color:#fff;
                        box-shadow:0 4px 12px rgba(26,110,245,0.3)">
                <span style="width:24px;height:24px;border-radius:6px;
                             background:rgba(255,255,255,0.2);
                             display:flex;align-items:center;justify-content:center;
                             font-size:0.6rem;font-weight:800">{nav_icons[p]}</span>
                {p}
            </div>""", unsafe_allow_html=True)
        else:
            # invisible label so button has no duplicate text
            if st.button(p, key=f"nav_{p}", use_container_width=True):
                st.session_state["page"] = p
                log_event(user["username"], user["role"], "PAGE_VIEW",
                          p.encode("ascii","ignore").decode())
                st.rerun()
            # override the button style to look like a nav item
            st.markdown(f"""
            <style>
            div[data-testid="stButton"] button[kind="secondary"][id*="nav_{p}"],
            div[data-testid="stButton"]:has(> button p:contains("{p}")) button {{
                background:transparent !important;
                box-shadow:none !important;
                color:#8892a4 !important;
                text-align:left !important;
                padding:10px 14px !important;
                border-radius:9px !important;
                margin:2px 0 !important;
                justify-content:flex-start !important;
            }}
            div[data-testid="stButton"]:has(> button p:contains("{p}")) button:hover {{
                background:rgba(255,255,255,0.05) !important;
                color:#e2e8f0 !important;
            }}
            </style>""", unsafe_allow_html=True)

    # Filters section
    st.markdown("""
    <div style="margin:16px 10px 8px;padding-top:14px;
                border-top:1px solid rgba(255,255,255,0.06)">
        <div style="font-size:0.68rem;font-weight:700;color:#4a5568;
                    text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px">
            Filters
        </div>
    </div>""", unsafe_allow_html=True)

    c_opts = ["All"] + sorted(df_all["Contract"].dropna().unique().tolist())
    sel_c  = st.selectbox("Contract", c_opts)
    i_opts = ["All"] + sorted(df_all["InternetService"].dropna().unique().tolist())
    sel_i  = st.selectbox("Internet Service", i_opts)
    t_rng  = st.slider("Tenure (months)",
                        int(df_all["tenure"].min()), int(df_all["tenure"].max()),
                        (int(df_all["tenure"].min()), int(df_all["tenure"].max())))

    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    # User strip pinned to bottom
    st.markdown(f"""
    <div style="position:fixed;bottom:0;left:0;width:250px;
                background:#161b27;
                border-top:1px solid rgba(255,255,255,0.06);
                padding:14px 18px">
        <div style="display:flex;align-items:center;gap:10px">
            <div class="avatar">{initials}</div>
            <div style="flex:1;min-width:0">
                <div style="font-size:0.82rem;font-weight:600;color:#fff;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                    {user['full_name'] or user['username']}</div>
                <div style="font-size:0.7rem;color:#8892a4">{user['role']}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:72px'></div>", unsafe_allow_html=True)
    if st.button("Logout", use_container_width=True):
        logout()


# Apply filters
df = df_all.copy()
if sel_c != "All": df = df[df["Contract"] == sel_c]
if sel_i != "All": df = df[df["InternetService"] == sel_i]
df = df[(df["tenure"] >= t_rng[0]) & (df["tenure"] <= t_rng[1])]
page = st.session_state["page"]


# ════════════════════════════════════════════════════════════
# PAGE 1  —  Dashboard
# ════════════════════════════════════════════════════════════
if page == "Dashboard":
    total   = len(df); churned = int(df["Churn"].sum())
    churn_r = churned/total*100 if total else 0
    rev_mo  = df["MonthlyCharges"].sum()
    high_r  = int((df["ChurnRisk"]=="High").sum()) if HAS_ML else 0

    st.markdown("""<div class="page-header">
        <div class="page-title">Churn Analytics Dashboard</div>
        <div class="page-subtitle">Real-time insights into customer retention and churn patterns</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='content-wrap'>", unsafe_allow_html=True)

    k1,k2,k3,k4 = st.columns(4)
    for col, label, val, delta, delta_class, icon, icon_class in [
        (k1,"Total Customers",f"{total:,}","Active accounts","kpi-delta-pos","USR","icon-blue"),
        (k2,"Churn Rate",f"{churn_r:.1f}%",f"{churned:,} customers lost","kpi-delta-neg","CHN","icon-red"),
        (k3,"Monthly Revenue",f"Rs.{rev_mo/1000:.1f}K","Total active base","kpi-delta-pos","REV","icon-green"),
        (k4,"At-Risk Customers",f"{high_r:,}","High Priority","kpi-delta-neg","ALT","icon-amber"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card">
              <div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="{delta_class}">{delta}</div>
              </div>
              <div class="kpi-icon {icon_class}">{icon}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    c1,c2 = st.columns([3,2])
    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title">Churn Rate by Contract Type</div>', unsafe_allow_html=True)
        cdf = (df.groupby("Contract")["Churn"].agg(["sum","count"])
                 .rename(columns={"sum":"Churned","count":"Total"})
                 .assign(**{"Churn %":lambda x:(x.Churned/x.Total*100).round(1)})
                 .reset_index())
        fig = px.bar(cdf, x="Contract", y="Churn %", text="Churn %",
                     color="Contract",
                     color_discrete_sequence=["#1a6ef5","#00d4ff","#00e676"])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          textfont_color="#fff", marker_line_width=0)
        st.plotly_chart(dfig(fig), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title">Churn by Service Type</div>', unsafe_allow_html=True)
        idf = (df.groupby("InternetService")["Churn"].agg(["sum","count"])
                 .rename(columns={"sum":"Churned","count":"Total"})
                 .assign(**{"Churn %":lambda x:(x.Churned/x.Total*100).round(1)})
                 .reset_index())
        fig = px.pie(idf, values="Churn %", names="InternetService",
                     color_discrete_sequence=["#1a6ef5","#00d4ff","#00e676","#b388ff"],
                     hole=0.35)
        fig.update_traces(textfont_color="#fff", textfont_size=12)
        st.plotly_chart(dfig(fig, 300), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<div class="chart-card"><div class="chart-title">Monthly Charges — Churned vs Retained</div>', unsafe_allow_html=True)
        fig = px.box(df, x=df["Churn"].map({0:"Retained",1:"Churned"}),
                     y="MonthlyCharges",
                     color=df["Churn"].map({0:"Retained",1:"Churned"}),
                     color_discrete_map={"Retained":"#1a6ef5","Churned":"#ff5252"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(dfig(fig), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-card"><div class="chart-title">Revenue at Risk by Segment</div>', unsafe_allow_html=True)
        rdf = (df[df["Churn"]==1].groupby("RevenueCategory")["MonthlyCharges"]
                 .sum().reset_index()
                 .rename(columns={"MonthlyCharges":"Lost"}))
        fig = px.bar(rdf, x="RevenueCategory", y="Lost", text="Lost",
                     color="RevenueCategory",
                     color_discrete_map={"Low":"#00e676","Medium":"#ffab00","High":"#ff5252"})
        fig.update_traces(texttemplate="Rs.%{text:,.0f}", textposition="outside",
                          textfont_color="#fff", marker_line_width=0)
        fig.update_layout(showlegend=False)
        st.plotly_chart(dfig(fig), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Payment method
    st.markdown('<div class="chart-card"><div class="chart-title">Churn Rate by Payment Method</div>', unsafe_allow_html=True)
    pdf = (df.groupby("PaymentMethod")["Churn"].agg(["sum","count"])
             .rename(columns={"sum":"Churned","count":"Total"})
             .assign(**{"Churn %":lambda x:(x.Churned/x.Total*100).round(1)})
             .reset_index().sort_values("Churn %", ascending=True))
    fig = px.bar(pdf, x="Churn %", y="PaymentMethod", orientation="h", text="Churn %",
                 color="Churn %", color_continuous_scale=["#1a6ef5","#ffab00","#ff5252"])
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                      textfont_color="#fff", marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(dfig(fig, 250), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 2  —  ML Predictions
# ════════════════════════════════════════════════════════════
elif page == "ML Predictions":
    st.markdown("""<div class="page-header">
        <div class="page-title">ML Churn Prediction</div>
        <div class="page-subtitle">Advanced machine learning models for customer churn prediction</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='content-wrap'>", unsafe_allow_html=True)

    mres = metrics_data.get("model_results", {})
    best = metrics_data.get("best_model", "")

    if mres:
        bm  = mres.get(best, list(mres.values())[0])
        acc = bm.get("accuracy",0)*100
        pre = bm.get("precision",0)*100
        rec = bm.get("recall",0)*100
        f1  = bm.get("f1",0)*100

        st.markdown('<div class="section-title">Current Model Performance</div>', unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        for col, label, val, bar in [
            (m1,"Accuracy",acc,"progress-green"),
            (m2,"Precision",pre,"progress-blue"),
            (m3,"Recall",rec,"progress-gray"),
            (m4,"F1 Score",f1,"progress-gray"),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="flex-direction:column;align-items:flex-start;gap:0">
                    <div style="display:flex;justify-content:space-between;width:100%">
                        <div class="kpi-label">{label}</div>
                        <span style="color:#b388ff;font-size:0.75rem;font-weight:600">LIVE</span>
                    </div>
                    <div class="kpi-value">{val:.1f}%</div>
                    <div class="progress-wrap">
                        <div class="{bar}" style="width:{min(val,100):.0f}%"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Run `python main.py --no-load` to train the model and see metrics here.")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col_form, col_result = st.columns(2)
    with col_form:
        st.markdown('<div class="chart-card"><div class="chart-title">Single Customer Prediction</div>', unsafe_allow_html=True)
        model_ready = os.path.exists(os.path.join(MODELS_DIR,"churn_model.pkl"))
        if not model_ready:
            st.warning("Run `python main.py --no-load` first to train the model.")
        else:
            tenure_p   = st.slider("Tenure (months)", 1, 72, 12, key="pt")
            contract_p = st.selectbox("Contract Type",
                                       ["Month-to-month","One year","Two year"], key="pc")
            payment_p  = st.selectbox("Payment Method",
                                       ["Electronic check","Mailed check",
                                        "Bank transfer (automatic)",
                                        "Credit card (automatic)"], key="pp")
            internet_p = st.selectbox("Internet Service",
                                       ["Fiber optic","DSL","No"], key="pi")
            monthly_p  = st.number_input("Monthly Charges (Rs.)", 18.0, 120.0, 65.0, key="pm")
            senior_p   = st.selectbox("Senior Citizen", ["No","Yes"], key="ps")

            if st.button("Run Prediction", use_container_width=True):
                customer = {
                    "SeniorCitizen": 1 if senior_p=="Yes" else 0,
                    "tenure": tenure_p, "MonthlyCharges": monthly_p,
                    "TotalCharges": monthly_p * tenure_p,
                    "gender_encoded": 1, "Partner_encoded": 0,
                    "Dependents_encoded": 0, "PhoneService_encoded": 1,
                    "PaperlessBilling_encoded": 1,
                    "AnnualRevenue": monthly_p*12,
                    "LoyaltyScore": round(tenure_p/72, 3),
                }
                customer_raw = {
                    "Contract": contract_p, "InternetService": internet_p,
                    "PaymentMethod": payment_p,
                    "SeniorCitizen": 1 if senior_p=="Yes" else 0,
                    "tenure": tenure_p, "MonthlyCharges": monthly_p,
                }
                from src.predict import predict_single
                result = predict_single(customer)
                st.session_state["last_pred"]     = result
                st.session_state["last_cust_raw"] = customer_raw
                log_event(user["username"], user["role"], "PREDICTION_RUN",
                          f"prob={result['probability']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="chart-card"><div class="chart-title">Customer Risk Profile</div>', unsafe_allow_html=True)
        if "last_pred" in st.session_state:
            res   = st.session_state["last_pred"]
            prob  = res["probability"]
            risk  = res["risk"]
            color = {"High":"#ff5252","Medium":"#ffab00","Low":"#00e676"}[risk]
            badge = {"High":"badge-high","Medium":"badge-medium","Low":"badge-low"}[risk]

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                number={"suffix":"%","font":{"size":34,"color":"#fff"}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#4a5568",
                               "tickfont":{"color":"#8892a4"}},
                       "bar":{"color":color,"thickness":0.28},
                       "bgcolor":"rgba(0,0,0,0)",
                       "steps":[{"range":[0,35],"color":"rgba(0,230,118,0.08)"},
                                 {"range":[35,65],"color":"rgba(255,171,0,0.08)"},
                                 {"range":[65,100],"color":"rgba(255,82,82,0.08)"}],
                       "bordercolor":"rgba(0,0,0,0)"}))
            fig.update_layout(height=210, paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#8892a4"),
                              margin=dict(t=20,b=0,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div style="text-align:center;margin-top:4px">
                <span class="{badge}">
                    {"HIGH RISK" if risk=="High" else
                     "MEDIUM RISK" if risk=="Medium" else "LOW RISK"}
                </span>
            </div>""", unsafe_allow_html=True)

            from src.recommend import recommend_for_single
            offer = recommend_for_single(st.session_state["last_cust_raw"], res)
            st.markdown(f"""
            <div style="margin-top:14px;background:rgba(26,110,245,0.07);
                        border:1px solid rgba(26,110,245,0.18);
                        border-radius:10px;padding:14px 16px">
                <div style="font-size:0.75rem;font-weight:600;color:#60a5fa;
                            margin-bottom:6px;text-transform:uppercase;
                            letter-spacing:.05em">Recommended Offer</div>
                <div style="font-size:0.88rem;font-weight:600;color:#fff">
                    {offer['title']}</div>
                <div style="font-size:0.78rem;color:#8892a4;margin-top:4px">
                    {offer['description']}</div>
                <div style="font-size:0.78rem;color:#00e676;margin-top:8px;font-weight:600">
                    Discount: {offer['discount']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;height:260px;color:#4a5568">
                <div style="width:48px;height:48px;border-radius:12px;
                            background:rgba(255,255,255,0.04);
                            display:flex;align-items:center;justify-content:center;
                            font-size:1.2rem;font-weight:700;color:#4a5568;
                            margin-bottom:12px">?</div>
                <div style="font-size:0.84rem">
                    Run a prediction to see customer profile</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance + confusion matrix
    fimp = metrics_data.get("feature_importance", {})
    if fimp and mres:
        st.markdown('<div class="section-title">Model Insights</div>', unsafe_allow_html=True)
        fi1, fi2 = st.columns(2)
        with fi1:
            st.markdown('<div class="chart-card"><div class="chart-title">Feature Importance</div>', unsafe_allow_html=True)
            fidf = pd.DataFrame(list(fimp.items()),
                                columns=["Feature","Importance"]).head(8)
            fig = px.bar(fidf, x="Importance", y="Feature", orientation="h",
                         color="Importance",
                         color_continuous_scale=["#1a2035","#1a6ef5","#00d4ff"])
            fig.update_layout(coloraxis_showscale=False, yaxis_title="")
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(dfig(fig, 280), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with fi2:
            cm = mres.get(best, {}).get("confusion_matrix")
            if cm:
                st.markdown('<div class="chart-card"><div class="chart-title">Confusion Matrix</div>', unsafe_allow_html=True)
                cmdf = pd.DataFrame(cm,
                                    index=["Actual: No Churn","Actual: Churn"],
                                    columns=["Pred: No Churn","Pred: Churn"])
                fig = px.imshow(cmdf, text_auto=True, aspect="auto",
                                color_continuous_scale=["#0d1117","#1a6ef5"])
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(dfig(fig, 280), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 3  —  Recommendations
# ════════════════════════════════════════════════════════════
elif page == "Recommendations":
    st.markdown("""<div class="page-header">
        <div class="page-title">Retention Recommendations</div>
        <div class="page-subtitle">AI-powered strategies to reduce churn and improve customer loyalty</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='content-wrap'>", unsafe_allow_html=True)

    campaigns = [
        {"label":"GFT","label_bg":"rgba(26,110,245,0.18)","label_color":"#60a5fa",
         "title":"Contract Upgrade Discount","sub":"Month-to-month customers",
         "impact":"+20%","cost":"Custom","status":"Active",
         "desc":"Switch to 1-year contract with 20% off for 6 months."},
        {"label":"REV","label_bg":"rgba(0,230,118,0.15)","label_color":"#00e676",
         "title":"Auto-Pay Cashback Reward","sub":"Electronic check users",
         "impact":"+15%","cost":"Rs.200/user","status":"Proposed",
         "desc":"Switch to automatic payment and get Rs.200 cashback on next bill."},
        {"label":"SUP","label_bg":"rgba(180,136,255,0.18)","label_color":"#b388ff",
         "title":"Free Tech Support Bundle","sub":"Fiber optic high-risk",
         "impact":"+18%","cost":"Free add-on","status":"Active",
         "desc":"Complimentary TechSupport and OnlineSecurity for 3 months."},
        {"label":"VIP","label_bg":"rgba(255,171,0,0.15)","label_color":"#ffab00",
         "title":"Loyalty Upgrade Program","sub":"Premium long-term customers",
         "impact":"+25%","cost":"Free upgrade","status":"Proposed",
         "desc":"Exclusive senior plan with 25% reduced monthly charge."},
    ]

    st.markdown('<div class="section-title">Recommended Campaigns</div>', unsafe_allow_html=True)
    cc = st.columns(4)
    for i, c in enumerate(campaigns):
        badge_cls = "badge-active" if c["status"]=="Active" else "badge-proposed"
        with cc[i]:
            st.markdown(f"""
            <div class="offer-card">
                <div class="offer-icon-box"
                     style="background:{c['label_bg']};color:{c['label_color']}">
                    {c['label']}
                </div>
                <div class="offer-title">{c['title']}</div>
                <div class="offer-subtitle">{c['sub']}</div>
                <div style="font-size:0.72rem;color:#8892a4;margin-bottom:14px;
                            line-height:1.5">{c['desc']}</div>
                <div class="offer-row">
                    <span class="offer-key">Retention Impact</span>
                    <span class="offer-val-pos">{c['impact']}</span>
                </div>
                <div class="offer-row">
                    <span class="offer-key">Est. Cost</span>
                    <span class="offer-val">{c['cost']}</span>
                </div>
                <div style="margin-top:14px;display:flex;
                            justify-content:space-between;align-items:center">
                    <span class="{badge_cls}">{c['status']}</span>
                    <span style="color:#1a6ef5;font-size:0.78rem;
                                 font-weight:600;cursor:pointer">View Details</span>
                </div>
            </div>""", unsafe_allow_html=True)

    if HAS_ML:
        st.markdown('<div class="section-title">High-Risk Customer Actions</div>', unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.8rem;color:#8892a4;margin-bottom:14px">
            Personalised recommendations for customers most likely to churn
        </div>""", unsafe_allow_html=True)

        at_risk = (df[df["ChurnRisk"]=="High"]
                   .sort_values("ChurnProbability", ascending=False).head(8))
        for _, row in at_risk.iterrows():
            cid   = str(row.get("customerID",""))
            ini   = cid[:2].upper() if cid else "??"
            prob  = row.get("ChurnProbability", 0)
            offer = row.get("OfferTitle", "Retention Call")
            con   = row.get("Contract", "—")
            mo    = row.get("MonthlyCharges", 0)

            st.markdown(f"""
            <div style="background:#1a2035;border:1px solid rgba(255,255,255,0.06);
                        border-radius:12px;padding:14px 20px;margin-bottom:8px;
                        display:flex;align-items:center;gap:16px;
                        transition:border-color .2s">
                <div class="avatar">{ini}</div>
                <div style="flex:1">
                    <div style="font-size:0.87rem;font-weight:600;color:#fff">{cid}</div>
                    <div style="font-size:0.73rem;color:#8892a4">{con}</div>
                </div>
                <div style="text-align:right;min-width:80px">
                    <div style="font-size:0.7rem;color:#8892a4">Monthly</div>
                    <div style="font-size:0.87rem;color:#fff;font-weight:600">
                        Rs.{mo:.0f}</div>
                </div>
                <div style="min-width:100px;text-align:center">
                    <span class="badge-high">Risk: {prob:.0%}</span>
                </div>
                <div style="text-align:right;min-width:170px">
                    <div style="font-size:0.7rem;color:#8892a4">Recommended</div>
                    <div style="font-size:0.78rem;color:#60a5fa;font-weight:500">
                        {offer}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        scols = [c for c in
                 ["customerID","ChurnProbability","ChurnRisk","Contract",
                  "MonthlyCharges","tenure","OfferTitle","OfferDiscount"]
                 if c in df.columns]
        csv_d = df[df["ChurnRisk"]=="High"][scols].to_csv(index=False).encode()
        st.download_button("Export High-Risk List", csv_d,
                           "high_risk_customers.csv","text/csv",
                           use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 4  —  Admin Panel
# ════════════════════════════════════════════════════════════
elif page == "Admin Panel":
    if not is_admin():
        st.error("Access Denied — Admin only.")
        st.stop()

    st.markdown("""<div class="page-header">
        <div class="page-title">Admin Panel</div>
        <div class="page-subtitle">Manage users, ETL processes, and system configuration</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='content-wrap'>", unsafe_allow_html=True)

    users_list = get_all_users()
    log_data   = get_access_log()
    log_df     = pd.DataFrame(log_data) if log_data else pd.DataFrame()

    s1,s2,s3,s4 = st.columns(4)
    for col, label, val, delta, dc, ic in [
        (s1,"Total Users",len(users_list),"Active accounts","kpi-delta-pos","icon-blue"),
        (s2,"Successful Logins",
         len(log_df[log_df["event"]=="LOGIN_SUCCESS"]) if not log_df.empty else 0,
         "Total sessions","kpi-delta-pos","icon-green"),
        (s3,"Failed Logins",
         len(log_df[log_df["event"]=="LOGIN_FAILED"]) if not log_df.empty else 0,
         "Security alerts","kpi-delta-neg","icon-red"),
        (s4,"Page Views",
         len(log_df[log_df["event"]=="PAGE_VIEW"]) if not log_df.empty else 0,
         "Total navigation","kpi-delta-pos","icon-amber"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card">
              <div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="{dc}">{delta}</div>
              </div>
              <div class="kpi-icon {ic}">AD</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(["  User Management  ","  Change Password  ","  Access Log  "])

    with tab1:
        st.markdown("""
        <div style="font-size:0.9rem;font-weight:600;color:#fff;margin-bottom:14px">
            User Accounts</div>
        <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.06);
                    border-radius:10px;overflow:hidden;margin-bottom:24px">
            <div class="user-row" style="border-bottom:1px solid rgba(255,255,255,0.08)">
                <span style="font-size:0.68rem;font-weight:700;color:#4a5568;
                             text-transform:uppercase;letter-spacing:.08em">User</span>
                <span style="font-size:0.68rem;font-weight:700;color:#4a5568;
                             text-transform:uppercase;letter-spacing:.08em">Role</span>
                <span style="font-size:0.68rem;font-weight:700;color:#4a5568;
                             text-transform:uppercase;letter-spacing:.08em">Created</span>
            </div>""", unsafe_allow_html=True)

        for u in users_list:
            ini = "".join([w[0].upper() for w in u["username"].split()[:2]])
            rb  = "badge-admin" if u["role"]=="admin" else "badge-user"
            st.markdown(f"""
            <div class="user-row">
                <div style="display:flex;align-items:center;gap:10px">
                    <div class="avatar">{ini}</div>
                    <div>
                        <div style="font-size:0.84rem;font-weight:600;color:#fff">
                            {u['username']}</div>
                        <div style="font-size:0.7rem;color:#8892a4">
                            {u.get('full_name','')}</div>
                    </div>
                </div>
                <span class="{rb}">{u['role'].capitalize()}</span>
                <span style="font-size:0.76rem;color:#8892a4">
                    {u.get('created_at','—')}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        ca, cd = st.columns(2)
        with ca:
            st.markdown('<div class="chart-card"><div class="chart-title">Add New User</div>', unsafe_allow_html=True)
            nu = st.text_input("Username",  key="nu")
            nf = st.text_input("Full Name", key="nf")
            np_ = st.text_input("Password", type="password", key="np")
            nr = st.selectbox("Role", ["user","admin"], key="nr")
            if st.button("Add User", use_container_width=True, key="add_btn"):
                ok, msg = add_user(nu, np_, nr, nf, user["username"])
                st.success(msg) if ok else st.error(msg)
                if ok: st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with cd:
            st.markdown('<div class="chart-card"><div class="chart-title">Delete User</div>', unsafe_allow_html=True)
            del_opts = [u["username"] for u in users_list
                        if u["username"] != user["username"]]
            if del_opts:
                del_t = st.selectbox("Select user", del_opts)
                st.warning(f"Permanently deletes **{del_t}**.")
                if st.button("Delete User", use_container_width=True, key="del_btn"):
                    ok, msg = delete_user(del_t, user["username"])
                    st.success(msg) if ok else st.error(msg)
                    if ok: st.rerun()
            else:
                st.info("No other users to delete.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        col_pw, _ = st.columns([1,1])
        with col_pw:
            st.markdown('<div class="chart-card"><div class="chart-title">Change Password</div>', unsafe_allow_html=True)
            all_u   = [u["username"] for u in get_all_users()]
            pw_tgt  = st.selectbox("Select User", all_u)
            pw1     = st.text_input("New Password",     type="password", key="pw1")
            pw2     = st.text_input("Confirm Password", type="password", key="pw2")
            if st.button("Update Password", use_container_width=True):
                if pw1 != pw2: st.error("Passwords do not match.")
                else:
                    ok, msg = change_password(pw_tgt, pw1, user["username"])
                    st.success(msg) if ok else st.error(msg)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        if not log_df.empty:
            fc1,fc2,fc3 = st.columns(3)
            with fc1:
                ev_opts = ["All"]+sorted(log_df["event"].unique().tolist())
                sel_ev  = st.selectbox("Event", ev_opts)
            with fc2:
                un_opts = ["All"]+sorted(log_df["username"].unique().tolist())
                sel_un  = st.selectbox("User", un_opts)
            with fc3:
                n_rows = st.number_input("Show last N", 10, 1000, 50, step=10)

            fl = log_df.copy()
            if sel_ev != "All": fl = fl[fl["event"]==sel_ev]
            if sel_un != "All": fl = fl[fl["username"]==sel_un]
            fl = fl.tail(int(n_rows)).sort_values("timestamp", ascending=False)
            st.dataframe(fl, use_container_width=True, height=360, hide_index=True)
            st.download_button("Download Log", log_df.to_csv(index=False).encode(),
                               "access_log.csv","text/csv", use_container_width=True)
        else:
            st.info("No log entries yet.")

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-bar">
    <div class="footer-title">
        Design and Implementation of Scalable ETL Pipeline for Telco Customer Churn Analysis
        using PySpark, PostgreSQL and Streamlit
    </div>
    <div class="footer-sub">
        Divyesh Joshi &nbsp;&middot;&nbsp; MC24097 &nbsp;&middot;&nbsp; MCA-II Sem IV
        &nbsp;&middot;&nbsp; IIMS Chinchwad, Pune
        &nbsp;&middot;&nbsp; Savitribai Phule Pune University &nbsp;&middot;&nbsp; 2025-26
    </div>
</div>
""", unsafe_allow_html=True)
