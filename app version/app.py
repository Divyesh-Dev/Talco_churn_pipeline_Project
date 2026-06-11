# ============================================================
# app.py — Telco Churn Analytics  |  Figma Design Theme
# Dark navy theme with blue-cyan gradient accents
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
from src.auth   import (is_logged_in, is_admin, current_user,
                         logout, render_login_page,
                         get_all_users, add_user, delete_user,
                         change_password, get_access_log, log_event)

MODELS_DIR   = os.path.join(_ROOT, "models")
SCORED_CSV   = os.path.join(_ROOT, "data", "processed", "churn_scored.csv")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
# GLOBAL CSS  — Figma dark theme
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Reset & Base ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
.main, .block-container {
    background: #0d1117 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* hide streamlit chrome */
header[data-testid="stHeader"]  { display:none !important; }
#MainMenu, footer                { display:none !important; }
.block-container                 { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    width: 252px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebarContent"]            { padding: 0 !important; }

/* hide default sidebar header collapse button */
[data-testid="collapsedControl"] { display:none !important; }

/* sidebar radio — hidden (we use custom HTML nav) */
[data-testid="stSidebar"] .stRadio             { display:none !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { display:none !important; }

/* ── Inputs ───────────────────────────────────────────── */
input, textarea, select,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #1e2d45 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
input::placeholder { color: #4a5568 !important; }
input:focus { border-color: #1a6ef5 !important; box-shadow: 0 0 0 2px rgba(26,110,245,0.25) !important; }

/* selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #1e2d45 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
[data-baseweb="select"] { background: #1e2d45 !important; }
[data-baseweb="popover"] li { background: #1a2035 !important; color: #e2e8f0 !important; }

/* sliders */
[data-testid="stSlider"] > div { color: #8892a4 !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #1a6ef5 !important; }

/* multiselect */
[data-testid="stMultiSelect"] > div > div {
    background: #1e2d45 !important;
    border-color: rgba(255,255,255,0.1) !important;
}
span[data-baseweb="tag"] { background: rgba(26,110,245,0.25) !important; color: #60a5fa !important; }

/* number input */
[data-testid="stNumberInput"] button { background: #1e2d45 !important; border-color: rgba(255,255,255,0.1) !important; color: #e2e8f0 !important; }

/* ── Buttons ──────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1a6ef5, #00d4ff) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 0.875rem !important; padding: 10px 20px !important;
    transition: opacity .2s, transform .1s !important;
}
.stButton > button:hover  { opacity: .9 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* secondary button style — logout/refresh */
.stButton.secondary > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #8892a4 !important;
}

/* ── Metrics ──────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #1a2035 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
[data-testid="stMetricLabel"] p  { color: #8892a4 !important; font-size: 0.82rem !important; }
[data-testid="stMetricValue"]    { color: #fff !important; font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"]    { font-size: 0.8rem !important; }

/* ── Dataframe ────────────────────────────────────────── */
[data-testid="stDataFrame"] { background: #1a2035 !important; border-radius: 12px !important; }
.dvn-scroller { background: #1a2035 !important; }

/* ── Tabs ─────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #1a2035 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border-bottom: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #8892a4 !important;
    border-radius: 7px !important;
    font-size: 0.85rem !important;
    padding: 8px 16px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg,#1a6ef5,#00d4ff) !important;
    color: #fff !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none !important; }
[data-testid="stTabPanel"] {
    background: #1a2035 !important;
    border-radius: 0 12px 12px 12px !important;
    padding: 20px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Divider ──────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 20px 0 !important; }

/* ── Alerts ───────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Custom components ─────────────────────────────────── */
.page-header { padding: 28px 32px 0; }
.page-title  { font-size: 1.6rem; font-weight: 700; color: #fff; margin-bottom: 4px; }
.page-sub    { font-size: 0.85rem; color: #8892a4; margin-bottom: 24px; }

.section-title {
    font-size: 1rem; font-weight: 600; color: #fff;
    margin: 24px 0 14px;
}

.kpi-card {
    background: #1a2035;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 20px;
    display: flex; justify-content: space-between; align-items: flex-start;
}
.kpi-label  { font-size: 0.8rem; color: #8892a4; margin-bottom: 8px; }
.kpi-value  { font-size: 1.7rem; font-weight: 700; color: #fff; }
.kpi-delta-pos { color: #00e676; font-size: 0.78rem; margin-top: 4px; }
.kpi-delta-neg { color: #ff5252; font-size: 0.78rem; margin-top: 4px; }
.kpi-icon {
    width: 44px; height: 44px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; flex-shrink: 0;
}
.icon-blue   { background: rgba(26,110,245,0.2); }
.icon-red    { background: rgba(255,82,82,0.2); }
.icon-green  { background: rgba(0,230,118,0.2); }
.icon-amber  { background: rgba(255,171,0,0.2); }

.chart-card {
    background: #1a2035;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.chart-title { font-size: 0.95rem; font-weight: 600; color: #fff; margin-bottom: 16px; }

.offer-card {
    background: #1a2035;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px;
    height: 100%;
}
.offer-icon  { font-size: 1.6rem; margin-bottom: 12px; }
.offer-title { font-size: 0.95rem; font-weight: 600; color: #fff; margin-bottom: 4px; }
.offer-sub   { font-size: 0.78rem; color: #8892a4; margin-bottom: 14px; }
.offer-row   { display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:6px; }
.offer-key   { color: #8892a4; }
.offer-val-green { color: #00e676; font-weight: 600; }
.offer-val   { color: #e2e8f0; }
.badge-active   { background:rgba(0,230,118,0.15); color:#00e676; font-size:0.7rem; padding:2px 8px; border-radius:4px; font-weight:600; }
.badge-proposed { background:rgba(26,110,245,0.15); color:#60a5fa; font-size:0.7rem; padding:2px 8px; border-radius:4px; font-weight:600; }
.badge-admin-role  { background:rgba(26,110,245,0.2); color:#60a5fa; font-size:0.72rem; padding:2px 8px; border-radius:4px; }
.badge-user-role   { background:rgba(0,230,118,0.15); color:#00e676; font-size:0.72rem; padding:2px 8px; border-radius:4px; }
.badge-high  { background:rgba(255,82,82,0.15); color:#ff5252; font-size:0.78rem; padding:3px 10px; border-radius:6px; font-weight:600; }
.badge-med   { background:rgba(255,171,0,0.15); color:#ffab00; font-size:0.78rem; padding:3px 10px; border-radius:6px; font-weight:600; }
.badge-low   { background:rgba(0,230,118,0.15); color:#00e676; font-size:0.78rem; padding:3px 10px; border-radius:6px; font-weight:600; }

.progress-bar-wrap { background: rgba(255,255,255,0.08); border-radius:4px; height:6px; margin-top:10px; width:100%; }
.progress-bar-fill-green { height:6px; border-radius:4px; background:linear-gradient(90deg,#00e676,#00b894); }
.progress-bar-fill-blue  { height:6px; border-radius:4px; background:linear-gradient(90deg,#1a6ef5,#00d4ff); }
.progress-bar-fill-gray  { height:6px; border-radius:4px; background:rgba(255,255,255,0.2); }

.avatar-circle {
    width:36px; height:36px; border-radius:50%;
    background:linear-gradient(135deg,#1a6ef5,#00d4ff);
    display:inline-flex; align-items:center; justify-content:center;
    color:#fff; font-size:0.75rem; font-weight:700; flex-shrink:0;
}

.content-wrap { padding: 0 32px 32px; }

.footer-bar {
    margin-top: 40px;
    border-top: 1px solid rgba(255,255,255,0.06);
    padding: 18px 32px;
    text-align: center;
}
.footer-title { font-size:0.75rem; font-weight:600; color:#8892a4; margin-bottom:4px; }
.footer-sub   { font-size:0.7rem; color:#4a5568; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# AUTH GATE
# ════════════════════════════════════════════════════════════
if not is_logged_in():
    # ── Custom Login Page ────────────────────────────────────
    st.markdown("""
    <style>
    .login-bg {
        min-height:100vh; background: radial-gradient(ellipse at 30% 50%, #0d2060 0%, #0d1117 60%);
        display:flex; align-items:center; justify-content:center;
        flex-direction:column; padding:40px 20px;
    }
    .login-card {
        background:#161b27; border:1px solid rgba(255,255,255,0.08);
        border-radius:16px; padding:40px 44px; width:100%; max-width:440px;
    }
    .login-logo {
        width:64px; height:64px; border-radius:14px;
        background:linear-gradient(135deg,#1a6ef5,#00d4ff);
        display:flex; align-items:center; justify-content:center;
        font-size:2rem; margin:0 auto 20px;
    }
    .login-title { font-size:1.5rem; font-weight:700; color:#fff; text-align:center; margin-bottom:6px; }
    .login-sub   { font-size:0.85rem; color:#8892a4; text-align:center; margin-bottom:32px; }
    .login-label { font-size:0.82rem; font-weight:500; color:#e2e8f0; margin-bottom:8px; display:block; }
    .login-footer-text { font-size:0.78rem; color:#4a5568; text-align:center; margin-top:24px; }
    .login-footer-text a { color:#1a6ef5; text-decoration:none; }
    </style>
    <div class="login-bg">
      <div class="login-card">
        <div class="login-logo">🗄️</div>
        <div class="login-title">Telco Churn Analytics</div>
        <div class="login-sub">ETL Dashboard &amp; Prediction Platform</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 0.8, 1])
    with col:
        with st.container():
            st.markdown('<p style="font-size:0.82rem;font-weight:500;color:#e2e8f0;margin-bottom:4px">Username</p>', unsafe_allow_html=True)
            username = st.text_input("", placeholder="Enter username", key="login_user", label_visibility="collapsed")
            st.markdown('<p style="font-size:0.82rem;font-weight:500;color:#e2e8f0;margin:12px 0 4px">Password</p>', unsafe_allow_html=True)
            password = st.text_input("", type="password", placeholder="••••••••", key="login_pass", label_visibility="collapsed")
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            if st.button("Sign In", use_container_width=True):
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    from src.auth import login
                    ok, msg = login(username, password)
                    if ok: st.rerun()
                    else:  st.error(msg)
            st.markdown("""
            <div class="login-footer-text">
                Don't have an account? <a href="#">Contact Administrator</a>
            </div>
            <div style="font-size:0.72rem;color:#4a5568;text-align:center;margin-top:32px">
                © 2026 Telco Analytics Platform. All rights reserved.
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
    np.random.seed(42); n=1000
    contracts=np.random.choice(["Month-to-month","One year","Two year"],n,p=[0.55,0.25,0.20])
    internet=np.random.choice(["Fiber optic","DSL","No"],n,p=[0.44,0.34,0.22])
    payment=np.random.choice(["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"],n,p=[0.34,0.23,0.22,0.21])
    tenure=np.random.randint(1,72,n); monthly=np.round(np.random.uniform(20,110,n),2)
    senior=np.random.binomial(1,0.16,n)
    churn=np.where((contracts=="Month-to-month")&(monthly>65),np.random.binomial(1,0.50,n),np.random.binomial(1,0.12,n))
    proba=np.where(churn==1,np.random.uniform(0.55,0.99,n),np.random.uniform(0.05,0.45,n))
    risk=np.where(proba>=0.65,"High",np.where(proba>=0.35,"Medium","Low"))
    rc=np.where(monthly<35,"Low",np.where(monthly<70,"Medium","High"))
    tg=pd.cut(tenure,[0,12,24,48,72],labels=["0-1 Year","1-2 Years","2-4 Years","4-6 Years"]).astype(str)
    def _offer(i):
        if risk[i]=="High" and "Month-to-month" in contracts[i]: return ("CONTRACT_UPGRADE","Contract Upgrade Discount","20%","Switch to 1-year contract, 20% off for 6 months.","Sales team call")
        if risk[i]=="High" and "Fiber" in internet[i]: return ("TECHSUPPORT_BUNDLE","Free Tech Support Bundle","Free add-on","Free TechSupport for 3 months.","Auto-activate bundle")
        if "Electronic check" in payment[i]: return ("AUTOPAY_CASHBACK","Auto-Pay Cashback","Rs.200 cashback","Switch to auto-pay, get Rs.200 cashback.","Send SMS link")
        return ("STANDARD_LOYALTY","Loyalty Reward","Rs.100 credit","Rs.100 bill credit this month.","Apply automatically")
    offers=list(map(_offer,range(n)))
    return pd.DataFrame({"customerID":[f"C{i:04d}" for i in range(n)],"gender":np.random.choice(["Male","Female"],n),
        "SeniorCitizen":senior,"Partner":np.random.choice(["Yes","No"],n),"Dependents":np.random.choice(["Yes","No"],n),
        "tenure":tenure,"MonthlyCharges":monthly,"TotalCharges":np.round(monthly*tenure,2),
        "AnnualRevenue":np.round(monthly*12,2),"LoyaltyScore":np.round(tenure/72,3),
        "Churn":churn,"Contract":contracts,"InternetService":internet,"PaymentMethod":payment,
        "tenure_group":tg,"RevenueCategory":rc,"ChurnProbability":np.round(proba,4),
        "ChurnRisk":risk,"WillChurn_Predicted":churn,
        "OfferID":[o[0] for o in offers],"OfferTitle":[o[1] for o in offers],
        "OfferDiscount":[o[2] for o in offers],"OfferDescription":[o[3] for o in offers],
        "OfferAction":[o[4] for o in offers]})

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


# ════════════════════════════════════════════════════════════
# PLOTLY DARK THEME HELPER
# ════════════════════════════════════════════════════════════
def dark_fig(fig, height=320):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892a4", family="Inter"),
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(0,0,0,0)",
                   tickfont=dict(color="#8892a4")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(0,0,0,0)",
                   tickfont=dict(color="#8892a4")),
    )
    return fig


# ════════════════════════════════════════════════════════════
# SIDEBAR — Figma design
# ════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

all_pages    = ["Dashboard","ML Predictions","Recommendations","Admin Panel"]
admin_only   = {"Admin Panel"}
visible      = [p for p in all_pages if p not in admin_only or is_admin()]

page_icons = {
    "Dashboard":       "⊞",
    "ML Predictions":  "◎",
    "Recommendations": "○",
    "Admin Panel":     "⚙",
}

# Build full sidebar in HTML
initials = "".join([w[0].upper() for w in (user["full_name"] or user["username"]).split()[:2]])
nav_html = ""
for p in visible:
    active_style = ("background:linear-gradient(135deg,#1a6ef5,#00d4ff);"
                    "color:#fff;font-weight:600;") if st.session_state["page"] == p else \
                   ("background:transparent;color:#8892a4;")
    nav_html += f"""
    <div style="{active_style}border-radius:10px;padding:11px 16px;
                 display:flex;align-items:center;gap:12px;cursor:pointer;
                 font-size:0.88rem;margin:2px 0;transition:all .2s"
         class="nav-item">
        <span style="font-size:1rem">{page_icons.get(p,'○')}</span> {p}
    </div>"""

with st.sidebar:
    # Brand
    st.markdown(f"""
    <div style="padding:20px 20px 16px;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:12px">
        <div style="display:flex;align-items:center;gap:12px">
            <div style="width:40px;height:40px;border-radius:10px;
                        background:linear-gradient(135deg,#1a6ef5,#00d4ff);
                        display:flex;align-items:center;justify-content:center;font-size:1.2rem">🗄️</div>
            <div>
                <div style="font-size:0.95rem;font-weight:700;color:#fff">Telco ETL</div>
                <div style="font-size:0.72rem;color:#8892a4">Analytics</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Nav buttons — using actual st.buttons styled to match
    for p in visible:
        is_active = st.session_state["page"] == p
        label = f"{page_icons.get(p,'○')}  {p}"
        if is_active:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a6ef5,#00d4ff);
                        border-radius:10px;padding:11px 16px;
                        display:flex;align-items:center;gap:10px;
                        font-size:0.88rem;font-weight:600;color:#fff;margin:2px 8px">
                {page_icons.get(p,'○')} &nbsp; {p}
            </div>""", unsafe_allow_html=True)
        else:
            if st.button(f"{page_icons.get(p,'○')}  {p}", key=f"nav_{p}",
                         use_container_width=True):
                st.session_state["page"] = p
                log_event(user["username"], user["role"], "PAGE_VIEW",
                          p.encode("ascii","ignore").decode())
                st.rerun()

    # Filters
    st.markdown("""
    <div style="margin:20px 8px 8px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.06)">
        <div style="font-size:0.72rem;font-weight:600;color:#4a5568;
                    text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px">Filters</div>
    </div>
    """, unsafe_allow_html=True)

    c_opts = ["All"] + sorted(df_all["Contract"].dropna().unique().tolist())
    sel_c  = st.selectbox("Contract", c_opts, label_visibility="visible")
    i_opts = ["All"] + sorted(df_all["InternetService"].dropna().unique().tolist())
    sel_i  = st.selectbox("Internet Service", i_opts, label_visibility="visible")
    t_range = st.slider("Tenure (months)",
                         int(df_all["tenure"].min()), int(df_all["tenure"].max()),
                         (int(df_all["tenure"].min()), int(df_all["tenure"].max())))
    if st.button("↺  Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    # User + logout pinned to bottom
    st.markdown(f"""
    <div style="position:fixed;bottom:0;width:220px;
                background:#161b27;border-top:1px solid rgba(255,255,255,0.06);
                padding:16px 20px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
            <div style="width:36px;height:36px;border-radius:50%;
                        background:linear-gradient(135deg,#1a6ef5,#00d4ff);
                        display:flex;align-items:center;justify-content:center;
                        font-size:0.78rem;font-weight:700;color:#fff">{initials}</div>
            <div>
                <div style="font-size:0.82rem;font-weight:600;color:#fff">
                    {user['full_name'] or user['username']}</div>
                <div style="font-size:0.7rem;color:#8892a4">{user['role']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
    if st.button("⇥  Logout", use_container_width=True):
        logout()

# Apply filters
df = df_all.copy()
if sel_c != "All": df = df[df["Contract"] == sel_c]
if sel_i != "All": df = df[df["InternetService"] == sel_i]
df = df[(df["tenure"] >= t_range[0]) & (df["tenure"] <= t_range[1])]
page = st.session_state["page"]

# Plotly color sequence for dark theme
COLORS = ["#1a6ef5","#00d4ff","#00e676","#ffab00","#ff5252","#b388ff","#40c4ff"]


# ════════════════════════════════════════════════════════════
# PAGE 1 — Dashboard
# ════════════════════════════════════════════════════════════
if page == "Dashboard":
    total   = len(df); churned = int(df["Churn"].sum())
    churn_r = churned/total*100 if total else 0
    rev_mo  = df["MonthlyCharges"].sum()
    high_r  = int((df["ChurnRisk"]=="High").sum()) if HAS_ML else 0

    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">Churn Analytics Dashboard</div>
        <div class="page-sub">Real-time insights into customer retention and churn patterns</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='content-wrap'>", unsafe_allow_html=True)

    # KPI row
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card">
            <div><div class="kpi-label">Total Customers</div>
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-delta-pos">↑ Active accounts</div></div>
            <div class="kpi-icon icon-blue">👥</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card">
            <div><div class="kpi-label">Churn Rate</div>
            <div class="kpi-value">{churn_r:.1f}%</div>
            <div class="kpi-delta-neg">↓ {churned:,} customers lost</div></div>
            <div class="kpi-icon icon-red">📉</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
            <div><div class="kpi-label">Monthly Revenue</div>
            <div class="kpi-value">Rs.{rev_mo/1000:.1f}K</div>
            <div class="kpi-delta-pos">↑ Total active</div></div>
            <div class="kpi-icon icon-green">💰</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card">
            <div><div class="kpi-label">At-Risk Customers</div>
            <div class="kpi-value">{high_r:,}</div>
            <div class="kpi-delta-neg">⚡ High Priority</div></div>
            <div class="kpi-icon icon-amber">⚠️</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Charts row 1
    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title">Churn Rate by Contract Type</div>', unsafe_allow_html=True)
        cdf=(df.groupby("Contract")["Churn"].agg(["sum","count"])
             .rename(columns={"sum":"Churned","count":"Total"})
             .assign(**{"Churn %":lambda x:(x.Churned/x.Total*100).round(1)}).reset_index())
        fig=px.bar(cdf,x="Contract",y="Churn %",color_discrete_sequence=["#1a6ef5"],text="Churn %")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside",marker_color=["#1a6ef5","#00d4ff","#00e676"])
        st.plotly_chart(dark_fig(fig),use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title">Churn by Service Type</div>', unsafe_allow_html=True)
        idf=(df.groupby("InternetService")["Churn"].agg(["sum","count"])
             .rename(columns={"sum":"Churned","count":"Total"})
             .assign(**{"Churn %":lambda x:(x.Churned/x.Total*100).round(1)}).reset_index())
        fig=px.pie(idf,values="Churn %",names="InternetService",
                   color_discrete_sequence=["#1a6ef5","#00d4ff","#00e676","#b388ff"])
        fig.update_traces(textfont_color="#fff")
        st.plotly_chart(dark_fig(fig,300),use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Charts row 2
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="chart-card"><div class="chart-title">Monthly Charges — Churned vs Retained</div>', unsafe_allow_html=True)
        fig=px.box(df,x=df["Churn"].map({0:"Retained",1:"Churned"}),y="MonthlyCharges",
                   color=df["Churn"].map({0:"Retained",1:"Churned"}),
                   color_discrete_map={"Retained":"#1a6ef5","Churned":"#ff5252"})
        st.plotly_chart(dark_fig(fig),use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="chart-card"><div class="chart-title">Revenue at Risk by Segment</div>', unsafe_allow_html=True)
        rdf=(df[df["Churn"]==1].groupby("RevenueCategory")["MonthlyCharges"].sum().reset_index()
             .rename(columns={"MonthlyCharges":"Lost"}))
        fig=px.bar(rdf,x="RevenueCategory",y="Lost",
                   color="RevenueCategory",
                   color_discrete_map={"Low":"#00e676","Medium":"#ffab00","High":"#ff5252"},
                   text="Lost")
        fig.update_traces(texttemplate="Rs.%{text:,.0f}",textposition="outside")
        st.plotly_chart(dark_fig(fig),use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — ML Predictions
# ════════════════════════════════════════════════════════════
elif page == "ML Predictions":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">ML Churn Prediction</div>
        <div class="page-sub">Advanced machine learning models for customer churn prediction</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='content-wrap'>", unsafe_allow_html=True)

    # Model performance cards
    mres = metrics_data.get("model_results",{})
    best = metrics_data.get("best_model","")
    if mres:
        bm = mres.get(best, list(mres.values())[0])
        acc = bm.get("accuracy",0)*100
        pre = bm.get("precision",0)*100
        rec = bm.get("recall",0)*100
        f1  = bm.get("f1",0)*100

        st.markdown('<div class="section-title">Current Model Performance</div>', unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        for col, label, val, color in [
            (m1,"Accuracy",acc,"green"),
            (m2,"Precision",pre,"blue"),
            (m3,"Recall",rec,"gray"),
            (m4,"F1 Score",f1,"gray"),
        ]:
            bar_class = f"progress-bar-fill-{color}"
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="flex-direction:column;align-items:flex-start">
                    <div style="display:flex;justify-content:space-between;width:100%;margin-bottom:10px">
                        <div class="kpi-label">{label}</div>
                        <span style="color:#b388ff;font-size:1rem">✦</span>
                    </div>
                    <div class="kpi-value" style="font-size:1.5rem">{val:.1f}%</div>
                    <div class="progress-bar-wrap">
                        <div class="{bar_class}" style="width:{val:.0f}%"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Predictor + Risk profile
    col_pred, col_risk = st.columns(2)
    with col_pred:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Single Customer Prediction</div>', unsafe_allow_html=True)

        if not os.path.exists(os.path.join(MODELS_DIR,"churn_model.pkl")):
            st.warning("Run `python main.py --no-load` to train the model.")
        else:
            tenure   = st.slider("Tenure (months)",1,72,12,key="p_ten")
            contract = st.selectbox("Contract",["Month-to-month","One year","Two year"],key="p_con")
            payment  = st.selectbox("Payment",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"],key="p_pay")
            internet = st.selectbox("Internet",["Fiber optic","DSL","No"],key="p_int")
            monthly  = st.number_input("Monthly Charges (Rs.)",18.0,120.0,65.0,key="p_mo")
            senior   = st.selectbox("Senior Citizen",["No","Yes"],key="p_sen")

            if st.button("Run Prediction", use_container_width=True):
                customer = {
                    "SeniorCitizen":1 if senior=="Yes" else 0,"tenure":tenure,
                    "MonthlyCharges":monthly,"TotalCharges":monthly*tenure,
                    "gender_encoded":1,"Partner_encoded":0,"Dependents_encoded":0,
                    "PhoneService_encoded":1,"PaperlessBilling_encoded":1,
                    "AnnualRevenue":monthly*12,"LoyaltyScore":round(tenure/72,3)
                }
                from src.predict import predict_single
                result = predict_single(customer)
                st.session_state["last_pred"] = result
                st.session_state["last_customer"] = {
                    "Contract":contract,"InternetService":internet,
                    "PaymentMethod":payment,"SeniorCitizen":1 if senior=="Yes" else 0,
                    "tenure":tenure,"MonthlyCharges":monthly
                }
                log_event(user["username"],user["role"],"PREDICTION_RUN",
                          f"prob={result['probability']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_risk:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Customer Risk Profile</div>', unsafe_allow_html=True)
        if "last_pred" in st.session_state:
            result = st.session_state["last_pred"]
            prob   = result["probability"]
            risk   = result["risk"]
            color  = {"High":"#ff5252","Medium":"#ffab00","Low":"#00e676"}[risk]
            badge  = {"High":"badge-high","Medium":"badge-med","Low":"badge-low"}[risk]

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                number={"suffix":"%","font":{"size":36,"color":"#fff"}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#4a5568"},
                       "bar":{"color":color},
                       "bgcolor":"rgba(0,0,0,0)",
                       "steps":[{"range":[0,35],"color":"rgba(0,230,118,0.1)"},
                                 {"range":[35,65],"color":"rgba(255,171,0,0.1)"},
                                 {"range":[65,100],"color":"rgba(255,82,82,0.1)"}],
                       "bordercolor":"rgba(0,0,0,0)"}))
            fig.update_layout(height=220,paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#8892a4"),margin=dict(t=20,b=0,l=20,r=20))
            st.plotly_chart(fig,use_container_width=True)

            st.markdown(f"""
            <div style="text-align:center;margin-top:8px">
                <span class="{badge}">{'HIGH RISK' if risk=='High' else ('MEDIUM RISK' if risk=='Medium' else 'LOW RISK')}</span>
            </div>""", unsafe_allow_html=True)

            from src.recommend import recommend_for_single
            offer = recommend_for_single(st.session_state["last_customer"], result)
            st.markdown(f"""
            <div style="margin-top:16px;background:rgba(26,110,245,0.08);
                        border:1px solid rgba(26,110,245,0.2);border-radius:10px;padding:14px">
                <div style="font-size:0.82rem;font-weight:600;color:#60a5fa;margin-bottom:6px">
                    Recommended Offer</div>
                <div style="font-size:0.88rem;color:#e2e8f0;font-weight:600">{offer['title']}</div>
                <div style="font-size:0.78rem;color:#8892a4;margin-top:4px">{offer['description']}</div>
                <div style="font-size:0.78rem;color:#00e676;margin-top:8px;font-weight:600">
                    Discount: {offer['discount']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#4a5568">
                <div style="font-size:2.5rem;margin-bottom:12px">📊</div>
                <div style="font-size:0.85rem">Run a prediction to see customer profile</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance
    fimp = metrics_data.get("feature_importance",{})
    if fimp and mres:
        st.markdown('<div class="section-title">Feature Importance & Model Comparison</div>', unsafe_allow_html=True)
        fi1, fi2 = st.columns(2)
        with fi1:
            st.markdown('<div class="chart-card"><div class="chart-title">Top Feature Importances</div>', unsafe_allow_html=True)
            fidf=pd.DataFrame(list(fimp.items()),columns=["Feature","Importance"]).head(8)
            fig=px.bar(fidf,x="Importance",y="Feature",orientation="h",
                       color_discrete_sequence=["#1a6ef5"])
            fig.update_traces(marker_color=["#00d4ff","#1a6ef5","#00d4ff","#1a6ef5","#00d4ff","#1a6ef5","#00d4ff","#1a6ef5"][:len(fidf)])
            st.plotly_chart(dark_fig(fig,280),use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with fi2:
            st.markdown('<div class="chart-card"><div class="chart-title">Model ROC-AUC Comparison</div>', unsafe_allow_html=True)
            auc_df=pd.DataFrame([{"Model":k,"ROC-AUC":v.get("roc_auc",0)} for k,v in mres.items()])
            fig=px.bar(auc_df,x="Model",y="ROC-AUC",text="ROC-AUC",
                       color_discrete_sequence=["#1a6ef5"])
            fig.update_traces(texttemplate="%{text:.3f}",textposition="outside",
                              marker_color=["#1a6ef5" if n!=best else "#00e676" for n in auc_df["Model"]])
            fig.update_layout(yaxis_range=[0,1.1])
            st.plotly_chart(dark_fig(fig,280),use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — Recommendations
# ════════════════════════════════════════════════════════════
elif page == "Recommendations":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Retention Recommendations</div>
        <div class="page-sub">AI-powered strategies to reduce churn and improve customer loyalty</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='content-wrap'>", unsafe_allow_html=True)

    # Offer campaign cards
    campaigns = [
        {"icon":"🎁","title":"Contract Upgrade Discount","sub":"Month-to-month customers",
         "impact":"+20%","cost":"Custom","status":"Active","sdesc":"Switch to annual contract with 20% off for 6 months."},
        {"icon":"💰","title":"Auto-Pay Cashback Reward","sub":"Electronic check users",
         "impact":"+15%","cost":"Rs.200/user","status":"Proposed","sdesc":"Switch to automatic payment and get Rs.200 cashback."},
        {"icon":"📞","title":"Free Tech Support Bundle","sub":"Fiber optic high-risk",
         "impact":"+18%","cost":"Free add-on","status":"Active","sdesc":"3 months free TechSupport and OnlineSecurity."},
        {"icon":"⭐","title":"Loyalty Upgrade Program","sub":"Tenure > 36 months",
         "impact":"+25%","cost":"Free upgrade","status":"Proposed","sdesc":"Senior customers get dedicated support and 25% off."},
    ]
    st.markdown('<div class="section-title">Recommended Campaigns</div>', unsafe_allow_html=True)
    cc = st.columns(4)
    for i, camp in enumerate(campaigns):
        badge = "badge-active" if camp["status"]=="Active" else "badge-proposed"
        with cc[i]:
            st.markdown(f"""
            <div class="offer-card">
                <div class="offer-icon">{camp['icon']}</div>
                <div class="offer-title">{camp['title']}</div>
                <div class="offer-sub">{camp['sub']}</div>
                <div class="offer-row"><span class="offer-key">Retention Impact</span>
                    <span class="offer-val-green">{camp['impact']}</span></div>
                <div class="offer-row"><span class="offer-key">Est. Cost</span>
                    <span class="offer-val">{camp['cost']}</span></div>
                <div style="margin-top:12px;display:flex;justify-content:space-between;align-items:center">
                    <span class="{badge}">{camp['status']}</span>
                    <span style="color:#1a6ef5;font-size:0.8rem;font-weight:600;cursor:pointer">View Details</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # At-risk customer list
    if HAS_ML:
        st.markdown('<div class="section-title">High-Risk Customer Actions</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
            <span style="font-size:0.82rem;color:#8892a4">
                Personalised recommendations for customers most likely to churn</span>
        </div>""", unsafe_allow_html=True)

        at_risk = df[df["ChurnRisk"]=="High"].sort_values("ChurnProbability",ascending=False).head(8)
        for _, row in at_risk.iterrows():
            cid = str(row.get("customerID",""))
            ini = cid[:2].upper() if cid else "??"
            prob = row.get("ChurnProbability",0)
            offer_title = row.get("OfferTitle","Retention Call")
            contract    = row.get("Contract","—")
            monthly     = row.get("MonthlyCharges",0)

            st.markdown(f"""
            <div style="background:#1a2035;border:1px solid rgba(255,255,255,0.06);
                        border-radius:12px;padding:16px 20px;margin-bottom:10px;
                        display:flex;align-items:center;gap:16px">
                <div class="avatar-circle">{ini}</div>
                <div style="flex:1">
                    <div style="font-size:0.88rem;font-weight:600;color:#fff">{cid}</div>
                    <div style="font-size:0.75rem;color:#8892a4">{contract}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:0.78rem;color:#8892a4">Monthly</div>
                    <div style="font-size:0.88rem;color:#fff;font-weight:600">Rs.{monthly:.0f}</div>
                </div>
                <div style="text-align:center;min-width:90px">
                    <span class="badge-high">Risk: {prob:.0%}</span>
                </div>
                <div style="text-align:right;min-width:160px">
                    <div style="font-size:0.75rem;color:#8892a4">Recommended</div>
                    <div style="font-size:0.78rem;color:#60a5fa;font-weight:500">{offer_title}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        scols=["customerID","ChurnProbability","ChurnRisk","Contract",
               "MonthlyCharges","tenure","OfferTitle","OfferDiscount"]
        scols=[c for c in scols if c in df.columns]
        csv_data=df[df["ChurnRisk"]=="High"][scols].to_csv(index=False).encode()
        st.download_button("⬇️ Export High-Risk List", csv_data,
                           "high_risk_customers.csv","text/csv", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — Admin Panel
# ════════════════════════════════════════════════════════════
elif page == "Admin Panel":
    if not is_admin():
        st.error("Access Denied — Admin only."); st.stop()

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Admin Panel</div>
        <div class="page-sub">Manage users, ETL processes, and system configuration</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='content-wrap'>", unsafe_allow_html=True)

    # System KPIs
    users_list = get_all_users()
    log_data   = get_access_log()
    log_df     = pd.DataFrame(log_data) if log_data else pd.DataFrame()

    s1,s2,s3,s4 = st.columns(4)
    with s1:
        st.markdown(f"""<div class="kpi-card">
            <div><div class="kpi-label">Total Users</div>
            <div class="kpi-value">{len(users_list)}</div>
            <div class="kpi-delta-pos">Active accounts</div></div>
            <div class="kpi-icon icon-blue">👥</div></div>""", unsafe_allow_html=True)
    with s2:
        logins = len(log_df[log_df["event"]=="LOGIN_SUCCESS"]) if not log_df.empty else 0
        st.markdown(f"""<div class="kpi-card">
            <div><div class="kpi-label">Successful Logins</div>
            <div class="kpi-value">{logins}</div>
            <div class="kpi-delta-pos">Total sessions</div></div>
            <div class="kpi-icon icon-green">✅</div></div>""", unsafe_allow_html=True)
    with s3:
        failed = len(log_df[log_df["event"]=="LOGIN_FAILED"]) if not log_df.empty else 0
        st.markdown(f"""<div class="kpi-card">
            <div><div class="kpi-label">Failed Logins</div>
            <div class="kpi-value">{failed}</div>
            <div class="kpi-delta-neg">Security alerts</div></div>
            <div class="kpi-icon icon-red">🔒</div></div>""", unsafe_allow_html=True)
    with s4:
        views = len(log_df[log_df["event"]=="PAGE_VIEW"]) if not log_df.empty else 0
        st.markdown(f"""<div class="kpi-card">
            <div><div class="kpi-label">Page Views</div>
            <div class="kpi-value">{views}</div>
            <div class="kpi-delta-pos">Total navigation</div></div>
            <div class="kpi-icon icon-amber">👁️</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["👥  User Management","🔑  Change Password","📋  Access Log"])

    with tab1:
        # User table — Figma style
        st.markdown("""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
            <div style="font-size:0.95rem;font-weight:600;color:#fff">User Accounts</div>
        </div>
        <div style="background:#1a2035;border:1px solid rgba(255,255,255,0.06);border-radius:12px;overflow:hidden;margin-bottom:24px">
            <div style="display:grid;grid-template-columns:2fr 1fr 1fr;padding:12px 20px;
                        border-bottom:1px solid rgba(255,255,255,0.06)">
                <span style="font-size:0.75rem;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:.06em">Name</span>
                <span style="font-size:0.75rem;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:.06em">Role</span>
                <span style="font-size:0.75rem;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:.06em">Created</span>
            </div>""", unsafe_allow_html=True)

        for u in users_list:
            ini = "".join([w[0].upper() for w in u["username"].split()[:2]])
            role_badge = "badge-admin-role" if u["role"]=="admin" else "badge-user-role"
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:2fr 1fr 1fr;padding:14px 20px;
                        border-bottom:1px solid rgba(255,255,255,0.04);align-items:center">
                <div style="display:flex;align-items:center;gap:10px">
                    <div class="avatar-circle" style="width:32px;height:32px;font-size:0.7rem">{ini}</div>
                    <div>
                        <div style="font-size:0.85rem;font-weight:600;color:#fff">{u['username']}</div>
                        <div style="font-size:0.72rem;color:#8892a4">{u.get('full_name','')}</div>
                    </div>
                </div>
                <span class="{role_badge}">{u['role'].capitalize()}</span>
                <span style="font-size:0.78rem;color:#8892a4">{u.get('created_at','—')}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        col_add, col_del = st.columns(2)
        with col_add:
            st.markdown('<div class="chart-card"><div class="chart-title">Add New User</div>', unsafe_allow_html=True)
            new_uname = st.text_input("Username",  key="nu")
            new_fname = st.text_input("Full Name", key="nf")
            new_pwd   = st.text_input("Password",  type="password", key="np")
            new_role  = st.selectbox("Role",["user","admin"], key="nr")
            if st.button("Add User", use_container_width=True, key="add_user_btn"):
                ok, msg = add_user(new_uname, new_pwd, new_role, new_fname, user["username"])
                st.success(msg) if ok else st.error(msg)
                if ok: st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_del:
            st.markdown('<div class="chart-card"><div class="chart-title">Delete User</div>', unsafe_allow_html=True)
            del_opts = [u["username"] for u in users_list if u["username"] != user["username"]]
            if del_opts:
                del_target = st.selectbox("Select user to delete", del_opts)
                st.warning(f"Permanently deletes **{del_target}** and all their access.")
                if st.button("Delete User", use_container_width=True, key="del_user_btn"):
                    ok, msg = delete_user(del_target, user["username"])
                    st.success(msg) if ok else st.error(msg)
                    if ok: st.rerun()
            else:
                st.info("No other users to delete.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="chart-card"><div class="chart-title">Change Password</div>', unsafe_allow_html=True)
        col_pw, _ = st.columns([1,1])
        with col_pw:
            all_unames  = [u["username"] for u in get_all_users()]
            pwd_target  = st.selectbox("Select User", all_unames)
            new_pwd1    = st.text_input("New Password",     type="password", key="pw1")
            new_pwd2    = st.text_input("Confirm Password", type="password", key="pw2")
            if st.button("Update Password", use_container_width=True):
                if new_pwd1 != new_pwd2: st.error("Passwords do not match.")
                else:
                    ok, msg = change_password(pwd_target, new_pwd1, user["username"])
                    st.success(msg) if ok else st.error(msg)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        if not log_df.empty:
            fc1,fc2,fc3 = st.columns(3)
            with fc1:
                ev_opts=["All"]+sorted(log_df["event"].unique().tolist())
                sel_ev=st.selectbox("Event",ev_opts)
            with fc2:
                un_opts=["All"]+sorted(log_df["username"].unique().tolist())
                sel_un=st.selectbox("User",un_opts)
            with fc3:
                n_rows=st.number_input("Show last N",10,1000,50,step=10)

            filtered=log_df.copy()
            if sel_ev!="All": filtered=filtered[filtered["event"]==sel_ev]
            if sel_un!="All": filtered=filtered[filtered["username"]==sel_un]
            filtered=filtered.tail(int(n_rows)).sort_values("timestamp",ascending=False)

            st.dataframe(filtered,use_container_width=True,height=360,hide_index=True)
            csv_log=log_df.to_csv(index=False).encode()
            st.download_button("Download Full Log",csv_log,"access_log.csv","text/csv",use_container_width=True)
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
        Divyesh Joshi &nbsp;|&nbsp; MC24097 &nbsp;|&nbsp; MCA-II Sem IV &nbsp;|&nbsp;
        IIMS Chinchwad, Pune &nbsp;|&nbsp; Savitribai Phule Pune University &nbsp;|&nbsp; 2025-26
    </div>
</div>
""", unsafe_allow_html=True)
