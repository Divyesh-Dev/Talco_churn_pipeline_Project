# ============================================================
# app.py — Telecom Churn Analytics Dashboard
# Sidebar navigation | Auth | 5 Pages
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
    page_title="Telecom Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
.offer-card  { background:#f0f4ff; border-left:4px solid #2a5298; border-radius:8px; padding:14px 18px; margin-top:12px; }
.section-hdr { font-size:1.2rem; font-weight:700; color:#2a5298; border-left:4px solid #2a5298; padding-left:10px; margin:18px 0 10px; }
.user-badge-admin { background:#fff0f0; color:#c0392b; border-radius:20px; padding:3px 12px; font-size:0.8rem; font-weight:600; }
.user-badge-user  { background:#f0fff4; color:#1a7f3c; border-radius:20px; padding:3px 12px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Gate ─────────────────────────────────────────────────────
if not is_logged_in():
    render_login_page()
    st.stop()

user = current_user()

# ── Data loaders ─────────────────────────────────────────────
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
        if risk[i]=="High" and "Month-to-month" in contracts[i]: return ("CONTRACT_UPGRADE","Contract Upgrade Discount","20%","Switch to 1-year, 20% off for 6 months.","Sales team call")
        if risk[i]=="High" and "Fiber" in internet[i]: return ("TECHSUPPORT_BUNDLE","Free Tech Support Bundle","Free add-on","Free TechSupport 3 months.","Auto-activate")
        if "Electronic check" in payment[i]: return ("AUTOPAY_CASHBACK","Auto-Pay Cashback","Rs.200 cashback","Switch to auto-pay.","Send SMS")
        return ("STANDARD_LOYALTY","Loyalty Reward","Rs.100 credit","Rs.100 bill credit.","Apply auto")
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
    candidates = [c for c in df.columns if c.startswith(f"{col}_")]
    if candidates:
        df = df.copy()
        df[col] = df[candidates].idxmax(axis=1).str.replace(f"{col}_","",regex=False)
    return df

for _c in ["Contract","InternetService","PaymentMethod"]:
    df_all = _reconstruct(df_all, _c)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn Analytics")

    badge_class = "user-badge-admin" if user["role"]=="admin" else "user-badge-user"
    st.markdown(
        f"👤 **{user['full_name'] or user['username']}**&nbsp;&nbsp;"
        f"<span class='{badge_class}'>{user['role'].upper()}</span>",
        unsafe_allow_html=True)
    st.caption(f"Data: **{source}**")
    st.divider()

    all_pages  = ["📊 Executive Dashboard","🔮 Churn Predictor",
                  "🚨 At-Risk Customers","🤖 Model Performance","🛡️ Admin Panel"]
    admin_only = {"🤖 Model Performance","🛡️ Admin Panel"}
    visible    = [p for p in all_pages if p not in admin_only or is_admin()]

    page = st.radio("Navigate", visible)

    page_clean = page.encode("ascii", errors="ignore").decode("ascii").strip()
    log_event(user["username"], user["role"], "PAGE_VIEW", page_clean)

    st.divider()
    st.markdown("### Filters")
    c_opts = ["All"]+sorted(df_all["Contract"].dropna().unique().tolist())
    sel_c  = st.selectbox("Contract", c_opts)
    i_opts = ["All"]+sorted(df_all["InternetService"].dropna().unique().tolist())
    sel_i  = st.selectbox("Internet Service", i_opts)
    t_range = st.slider("Tenure (months)",
                         int(df_all["tenure"].min()), int(df_all["tenure"].max()),
                         (int(df_all["tenure"].min()), int(df_all["tenure"].max())))
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    st.divider()
    if st.button("🚪 Logout", use_container_width=True, type="primary"):
        logout()

# Apply filters
df = df_all.copy()
if sel_c != "All": df = df[df["Contract"] == sel_c]
if sel_i != "All": df = df[df["InternetService"] == sel_i]
df = df[(df["tenure"] >= t_range[0]) & (df["tenure"] <= t_range[1])]


# ══════════════════════════════════════════════════════════════
# PAGE 1 — Executive Dashboard
# ══════════════════════════════════════════════════════════════
if page == "📊 Executive Dashboard":
    st.title("📊 Executive Dashboard")
    st.caption(f"Showing **{len(df):,}** customers after filters")

    total=len(df); churned=int(df["Churn"].sum())
    churn_r=churned/total*100 if total else 0
    rev_loss=df[df["Churn"]==1]["MonthlyCharges"].sum()
    high_risk=int((df["ChurnRisk"]=="High").sum()) if HAS_ML else 0

    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("👥 Customers",    f"{total:,}")
    c2.metric("🚪 Churned",      f"{churned:,}")
    c3.metric("📉 Churn Rate",   f"{churn_r:.1f}%")
    c4.metric("💸 Rev Lost/mo",  f"Rs.{rev_loss:,.0f}")
    c5.metric("⏳ Avg Tenure",   f"{df['tenure'].mean():.1f} mo")
    c6.metric("🔴 High Risk",    f"{high_risk:,}")
    st.divider()

    ca,cb=st.columns([1,2])
    with ca:
        st.markdown('<div class="section-hdr">Churn Split</div>',unsafe_allow_html=True)
        fig=px.pie(values=[total-churned,churned],names=["Retained","Churned"],
                   color_discrete_sequence=["#2a5298","#e74c3c"],hole=0.5)
        fig.update_layout(height=280,margin=dict(t=5,b=5),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with cb:
        st.markdown('<div class="section-hdr">Churn Rate by Contract</div>',unsafe_allow_html=True)
        cdf=(df.groupby("Contract")["Churn"].agg(["sum","count"])
             .rename(columns={"sum":"Churned","count":"Total"})
             .assign(**{"Churn %":lambda x:(x.Churned/x.Total*100).round(1)}).reset_index())
        fig=px.bar(cdf,x="Contract",y="Churn %",color="Churn %",
                   color_continuous_scale=["#2a5298","#e74c3c"],text="Churn %")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig.update_layout(height=280,margin=dict(t=5,b=5),coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)

    cc,cd=st.columns(2)
    with cc:
        st.markdown('<div class="section-hdr">Churn by Internet Service</div>',unsafe_allow_html=True)
        idf=(df.groupby("InternetService")["Churn"].agg(["sum","count"])
             .rename(columns={"sum":"Churned","count":"Total"})
             .assign(**{"Churn %":lambda x:(x.Churned/x.Total*100).round(1)}).reset_index())
        fig=px.bar(idf,x="InternetService",y="Churn %",color="InternetService",
                   color_discrete_sequence=px.colors.qualitative.Set2,text="Churn %")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig.update_layout(height=270,margin=dict(t=5,b=5),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with cd:
        st.markdown('<div class="section-hdr">Churn by Payment Method</div>',unsafe_allow_html=True)
        pdf=(df.groupby("PaymentMethod")["Churn"].agg(["sum","count"])
             .rename(columns={"sum":"Churned","count":"Total"})
             .assign(**{"Churn %":lambda x:(x.Churned/x.Total*100).round(1)}).reset_index())
        fig=px.bar(pdf,x="PaymentMethod",y="Churn %",color="Churn %",
                   color_continuous_scale=["#27ae60","#e74c3c"],text="Churn %")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig.update_layout(height=270,margin=dict(t=5,b=5),coloraxis_showscale=False,xaxis_tickangle=-15)
        st.plotly_chart(fig,use_container_width=True)

    ce,cf=st.columns(2)
    with ce:
        st.markdown('<div class="section-hdr">Monthly Charges: Churned vs Retained</div>',unsafe_allow_html=True)
        fig=px.box(df,x=df["Churn"].map({0:"Retained",1:"Churned"}),y="MonthlyCharges",
                   color=df["Churn"].map({0:"Retained",1:"Churned"}),
                   color_discrete_map={"Retained":"#2a5298","Churned":"#e74c3c"})
        fig.update_layout(height=270,margin=dict(t=5,b=5),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with cf:
        st.markdown('<div class="section-hdr">Revenue at Risk by Segment</div>',unsafe_allow_html=True)
        rdf=(df[df["Churn"]==1].groupby("RevenueCategory")["MonthlyCharges"].sum().reset_index()
             .rename(columns={"MonthlyCharges":"Lost"}))
        fig=px.bar(rdf,x="RevenueCategory",y="Lost",color="RevenueCategory",
                   color_discrete_sequence=["#27ae60","#f39c12","#e74c3c"],text="Lost")
        fig.update_traces(texttemplate="Rs.%{text:,.0f}",textposition="outside")
        fig.update_layout(height=270,margin=dict(t=5,b=5),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — Churn Predictor
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Churn Predictor":
    st.title("🔮 Churn Predictor")
    st.markdown("Enter customer details to predict churn risk and get a retention offer.")
    if not os.path.exists(os.path.join(MODELS_DIR,"churn_model.pkl")):
        st.error("Model not found. Run `python main.py --no-load` first.")
        st.stop()

    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("**Account Details**")
        tenure=st.slider("Tenure (months)",1,72,12)
        contract=st.selectbox("Contract Type",["Month-to-month","One year","Two year"])
        payment=st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        paperless=st.selectbox("Paperless Billing",["Yes","No"])
    with c2:
        st.markdown("**Services**")
        internet=st.selectbox("Internet Service",["Fiber optic","DSL","No"])
        phone=st.selectbox("Phone Service",["Yes","No"])
        monthly=st.number_input("Monthly Charges (Rs.)",18.0,120.0,65.0,0.5)
        total_c=st.number_input("Total Charges (Rs.)",0.0,9000.0,float(monthly*tenure),1.0)
    with c3:
        st.markdown("**Demographics**")
        senior=st.selectbox("Senior Citizen",["No","Yes"])
        gender=st.selectbox("Gender",["Male","Female"])
        partner=st.selectbox("Partner",["Yes","No"])
        dependents=st.selectbox("Dependents",["Yes","No"])

    st.divider()
    if st.button("🔮 Predict Churn & Get Offer", type="primary", use_container_width=True):
        customer={"SeniorCitizen":1 if senior=="Yes" else 0,"tenure":tenure,
                  "MonthlyCharges":monthly,"TotalCharges":total_c,
                  "gender_encoded":1 if gender=="Male" else 0,
                  "Partner_encoded":1 if partner=="Yes" else 0,
                  "Dependents_encoded":1 if dependents=="Yes" else 0,
                  "PhoneService_encoded":1 if phone=="Yes" else 0,
                  "PaperlessBilling_encoded":1 if paperless=="Yes" else 0,
                  "AnnualRevenue":monthly*12,"LoyaltyScore":round(tenure/72,3)}
        customer_raw={"Contract":contract,"InternetService":internet,"PaymentMethod":payment,
                      "SeniorCitizen":1 if senior=="Yes" else 0,"tenure":tenure,"MonthlyCharges":monthly}
        from src.predict   import predict_single
        from src.recommend import recommend_for_single
        result=predict_single(customer)
        offer=recommend_for_single(customer_raw,result)
        log_event(user["username"],user["role"],"PREDICTION_RUN",
                  f"prob={result['probability']:.2f} risk={result['risk']}")
        prob=result["probability"]; risk=result["risk"]
        color={"High":"#e74c3c","Medium":"#f39c12","Low":"#27ae60"}[risk]
        fig=go.Figure(go.Indicator(mode="gauge+number",value=prob*100,
            number={"suffix":"%","font":{"size":40}},
            title={"text":"Churn Probability","font":{"size":18}},
            gauge={"axis":{"range":[0,100]},"bar":{"color":color},
                   "steps":[{"range":[0,35],"color":"#d4edda"},
                             {"range":[35,65],"color":"#fef3cd"},
                             {"range":[65,100],"color":"#fde8e8"}]}))
        fig.update_layout(height=270,margin=dict(t=30,b=10,l=20,r=20))
        cg,co=st.columns(2)
        with cg:
            st.plotly_chart(fig,use_container_width=True)
            verdict="🔴 HIGH RISK" if risk=="High" else ("🟡 MEDIUM RISK" if risk=="Medium" else "🟢 LOW RISK")
            st.markdown(f"<h3 style='text-align:center;color:{color}'>{verdict}</h3>",unsafe_allow_html=True)
        with co:
            st.markdown("### 🎁 Recommended Retention Offer")
            st.markdown(f"""<div class="offer-card">
<h4 style="margin:0 0 8px;color:#2a5298">🏷️ {offer['title']}</h4>
<p style="margin:0 0 6px">{offer['description']}</p>
<p style="margin:0 0 4px"><strong>Discount:</strong> {offer['discount']}</p>
<p style="margin:0;color:#555"><strong>Action:</strong> {offer['action']}</p>
</div>""",unsafe_allow_html=True)
            st.markdown(f"""
| Field | Value |
|---|---|
| Churn Probability | **{prob:.1%}** |
| Risk Level | **{risk}** |
| Prediction | **{'Will Churn' if result['will_churn'] else 'Will Stay'}** |
| Annual Revenue at Risk | **Rs.{monthly*12:,.0f}** |
""")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — At-Risk Customers
# ══════════════════════════════════════════════════════════════
elif page == "🚨 At-Risk Customers":
    st.title("🚨 At-Risk Customer List")
    if not HAS_ML:
        st.warning("Run `python main.py --no-load` to generate predictions.")
        st.stop()

    c1,c2,c3=st.columns(3)
    with c1: risk_filter=st.multiselect("Risk Level",["High","Medium","Low"],default=["High","Medium"])
    with c2: min_prob=st.slider("Min Churn Probability",0.0,1.0,0.5,0.05)
    with c3:
        offer_opts=df["OfferTitle"].dropna().unique().tolist() if "OfferTitle" in df.columns else []
        offer_filter=st.multiselect("Filter by Offer",offer_opts)

    at_risk=df[df["ChurnRisk"].isin(risk_filter)&(df["ChurnProbability"]>=min_prob)].copy()
    if offer_filter and "OfferTitle" in at_risk.columns:
        at_risk=at_risk[at_risk["OfferTitle"].isin(offer_filter)]
    at_risk=at_risk.sort_values("ChurnProbability",ascending=False)

    k1,k2,k3,k4=st.columns(4)
    k1.metric("🚨 At-Risk",        f"{len(at_risk):,}")
    k2.metric("💸 Rev at Risk/mo", f"Rs.{at_risk['MonthlyCharges'].sum():,.0f}")
    k3.metric("📊 Avg Prob",       f"{at_risk['ChurnProbability'].mean():.1%}")
    k4.metric("📅 Avg Tenure",     f"{at_risk['tenure'].mean():.1f} mo")
    st.divider()

    if "OfferTitle" in at_risk.columns:
        ca2,cb2=st.columns(2)
        with ca2:
            st.markdown('<div class="section-hdr">Offers Recommended</div>',unsafe_allow_html=True)
            odf=at_risk["OfferTitle"].value_counts().reset_index(); odf.columns=["Offer","Count"]
            fig=px.bar(odf,x="Count",y="Offer",orientation="h",color="Count",
                       color_continuous_scale=["#a8c8ff","#2a5298"])
            fig.update_layout(height=280,margin=dict(t=5,b=5),coloraxis_showscale=False,yaxis_title="")
            st.plotly_chart(fig,use_container_width=True)
        with cb2:
            st.markdown('<div class="section-hdr">Risk Distribution</div>',unsafe_allow_html=True)
            rdf2=at_risk["ChurnRisk"].value_counts().reset_index(); rdf2.columns=["Risk","Count"]
            fig=px.pie(rdf2,values="Count",names="Risk",hole=0.4,
                       color="Risk",color_discrete_map={"High":"#e74c3c","Medium":"#f39c12","Low":"#27ae60"})
            fig.update_layout(height=280,margin=dict(t=5,b=5))
            st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="section-hdr">Customer Details</div>',unsafe_allow_html=True)
    scols=["customerID","ChurnProbability","ChurnRisk","Contract","InternetService",
           "PaymentMethod","MonthlyCharges","tenure","OfferTitle","OfferDiscount"]
    scols=[c for c in scols if c in at_risk.columns]
    disp=at_risk[scols].copy()
    disp["ChurnProbability"]=(disp["ChurnProbability"]*100).round(1).astype(str)+"%"
    st.dataframe(disp,use_container_width=True,height=380)
    csv_data=at_risk[scols].to_csv(index=False).encode()
    st.download_button("⬇️ Export to CSV",csv_data,"at_risk_customers.csv","text/csv",use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — Model Performance  (Admin only)
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    if not is_admin():
        st.error("🔒 Access Denied — Admin only."); st.stop()

    st.title("🤖 Model Performance")
    if not metrics_data:
        st.warning("Run `python main.py --no-load` to train the model."); st.stop()

    best=metrics_data.get("best_model","N/A")
    mres=metrics_data.get("model_results",{})
    fimp=metrics_data.get("feature_importance",{})

    st.markdown(f"### Best Model Selected: `{best}`")
    st.divider()

    st.markdown('<div class="section-hdr">Model Comparison</div>',unsafe_allow_html=True)
    rows=[{"Model":k,"Accuracy":f"{v.get('accuracy',0):.3f}","Precision":f"{v.get('precision',0):.3f}",
           "Recall":f"{v.get('recall',0):.3f}","F1":f"{v.get('f1',0):.3f}",
           "ROC-AUC":f"{v.get('roc_auc',0):.3f}","Best":"✅" if k==best else ""}
          for k,v in mres.items()]
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.divider()

    cp1,cp2=st.columns(2)
    with cp1:
        st.markdown('<div class="section-hdr">Feature Importance (Top 10)</div>',unsafe_allow_html=True)
        if fimp:
            fidf=pd.DataFrame(list(fimp.items()),columns=["Feature","Importance"]).head(10)
            fig=px.bar(fidf,x="Importance",y="Feature",orientation="h",color="Importance",
                       color_continuous_scale=["#a8c8ff","#2a5298"])
            fig.update_layout(height=340,margin=dict(t=5,b=5),coloraxis_showscale=False,yaxis_title="")
            st.plotly_chart(fig,use_container_width=True)
    with cp2:
        st.markdown('<div class="section-hdr">Confusion Matrix</div>',unsafe_allow_html=True)
        cm=mres.get(best,{}).get("confusion_matrix")
        if cm:
            cmdf=pd.DataFrame(cm,index=["Actual: No Churn","Actual: Churn"],
                              columns=["Pred: No Churn","Pred: Churn"])
            fig=px.imshow(cmdf,text_auto=True,aspect="auto",
                          color_continuous_scale=["#eaf2ff","#2a5298"])
            fig.update_layout(height=300,margin=dict(t=5,b=5))
            st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="section-hdr">ROC-AUC Comparison</div>',unsafe_allow_html=True)
    auc_df=pd.DataFrame([{"Model":k,"ROC-AUC":v.get("roc_auc",0)} for k,v in mres.items()])
    fig=px.bar(auc_df,x="Model",y="ROC-AUC",color="Model",text="ROC-AUC",
               color_discrete_sequence=["#2a5298","#e74c3c","#27ae60"])
    fig.update_traces(texttemplate="%{text:.3f}",textposition="outside")
    fig.update_layout(height=260,margin=dict(t=5,b=5),showlegend=False,yaxis_range=[0,1.1])
    st.plotly_chart(fig,use_container_width=True)

    if fimp:
        st.divider()
        st.markdown("### Top Churn Drivers")
        top3=list(fimp.keys())[:3]
        cols=st.columns(3)
        for col,feat in zip(cols,top3):
            col.info(f"**#{top3.index(feat)+1}:** `{feat}`\n\nImportance: **{fimp[feat]:.4f}**")


# ══════════════════════════════════════════════════════════════
# PAGE 5 — Admin Panel  (Admin only)
# ══════════════════════════════════════════════════════════════
elif page == "🛡️ Admin Panel":
    if not is_admin():
        st.error("🔒 Access Denied — Admin only."); st.stop()

    st.title("🛡️ Admin Panel")
    tab1,tab2,tab3=st.tabs(["👥 User Management","🔑 Change Password","📋 Access Log"])

    with tab1:
        st.markdown('<div class="section-hdr">All Users</div>',unsafe_allow_html=True)
        users_list=get_all_users()
        st.dataframe(pd.DataFrame(users_list),use_container_width=True,hide_index=True)
        st.divider()

        col_add,col_del=st.columns(2)
        with col_add:
            st.markdown("#### Add New User")
            with st.container(border=True):
                new_username=st.text_input("Username",  key="new_uname")
                new_fullname=st.text_input("Full Name", key="new_fname")
                new_password=st.text_input("Password",  type="password",key="new_pwd")
                new_role    =st.selectbox("Role",["user","admin"],key="new_role")
                if st.button("Add User",type="primary",use_container_width=True):
                    ok,msg=add_user(new_username,new_password,new_role,new_fullname,user["username"])
                    st.success(msg) if ok else st.error(msg)
                    if ok: st.rerun()

        with col_del:
            st.markdown("#### Delete User")
            with st.container(border=True):
                del_options=[u["username"] for u in users_list if u["username"]!=user["username"]]
                if del_options:
                    del_target=st.selectbox("Select user",del_options)
                    st.warning(f"This will permanently delete **{del_target}**.")
                    if st.button("Delete User",type="primary",use_container_width=True):
                        ok,msg=delete_user(del_target,user["username"])
                        st.success(msg) if ok else st.error(msg)
                        if ok: st.rerun()
                else:
                    st.info("No other users to delete.")

    with tab2:
        st.markdown('<div class="section-hdr">Change Password</div>',unsafe_allow_html=True)
        with st.container(border=True):
            all_usernames=[u["username"] for u in get_all_users()]
            pwd_target=st.selectbox("Select User",all_usernames,key="pwd_target")
            new_pwd1=st.text_input("New Password",    type="password",key="pw1")
            new_pwd2=st.text_input("Confirm Password",type="password",key="pw2")
            if st.button("Update Password",type="primary"):
                if new_pwd1!=new_pwd2: st.error("Passwords do not match.")
                else:
                    ok,msg=change_password(pwd_target,new_pwd1,user["username"])
                    st.success(msg) if ok else st.error(msg)

    with tab3:
        st.markdown('<div class="section-hdr">Access Log</div>',unsafe_allow_html=True)
        log_data=get_access_log()
        if log_data:
            log_df=pd.DataFrame(log_data)
            lc1,lc2,lc3=st.columns(3)
            with lc1:
                ev_opts=["All"]+sorted(log_df["event"].unique().tolist())
                sel_ev=st.selectbox("Filter by Event",ev_opts)
            with lc2:
                un_opts=["All"]+sorted(log_df["username"].unique().tolist())
                sel_un=st.selectbox("Filter by User",un_opts)
            with lc3:
                n_rows=st.number_input("Show last N",10,1000,100,step=10)

            filtered=log_df.copy()
            if sel_ev!="All": filtered=filtered[filtered["event"]==sel_ev]
            if sel_un!="All": filtered=filtered[filtered["username"]==sel_un]
            filtered=filtered.tail(int(n_rows)).sort_values("timestamp",ascending=False)

            lk1,lk2,lk3,lk4=st.columns(4)
            lk1.metric("Total Events",        len(log_df))
            lk2.metric("Successful Logins",   len(log_df[log_df["event"]=="LOGIN_SUCCESS"]))
            lk3.metric("Failed Logins",       len(log_df[log_df["event"]=="LOGIN_FAILED"]))
            lk4.metric("Page Views",          len(log_df[log_df["event"]=="PAGE_VIEW"]))

            st.dataframe(filtered,use_container_width=True,height=380,hide_index=True)
            csv_log=log_df.to_csv(index=False).encode()
            st.download_button("Download Full Log",csv_log,"access_log.csv","text/csv",use_container_width=True)
        else:
            st.info("No log entries yet.")


# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;border-top:1px solid #e0e0e0;padding:18px 0;text-align:center">
    <div style="font-size:0.75rem;font-weight:600;color:#555;margin-bottom:4px">
        Design and Implementation of Scalable ETL Pipeline for Telco Customer Churn Analysis
        using PySpark, PostgreSQL and Streamlit
    </div>
    <div style="font-size:0.7rem;color:#999">
        Divyesh Joshi &nbsp;|&nbsp; MC24097 &nbsp;|&nbsp; MCA-II Sem IV
        &nbsp;|&nbsp; IIMS Chinchwad, Pune &nbsp;|&nbsp;
        Savitribai Phule Pune University &nbsp;|&nbsp; 2025-26
    </div>
</div>
""", unsafe_allow_html=True)
