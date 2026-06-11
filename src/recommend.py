# ============================================================
# recommend.py — Retention Offer Recommendation Engine
# Maps each at-risk customer to a personalised retention offer
# ============================================================

import pandas as pd
from .utils import log

# ── Offer catalogue ──────────────────────────────────────────
# Each offer has: id, title, description, discount, action
OFFERS = {
    "CONTRACT_UPGRADE": {
        "id":          "CONTRACT_UPGRADE",
        "title":       "Contract Upgrade Discount",
        "description": "Switch to a 1-year contract and get 20% off your monthly bill for 6 months.",
        "discount":    "20%",
        "action":      "Sales team to call and offer contract upgrade",
        "priority":    1,
    },
    "TECHSUPPORT_BUNDLE": {
        "id":          "TECHSUPPORT_BUNDLE",
        "title":       "Free Tech Support Bundle",
        "description": "Complimentary TechSupport + OnlineSecurity add-on for 3 months.",
        "discount":    "Free add-on",
        "action":      "Auto-activate support bundle on account",
        "priority":    2,
    },
    "AUTOPAY_CASHBACK": {
        "id":          "AUTOPAY_CASHBACK",
        "title":       "Auto-Pay Cashback Reward",
        "description": "Switch to automatic payment and receive ₹200 cashback on next bill.",
        "discount":    "₹200 cashback",
        "action":      "Send SMS with auto-pay setup link",
        "priority":    3,
    },
    "LOYALTY_UPGRADE": {
        "id":          "LOYALTY_UPGRADE",
        "title":       "Loyalty Service Upgrade",
        "description": "As a valued long-term customer, enjoy a free service tier upgrade for 6 months.",
        "discount":    "Free upgrade",
        "action":      "Apply service upgrade automatically",
        "priority":    2,
    },
    "SENIOR_PLAN": {
        "id":          "SENIOR_PLAN",
        "title":       "Senior Customer Special Plan",
        "description": "Exclusive senior plan with 25% reduced monthly charge and dedicated support.",
        "discount":    "25%",
        "action":      "Assign dedicated customer care agent",
        "priority":    1,
    },
    "STREAMING_BUNDLE": {
        "id":          "STREAMING_BUNDLE",
        "title":       "Entertainment Bundle",
        "description": "Add StreamingTV + StreamingMovies at 50% off for 3 months.",
        "discount":    "50% for 3 months",
        "action":      "Send activation link via email",
        "priority":    3,
    },
    "RETENTION_CALL": {
        "id":          "RETENTION_CALL",
        "title":       "Priority Retention Call",
        "description": "High-risk customer flagged for immediate personal outreach by retention specialist.",
        "discount":    "Custom offer",
        "action":      "Escalate to retention team — call within 24 hours",
        "priority":    1,
    },
    "STANDARD_LOYALTY": {
        "id":          "STANDARD_LOYALTY",
        "title":       "Loyalty Appreciation Reward",
        "description": "Thank you for being with us. Enjoy ₹100 bill credit this month.",
        "discount":    "₹100 credit",
        "action":      "Apply bill credit automatically",
        "priority":    4,
    },
}


def recommend_offer(row: pd.Series) -> dict:
    """
    Rule-based offer recommendation for a single customer row.
    Returns the most relevant offer dict.
    """
    prob      = float(row.get("ChurnProbability", 0.5))
    risk      = str(row.get("ChurnRisk", "Medium"))
    contract  = str(row.get("Contract", ""))
    internet  = str(row.get("InternetService", ""))
    payment   = str(row.get("PaymentMethod", ""))
    senior    = int(row.get("SeniorCitizen", 0))
    tenure    = float(row.get("tenure", 12))
    monthly   = float(row.get("MonthlyCharges", 50))

    # Rule 1 — Highest priority: very high risk → retention call
    if prob >= 0.80:
        return OFFERS["RETENTION_CALL"]

    # Rule 2 — Senior citizen → senior plan
    if senior == 1 and risk in ["High", "Medium"]:
        return OFFERS["SENIOR_PLAN"]

    # Rule 3 — Month-to-month → push to annual contract
    if "Month-to-month" in contract and risk in ["High", "Medium"]:
        return OFFERS["CONTRACT_UPGRADE"]

    # Rule 4 — Fiber optic with no tech support → add support bundle
    if "Fiber" in internet and risk in ["High", "Medium"]:
        return OFFERS["TECHSUPPORT_BUNDLE"]

    # Rule 5 — Electronic check (highest churn payment method) → autopay cashback
    if "Electronic check" in payment and risk in ["High", "Medium"]:
        return OFFERS["AUTOPAY_CASHBACK"]

    # Rule 6 — Long tenure loyal customer at risk → loyalty upgrade
    if tenure > 36 and risk == "High":
        return OFFERS["LOYALTY_UPGRADE"]

    # Rule 7 — High monthly charge with no streaming → add streaming bundle
    if monthly > 70 and risk in ["High", "Medium"]:
        return OFFERS["STREAMING_BUNDLE"]

    # Default — standard loyalty reward for low/medium risk
    return OFFERS["STANDARD_LOYALTY"]


def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add OfferID, OfferTitle, OfferDescription, OfferDiscount, OfferAction
    columns to the dataframe for every customer.
    """
    log("RECOMMEND", f"Generating offers for {len(df):,} customers …")

    offers_applied = df.apply(recommend_offer, axis=1)

    df = df.copy()
    df["OfferID"]          = offers_applied.apply(lambda o: o["id"])
    df["OfferTitle"]       = offers_applied.apply(lambda o: o["title"])
    df["OfferDescription"] = offers_applied.apply(lambda o: o["description"])
    df["OfferDiscount"]    = offers_applied.apply(lambda o: o["discount"])
    df["OfferAction"]      = offers_applied.apply(lambda o: o["action"])

    # Summary log
    offer_counts = df["OfferTitle"].value_counts()
    log("RECOMMEND", "Offer distribution:")
    for offer, count in offer_counts.items():
        log("RECOMMEND", f"  {offer:<40} → {count:,} customers")

    log("RECOMMEND", "Recommendations complete ✓")
    return df


def recommend_for_single(customer_data: dict, prediction: dict) -> dict:
    """
    Generate an offer for a single customer (Streamlit form use).

    Parameters
    ----------
    customer_data : raw customer fields dict
    prediction    : output of predict_single()
    """
    row = pd.Series({**customer_data,
                     "ChurnProbability": prediction["probability"],
                     "ChurnRisk":        prediction["risk"]})
    return recommend_offer(row)
