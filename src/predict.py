# ============================================================
# predict.py — Score every customer with churn probability
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np
from .utils import log

_SRC  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
MODELS_DIR    = os.path.join(_ROOT, "models")
MODEL_PATH    = os.path.join(MODELS_DIR, "churn_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Train first:  python -m src.train_model"
        )
    model    = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    log("PREDICT", f"Model loaded. Features used: {len(features)}")
    return model, features


def predict_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ChurnProbability, ChurnRisk columns to the dataframe.

    Parameters
    ----------
    df : cleaned & feature-engineered DataFrame (output of run_transform)

    Returns
    -------
    df with extra columns: ChurnProbability, ChurnRisk, WillChurn_Predicted
    """
    model, features = load_model()

    available = [c for c in features if c in df.columns]
    X = df[available].copy()

    # Fill any nulls with column median
    X = X.fillna(X.median(numeric_only=True))

    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    df = df.copy()
    df["ChurnProbability"]   = np.round(proba, 4)
    df["WillChurn_Predicted"] = pred
    df["ChurnRisk"] = pd.cut(
        df["ChurnProbability"],
        bins=[-0.001, 0.35, 0.65, 1.001],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    churned_pct = pred.mean() * 100
    log("PREDICT", f"Scored {len(df):,} customers")
    log("PREDICT", f"Predicted churn: {pred.sum():,} ({churned_pct:.1f}%)")
    log("PREDICT", f"High risk: {(df['ChurnRisk']=='High').sum():,}")
    log("PREDICT", f"Medium risk: {(df['ChurnRisk']=='Medium').sum():,}")
    log("PREDICT", f"Low risk: {(df['ChurnRisk']=='Low').sum():,}")

    return df


def predict_single(customer: dict) -> dict:
    """
    Score a single customer dict (for the Streamlit prediction form).

    Parameters
    ----------
    customer : dict with keys matching FEATURE_COLS

    Returns
    -------
    dict: { probability, risk, will_churn }
    """
    model, features = load_model()
    row = pd.DataFrame([customer])

    available = [c for c in features if c in row.columns]
    X = row[available].fillna(0)

    # Pad missing columns with 0
    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features]

    proba = float(model.predict_proba(X)[0, 1])
    risk  = "High" if proba >= 0.65 else ("Medium" if proba >= 0.35 else "Low")

    return {
        "probability": round(proba, 4),
        "risk":        risk,
        "will_churn":  proba >= 0.5,
    }
