# ============================================================
# transform.py — Phase 2: Cleaning + Transformation + Feature Engineering
# ============================================================

import numpy as np
import pandas as pd
from .config import TENURE_BINS, TENURE_LABELS, PROCESSED_DATA
from .utils import log
import os


# ── Step 1: Clean ────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    log("CLEAN", "Starting data cleaning …")
    df = df.copy()

    # TotalCharges has spaces instead of numeric values for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    filled = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    log("CLEAN", f"TotalCharges: fixed {filled} non-numeric entries → median imputed")

    # Drop duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    log("CLEAN", f"Dropped {before - len(df)} duplicate rows")

    # Standardise string columns (strip whitespace, title-case)
    str_cols = df.select_dtypes("object").columns.tolist()
    for col in str_cols:
        df[col] = df[col].str.strip()

    log("CLEAN", "Cleaning complete ✓")
    return df


# ── Step 2: Transform ────────────────────────────────────────

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    log("TRANSFORM", "Starting transformations …")
    df = df.copy()

    # Binary-encode Churn
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # Binary-encode gender
    df["gender_encoded"] = df["gender"].map({"Male": 1, "Female": 0}).astype(int)

    # Binary-encode Partner, Dependents, PhoneService, PaperlessBilling
    yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in yes_no_cols:
        df[f"{col}_encoded"] = df[col].map({"Yes": 1, "No": 0}).astype(int)

    # Preserve original label columns for dashboard BEFORE one-hot encoding
    for _col in ["Contract", "InternetService", "PaymentMethod"]:
        if _col in df.columns:
            df[f"{_col}_orig"] = df[_col]

    # One-hot encode multi-class categoricals
    ohe_cols = ["MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    # Restore plain-name columns (dashboard filters need Contract, InternetService, PaymentMethod)
    for _col in ["Contract", "InternetService", "PaymentMethod"]:
        if f"{_col}_orig" in df.columns:
            df.rename(columns={f"{_col}_orig": _col}, inplace=True)

    # Tenure group
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
    )

    log("TRANSFORM", "Transformations complete ✓")
    return df


# ── Step 3: Feature Engineering ──────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log("FEATURES", "Engineering new features …")
    df = df.copy()

    # Revenue category based on MonthlyCharges
    conditions = [
        df["MonthlyCharges"] < 35,
        (df["MonthlyCharges"] >= 35) & (df["MonthlyCharges"] < 70),
        df["MonthlyCharges"] >= 70,
    ]
    df["RevenueCategory"] = np.select(conditions, ["Low", "Medium", "High"], default="Low").astype(str)

    # Estimated total revenue potential (annualised)
    df["AnnualRevenue"] = df["MonthlyCharges"] * 12

    # Loyalty score: tenure normalised 0-1
    df["LoyaltyScore"] = df["tenure"] / df["tenure"].max()

    log("FEATURES", "Feature engineering complete ✓")
    return df


# ── Step 4: Validate ─────────────────────────────────────────

def validate_data(df: pd.DataFrame) -> bool:
    log("VALIDATE", "Running post-transform validations …")
    issues = []

    null_totals = df.isnull().sum()
    critical_nulls = null_totals[null_totals > 0]
    if not critical_nulls.empty:
        issues.append(f"Nulls found:\n{critical_nulls}")

    if df.duplicated().sum() > 0:
        issues.append("Duplicate rows detected after cleaning.")

    if df["Churn"].dtype not in [int, "int64", "int32"]:
        issues.append("Churn column is not integer-encoded.")

    if issues:
        for i in issues:
            log("VALIDATE", f"⚠️  {i}")
        return False

    log("VALIDATE", "All checks passed ✓")
    return True


# ── Master transform pipeline ─────────────────────────────────

def run_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning → transformation → feature engineering pipeline.
    Also saves a processed CSV for reference.
    """
    df = clean_data(df)
    df = transform_data(df)
    df = engineer_features(df)
    validate_data(df)

    # Save processed CSV
    os.makedirs(os.path.dirname(PROCESSED_DATA), exist_ok=True)
    df.to_csv(PROCESSED_DATA, index=False)
    log("TRANSFORM", f"Processed data saved → {PROCESSED_DATA}")

    return df
