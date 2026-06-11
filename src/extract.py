# ============================================================
# extract.py — Phase 1: Data Extraction
# ============================================================

import os
import shutil
import pandas as pd
from src.config import RAW_DATA, BACKUP_DATA
from src.utils import log, validate_columns

REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def extract_data(filepath: str = RAW_DATA) -> pd.DataFrame:
    """
    Read the raw Telco-Customer-Churn CSV, validate it, and
    create a timestamped backup copy.

    Returns
    -------
    pd.DataFrame  — raw (untouched) data
    """
    log("EXTRACT", f"Reading CSV: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download it from Kaggle: https://www.kaggle.com/blastchar/telco-customer-churn\n"
            f"and place it at: {filepath}"
        )

    df = pd.read_csv(filepath)

    log("EXTRACT", f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Validate required columns ────────────────────────────
    validate_columns(df, REQUIRED_COLUMNS)

    # ── Quick sanity report ──────────────────────────────────
    null_counts = df.isnull().sum()
    if null_counts.any():
        log("EXTRACT", f"Nulls detected:\n{null_counts[null_counts > 0]}")
    else:
        log("EXTRACT", "No null values in raw data.")

    dup_count = df.duplicated().sum()
    log("EXTRACT", f"Duplicate rows: {dup_count}")

    log("EXTRACT", f"Churn distribution:\n{df['Churn'].value_counts()}")

    # ── Backup ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(BACKUP_DATA), exist_ok=True)
    shutil.copy2(filepath, BACKUP_DATA)
    log("EXTRACT", f"Backup saved → {BACKUP_DATA}")

    return df
