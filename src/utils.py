# ============================================================
# utils.py — Shared helpers
# ============================================================

import datetime


def log(stage: str, message: str):
    """Pretty-print a timestamped ETL log message."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{stage:^10}] {message}")


def validate_columns(df, required: list):
    """Raise an error if any required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    log("VALIDATE", f"All {len(required)} required columns present ✓")
