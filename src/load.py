# ============================================================
# load.py — Phase 3: Load processed data into PostgreSQL
# ============================================================

import pandas as pd
from sqlalchemy import create_engine, text
from src.config import DB_URL, TABLE_CLEANED, TABLE_FEATURES, CHUNK_SIZE
from src.utils import log


def get_engine():
    """Return a SQLAlchemy engine for the configured PostgreSQL database."""
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log("LOAD", "Database connection successful ✓")
        return engine
    except Exception as e:
        raise ConnectionError(
            f"Cannot connect to PostgreSQL: {e}\n"
            "Check your DB_HOST / DB_USER / DB_PASSWORD env vars or src/config.py"
        )


def create_schema(engine):
    """Create the database and tables if they do not exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS customer_churn_raw (
        customerID       VARCHAR(50),
        gender           VARCHAR(10),
        SeniorCitizen    SMALLINT,
        Partner          VARCHAR(10),
        Dependents       VARCHAR(10),
        tenure           INT,
        PhoneService     VARCHAR(10),
        MultipleLines    VARCHAR(50),
        InternetService  VARCHAR(50),
        OnlineSecurity   VARCHAR(50),
        OnlineBackup     VARCHAR(50),
        DeviceProtection VARCHAR(50),
        TechSupport      VARCHAR(50),
        StreamingTV      VARCHAR(50),
        StreamingMovies  VARCHAR(50),
        Contract         VARCHAR(50),
        PaperlessBilling VARCHAR(10),
        PaymentMethod    VARCHAR(100),
        MonthlyCharges   FLOAT,
        TotalCharges     FLOAT,
        Churn            VARCHAR(10)
    );

    CREATE TABLE IF NOT EXISTS customer_churn_cleaned (
        customerID        VARCHAR(50) PRIMARY KEY,
        gender            VARCHAR(10),
        SeniorCitizen     SMALLINT,
        Partner           VARCHAR(10),
        Dependents        VARCHAR(10),
        tenure            INT,
        MonthlyCharges    FLOAT,
        TotalCharges      FLOAT,
        Churn             SMALLINT,
        tenure_group      VARCHAR(20),
        RevenueCategory   VARCHAR(10),
        AnnualRevenue     FLOAT,
        LoyaltyScore      FLOAT,
        Contract          VARCHAR(50),
        PaymentMethod     VARCHAR(100),
        InternetService   VARCHAR(50)
    );
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    log("LOAD", "Schema verified / created ✓")


def load_raw(df_raw: pd.DataFrame, engine):
    """Load the original (pre-transform) data for audit trail."""
    # Keep only the 21 original columns to match raw schema
    original_cols = [
        "customerID","gender","SeniorCitizen","Partner","Dependents",
        "tenure","PhoneService","MultipleLines","InternetService",
        "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
        "StreamingTV","StreamingMovies","Contract","PaperlessBilling",
        "PaymentMethod","MonthlyCharges","TotalCharges","Churn",
    ]
    cols = [c for c in original_cols if c in df_raw.columns]
    df_raw[cols].to_sql(
        TABLE_CLEANED.replace("cleaned", "raw"),   # customer_churn_raw
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=CHUNK_SIZE,
    )
    log("LOAD", f"Raw data loaded → table '{TABLE_CLEANED.replace('cleaned','raw')}' ✓")


def load_cleaned(df: pd.DataFrame, engine):
    """
    Load the cleaned + feature-engineered DataFrame into PostgreSQL.
    Only the analytical columns are kept (avoids dumping 60+ OHE cols).
    """
    keep_cols = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "MonthlyCharges", "TotalCharges", "Churn",
        "tenure_group", "RevenueCategory", "AnnualRevenue", "LoyaltyScore",
    ]
    # Recover Contract / PaymentMethod / InternetService from OHE if present
    for original_col in ["Contract", "PaymentMethod", "InternetService"]:
        ohe_candidates = [c for c in df.columns if c.startswith(f"{original_col}_")]
        if ohe_candidates and original_col not in df.columns:
            # Reverse-reconstruct the label from the 1-hot columns
            df[original_col] = (
                df[ohe_candidates]
                .idxmax(axis=1)
                .str.replace(f"{original_col}_", "", regex=False)
            )
        if original_col in df.columns:
            keep_cols.append(original_col)

    cols_to_load = [c for c in keep_cols if c in df.columns]
    df_load = df[cols_to_load].copy()
    df_load["tenure_group"] = df_load["tenure_group"].astype(str)

    df_load.to_sql(
        TABLE_CLEANED,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=CHUNK_SIZE,
    )
    log("LOAD", f"Cleaned data loaded → table '{TABLE_CLEANED}' ({len(df_load):,} rows) ✓")


def load_data(df_transformed: pd.DataFrame, df_raw: pd.DataFrame = None):
    """
    Master load function. Call this after run_transform().

    Parameters
    ----------
    df_transformed : pd.DataFrame  — output of run_transform()
    df_raw         : pd.DataFrame  — original extract_data() output (optional, for audit)
    """
    engine = get_engine()
    create_schema(engine)

    if df_raw is not None:
        load_raw(df_raw, engine)

    load_cleaned(df_transformed, engine)
    log("LOAD", "All data loaded successfully ✓")
