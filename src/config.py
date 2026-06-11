# ============================================================
# config.py — Central configuration for ETL pipeline
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA     = os.path.join(BASE_DIR, "data", "raw",       "Telco-Customer-Churn.csv")
PROCESSED_DATA = os.path.join(BASE_DIR, "data", "processed", "churn_cleaned.csv")
BACKUP_DATA  = os.path.join(BASE_DIR, "data", "backup",    "churn_backup.csv")

# ── PostgreSQL connection ────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME",     "etl_churn_db"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "Divyesh"),
}

# SQLAlchemy connection string (used by pandas .to_sql / read_sql)
DB_URL = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# ── Table names ───────────────────────────────────────────────
TABLE_RAW         = "customer_churn_raw"
TABLE_CLEANED     = "customer_churn_cleaned"
TABLE_FEATURES    = "customer_churn_features"

# ── ETL settings ─────────────────────────────────────────────
CHUNK_SIZE    = 5_000          # rows per chunk when loading to DB
PYSPARK_MASTER = "local[*]"   # change to spark://host:port for cluster

# ── Tenure bins ───────────────────────────────────────────────
TENURE_BINS   = [0, 12, 24, 48, 72]
TENURE_LABELS = ["0-1 Year", "1-2 Years", "2-4 Years", "4-6 Years"]
