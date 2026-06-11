-- ============================================================
-- schema.sql — PostgreSQL database schema
-- Run once to initialise the database:
--   psql -U postgres -f sql/schema.sql
-- ============================================================

-- Create database (run as superuser outside a transaction block)
-- CREATE DATABASE etl_churn_db;
-- \c etl_churn_db

-- ── Raw audit table ──────────────────────────────────────────
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

-- ── Cleaned / analytical table ────────────────────────────────
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
