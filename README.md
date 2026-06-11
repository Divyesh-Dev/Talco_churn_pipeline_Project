# 📡 Telecom Customer Churn ETL & Analytics Pipeline

**Student:** Divyesh Joshi | MC24097 | MCA-II Sem IV (2024-26)  
**Guide:** Prof. Yugandhara Patil  
**University:** Savitribai Phule Pune University — IIMS Chinchwad, Pune

---

## Project Overview

End-to-end ETL pipeline for Telecom Customer Churn analysis using:
**Python · Pandas · PySpark · PostgreSQL · Streamlit · Plotly**

---

## Quick Start (Step by Step)

### Step 1 — Clone & enter project
```bash
cd ETL_project
```

### Step 2 — Create virtual environment
```bash
python -m venv etl_env

# Windows
etl_env\Scripts\activate

# Linux / Mac
source etl_env/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download dataset
1. Go to: https://www.kaggle.com/blastchar/telco-customer-churn
2. Download `Telco-Customer-Churn.csv`
3. Place it at: `data/raw/Telco-Customer-Churn.csv`

### Step 5 — Setup PostgreSQL
```sql
-- In psql as superuser:
CREATE DATABASE etl_churn_db;
CREATE USER postgres WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE etl_churn_db TO postgres;
```

Or set environment variables to override defaults:
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=etl_churn_db
export DB_USER=postgres
export DB_PASSWORD=yourpassword
```

### Step 6 — Run the ETL pipeline
```bash
python main.py
```

Optional flags:
```bash
python main.py --spark      # also run PySpark pipeline
python main.py --no-load    # skip DB, output CSV only
```

### Step 7 — Launch Streamlit dashboard
```bash
streamlit run app.py
```
Open your browser at: http://localhost:8501

---

# ⚙️ Step 1 — Install Dependencies

```powershell
pip install pytest pytest-cov jupyter
```

---

# 📓 Step 2 — Run Notebooks

## Open All Notebooks

```powershell
jupyter notebook notebooks/
```

## Open Specific Notebook

```powershell
jupyter notebook notebooks/01_data_understanding.ipynb
```

## Execute Notebook Cells

Run all notebook cells in order:

- Menu → Cell → Run All
- OR use `Shift + Enter` cell by cell

---

# 🧪 Step 3 — Run Unit Tests

## Run All Tests

```powershell
python -m pytest tests/ -v
```

## Run with Coverage Report

```powershell
python -m pytest tests/ -v --tb=short --cov=src
```

## Run Single Test Class

```powershell
python -m pytest tests/ -v -k "TestCleaning"
```

---

# 📘 What Each Notebook Does

| Notebook | Purpose |
|---|---|
| `01_data_understanding` | Load raw CSV, inspect shape, datatypes, missing values, churn distribution, correlation matrix. Run this first. |
| `02_data_cleaning` | Fix `TotalCharges`, impute median values, remove duplicates, strip whitespace. Includes before vs after comparison. |
| `03_feature_engineering` | Binary encoding, One-Hot Encoding (OHE), tenure groups, RevenueCategory, AnnualRevenue, LoyaltyScore creation. |
| `04_visualization` | Business KPI charts, churn analysis, revenue-at-risk, ML performance, offer distribution, final dashboard summary. |

---

# 📋 What Each Test Class Covers (9 Classes, 35 Tests)

| Test Class | Tests | Validation |
|---|---|---|
| `TestExtract` | 5 | Column presence, uniqueness validation, correct dataset shape |
| `TestCleaning` | 6 | TotalCharges correction, null handling, duplicate removal, whitespace cleanup |
| `TestTransform` | 10 | Churn encoding, OHE validation, Contract/Internet/Payment preservation |
| `TestFeatureEngineering` | 7 | New columns, AnnualRevenue calculation, LoyaltyScore range validation |
| `TestValidation` | 3 | Pipeline validation logic, datatype and null checks |
| `TestPredict` | 3 | Probability range `[0,1]`, output format, model validation |
| `TestRecommend` | 7 | Offer recommendation rules and required fields |
| `TestAuth` | 4 | Password hashing, verification, admin CRUD operations |
| `TestConfig` | 3 | Import validation, string path validation, sorted tenure bins |

---

## Project Structure

```
ETL_project/
├── app.py                  ← Streamlit dashboard
├── main.py                 ← ETL pipeline runner
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py           ← DB & path configuration
│   ├── extract.py          ← Phase 1: CSV extraction
│   ├── transform.py        ← Phase 2: Cleaning + features
│   ├── load.py             ← Phase 3: PostgreSQL loading
│   ├── spark_etl.py        ← PySpark pipeline (scalable)
│   └── utils.py            ← Shared helpers
│
├── sql/
│   ├── schema.sql          ← Table definitions
│   └── queries.sql         ← Analytical queries
│
├── data/
│   ├── raw/                ← Place dataset here
│   ├── processed/          ← Auto-generated cleaned CSV
│   └── backup/             ← Auto-generated backup
│
│── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_visualization.ipynb
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_etl.py
```

---

## ETL Workflow

```
CSV Dataset
    ↓  extract.py      → validate + backup
    ↓  transform.py    → clean + encode + feature engineer
    ↓  load.py         → PostgreSQL (2 tables)
    ↓  spark_etl.py    → distributed processing (optional)
    ↓  app.py          → Streamlit dashboard
```

---

## Dashboard Pages

| Section | Content |
|---|---|
| KPI Row | Total customers, Churn count, Churn %, Revenue lost, Avg tenure |
| Row 1 | Churn pie chart + Contract bar chart |
| Row 2 | Internet service + Payment method breakdown |
| Row 3 | Monthly charges boxplot + Tenure group trend |
| Row 4 | Revenue at risk by segment |
| Row 5 | Interactive data explorer with filters |

---

## Key Insights (Academic)

- Month-to-month contract customers churn significantly more
- Electronic check users show the highest churn rate
- Fiber optic internet users churn more than DSL users
- High monthly charges correlate with higher churn probability
- Longer-tenure customers are far more loyal
