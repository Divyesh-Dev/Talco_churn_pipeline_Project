-- ============================================================
-- queries.sql — Analytical queries for the Streamlit dashboard
-- ============================================================

-- 1. Overall KPIs
SELECT
    COUNT(*)                                   AS total_customers,
    SUM(Churn)                                 AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)   AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2)              AS total_monthly_revenue,
    ROUND(AVG(MonthlyCharges), 2)             AS avg_monthly_charge,
    ROUND(AVG(tenure), 1)                     AS avg_tenure_months
FROM customer_churn_cleaned;

-- 2. Churn by Contract Type
SELECT
    Contract,
    COUNT(*)                                   AS total,
    SUM(Churn)                                 AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)   AS churn_rate_pct
FROM customer_churn_cleaned
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 3. Churn by Internet Service
SELECT
    InternetService,
    COUNT(*)                                   AS total,
    SUM(Churn)                                 AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)   AS churn_rate_pct
FROM customer_churn_cleaned
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;

-- 4. Churn by Payment Method
SELECT
    PaymentMethod,
    COUNT(*)                                   AS total,
    SUM(Churn)                                 AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)   AS churn_rate_pct
FROM customer_churn_cleaned
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- 5. Revenue at risk (churned customers)
SELECT
    ROUND(SUM(MonthlyCharges), 2)   AS lost_monthly_revenue,
    ROUND(SUM(AnnualRevenue), 2)    AS lost_annual_revenue
FROM customer_churn_cleaned
WHERE Churn = 1;

-- 6. Churn by Tenure Group
SELECT
    tenure_group,
    COUNT(*)                                   AS total,
    SUM(Churn)                                 AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)   AS churn_rate_pct
FROM customer_churn_cleaned
GROUP BY tenure_group
ORDER BY tenure_group;

-- 7. Churn by Revenue Category
SELECT
    RevenueCategory,
    COUNT(*)                                   AS total,
    SUM(Churn)                                 AS churned,
    ROUND(SUM(Churn) * 100.0 / COUNT(*), 2)   AS churn_rate_pct
FROM customer_churn_cleaned
GROUP BY RevenueCategory
ORDER BY churn_rate_pct DESC;

-- 8. High-risk customers (month-to-month + high charges + churned)
SELECT customerID, MonthlyCharges, TotalCharges, tenure
FROM customer_churn_cleaned
WHERE Contract = 'Month-to-month'
  AND MonthlyCharges > 70
  AND Churn = 1
ORDER BY MonthlyCharges DESC
LIMIT 20;
