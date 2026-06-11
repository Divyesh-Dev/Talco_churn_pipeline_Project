# ============================================================
# tests/test_etl.py — Unit Tests for the Churn ETL + ML Project
#
# Run all tests:    python -m pytest tests/ -v
# Run one file:     python -m pytest tests/test_etl.py -v
# Run with report:  python -m pytest tests/ -v --tb=short
# ============================================================

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

# ── Project root on path ─────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


# ════════════════════════════════════════════════════════════
# Fixtures — shared test data
# ════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def sample_raw_df():
    """Minimal raw DataFrame that mimics Telco CSV structure."""
    data = {
        "customerID":      ["C001","C002","C003","C004","C005"],
        "gender":          ["Male","Female","Male","Female","Male"],
        "SeniorCitizen":   [0, 0, 1, 0, 1],
        "Partner":         ["Yes","No","No","Yes","No"],
        "Dependents":      ["No","No","Yes","No","Yes"],
        "tenure":          [1, 24, 48, 12, 6],
        "PhoneService":    ["Yes","Yes","No","Yes","Yes"],
        "MultipleLines":   ["No","Yes","No phone service","No","Yes"],
        "InternetService": ["DSL","Fiber optic","DSL","No","Fiber optic"],
        "OnlineSecurity":  ["No","Yes","No","No internet service","No"],
        "OnlineBackup":    ["Yes","No","Yes","No internet service","No"],
        "DeviceProtection":["No","Yes","No","No internet service","Yes"],
        "TechSupport":     ["No","No","Yes","No internet service","No"],
        "StreamingTV":     ["No","Yes","No","No internet service","No"],
        "StreamingMovies": ["No","Yes","No","No internet service","Yes"],
        "Contract":        ["Month-to-month","One year","Two year","Month-to-month","Month-to-month"],
        "PaperlessBilling":["Yes","No","No","Yes","Yes"],
        "PaymentMethod":   ["Electronic check","Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)","Electronic check"],
        "MonthlyCharges":  [29.85, 56.95, 53.85, 42.30, 89.10],
        "TotalCharges":    ["29.85","1889.50","2701.11","  ","534.60"],
        "Churn":           ["No","No","No","No","Yes"],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def cleaned_df(sample_raw_df):
    from src.transform import clean_data
    return clean_data(sample_raw_df.copy())


@pytest.fixture(scope="session")
def transformed_df(cleaned_df):
    from src.transform import transform_data
    return transform_data(cleaned_df.copy())


@pytest.fixture(scope="session")
def featured_df(transformed_df):
    from src.transform import engineer_features
    return engineer_features(transformed_df.copy())


# ════════════════════════════════════════════════════════════
# 1. EXTRACT TESTS
# ════════════════════════════════════════════════════════════

class TestExtract:

    def test_validate_columns_pass(self, sample_raw_df):
        """All 21 required columns should be present."""
        from src.utils import validate_columns
        required = [
            "customerID","gender","SeniorCitizen","Partner","Dependents",
            "tenure","PhoneService","MultipleLines","InternetService",
            "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
            "StreamingTV","StreamingMovies","Contract","PaperlessBilling",
            "PaymentMethod","MonthlyCharges","TotalCharges","Churn",
        ]
        # Should not raise
        validate_columns(sample_raw_df, required)

    def test_validate_columns_fail(self, sample_raw_df):
        """Missing column should raise ValueError."""
        from src.utils import validate_columns
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_columns(sample_raw_df, ["customerID", "NONEXISTENT_COL"])

    def test_dataframe_not_empty(self, sample_raw_df):
        assert len(sample_raw_df) > 0, "Dataset should not be empty"

    def test_correct_shape(self, sample_raw_df):
        assert sample_raw_df.shape[1] == 21, "Raw data should have 21 columns"

    def test_customer_id_unique(self, sample_raw_df):
        assert sample_raw_df["customerID"].nunique() == len(sample_raw_df), \
            "customerID should be unique"


# ════════════════════════════════════════════════════════════
# 2. CLEAN TESTS
# ════════════════════════════════════════════════════════════

class TestCleaning:

    def test_total_charges_numeric(self, cleaned_df):
        """TotalCharges must be float after cleaning."""
        assert pd.api.types.is_float_dtype(cleaned_df["TotalCharges"]), \
            "TotalCharges should be float64"

    def test_no_null_total_charges(self, cleaned_df):
        """No nulls in TotalCharges after imputation."""
        assert cleaned_df["TotalCharges"].isnull().sum() == 0, \
            "TotalCharges should have no nulls after cleaning"

    def test_blank_total_charges_fixed(self, sample_raw_df):
        """Blank TotalCharges entries (new customers) should be imputed."""
        from src.transform import clean_data
        df = clean_data(sample_raw_df.copy())
        blank_before = (sample_raw_df["TotalCharges"].str.strip() == "").sum()
        assert blank_before > 0, "Test data should have blank TotalCharges"
        assert df["TotalCharges"].isnull().sum() == 0, "Should be imputed after cleaning"

    def test_no_duplicates(self, cleaned_df):
        """Cleaning should remove all duplicate rows."""
        assert cleaned_df.duplicated().sum() == 0, "No duplicates should remain"

    def test_strings_stripped(self, cleaned_df):
        """String columns should have no leading/trailing whitespace."""
        str_cols = cleaned_df.select_dtypes("object").columns
        for col in str_cols:
            has_space = cleaned_df[col].str.startswith(" ").any() or \
                        cleaned_df[col].str.endswith(" ").any()
            assert not has_space, f"Column '{col}' has unstripped whitespace"

    def test_row_count_preserved(self, sample_raw_df, cleaned_df):
        """Cleaning should not drop rows (no dups in test data)."""
        assert len(cleaned_df) == len(sample_raw_df)


# ════════════════════════════════════════════════════════════
# 3. TRANSFORM TESTS
# ════════════════════════════════════════════════════════════

class TestTransform:

    def test_churn_binary_encoded(self, transformed_df):
        """Churn should be 0 or 1 only."""
        unique_vals = set(transformed_df["Churn"].unique())
        assert unique_vals <= {0, 1}, f"Churn should be binary, got: {unique_vals}"

    def test_churn_dtype_int(self, transformed_df):
        assert pd.api.types.is_integer_dtype(transformed_df["Churn"]), \
            "Churn should be integer type"

    def test_gender_encoded_created(self, transformed_df):
        assert "gender_encoded" in transformed_df.columns

    def test_gender_encoded_binary(self, transformed_df):
        unique_vals = set(transformed_df["gender_encoded"].unique())
        assert unique_vals <= {0, 1}

    def test_encoded_columns_exist(self, transformed_df):
        expected = ["Partner_encoded","Dependents_encoded",
                    "PhoneService_encoded","PaperlessBilling_encoded"]
        for col in expected:
            assert col in transformed_df.columns, f"Missing column: {col}"

    def test_tenure_group_created(self, transformed_df):
        assert "tenure_group" in transformed_df.columns

    def test_tenure_group_values(self, transformed_df):
        valid = {"0-1 Year","1-2 Years","2-4 Years","4-6 Years"}
        actual = set(transformed_df["tenure_group"].dropna().unique())
        assert actual <= valid, f"Unexpected tenure_group values: {actual - valid}"

    def test_contract_column_preserved(self, transformed_df):
        """Contract label column must survive OHE for dashboard filtering."""
        assert "Contract" in transformed_df.columns, \
            "Contract column must be preserved after OHE"

    def test_internet_service_preserved(self, transformed_df):
        assert "InternetService" in transformed_df.columns

    def test_payment_method_preserved(self, transformed_df):
        assert "PaymentMethod" in transformed_df.columns


# ════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING TESTS
# ════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    def test_revenue_category_created(self, featured_df):
        assert "RevenueCategory" in featured_df.columns

    def test_revenue_category_values(self, featured_df):
        valid = {"Low","Medium","High"}
        actual = set(featured_df["RevenueCategory"].unique())
        assert actual <= valid, f"Unexpected RevenueCategory values: {actual}"

    def test_annual_revenue_created(self, featured_df):
        assert "AnnualRevenue" in featured_df.columns

    def test_annual_revenue_equals_12x_monthly(self, featured_df):
        """AnnualRevenue = MonthlyCharges * 12."""
        expected = (featured_df["MonthlyCharges"] * 12).round(2)
        actual   = featured_df["AnnualRevenue"].round(2)
        pd.testing.assert_series_equal(actual, expected,
                                        check_names=False, check_dtype=False)

    def test_loyalty_score_created(self, featured_df):
        assert "LoyaltyScore" in featured_df.columns

    def test_loyalty_score_range(self, featured_df):
        """LoyaltyScore must be between 0 and 1."""
        assert featured_df["LoyaltyScore"].between(0, 1).all(), \
            "LoyaltyScore must be in [0, 1]"

    def test_no_nulls_after_features(self, featured_df):
        critical = ["RevenueCategory","AnnualRevenue","LoyaltyScore","tenure_group"]
        for col in critical:
            if col in featured_df.columns:
                assert featured_df[col].isnull().sum() == 0, \
                    f"{col} should have no nulls after feature engineering"


# ════════════════════════════════════════════════════════════
# 5. VALIDATION TESTS
# ════════════════════════════════════════════════════════════

class TestValidation:

    def test_validate_passes_on_clean_data(self, featured_df):
        from src.transform import validate_data
        result = validate_data(featured_df)
        assert result is True, "Validation should pass on fully cleaned data"

    def test_validate_fails_on_nulls(self):
        from src.transform import validate_data
        bad_df = pd.DataFrame({"Churn": [0, 1, None], "col": [1.0, 2.0, None]})
        result = validate_data(bad_df)
        assert result is False, "Validation should fail when nulls are present"

    def test_validate_fails_on_non_integer_churn(self):
        from src.transform import validate_data
        bad_df = pd.DataFrame({"Churn": ["Yes","No","Yes"]})
        result = validate_data(bad_df)
        assert result is False


# ════════════════════════════════════════════════════════════
# 6. ML — PREDICT TESTS
# ════════════════════════════════════════════════════════════

class TestPredict:
    """Tests run only if the trained model file exists."""

    MODEL_PATH = os.path.join(_ROOT, "models", "churn_model.pkl")

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_ROOT, "models", "churn_model.pkl")),
        reason="Model not trained yet — run: python main.py --no-load"
    )
    def test_predict_single_returns_dict(self):
        from src.predict import predict_single
        customer = {
            "SeniorCitizen":1,"tenure":6,"MonthlyCharges":90.0,
            "TotalCharges":540.0,"gender_encoded":1,
            "Partner_encoded":0,"Dependents_encoded":0,
            "PhoneService_encoded":1,"PaperlessBilling_encoded":1,
            "AnnualRevenue":1080.0,"LoyaltyScore":0.083
        }
        result = predict_single(customer)
        assert isinstance(result, dict)
        assert "probability" in result
        assert "risk" in result
        assert "will_churn" in result

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_ROOT, "models", "churn_model.pkl")),
        reason="Model not trained yet"
    )
    def test_predict_probability_range(self):
        from src.predict import predict_single
        customer = {
            "SeniorCitizen":0,"tenure":48,"MonthlyCharges":45.0,
            "TotalCharges":2160.0,"gender_encoded":0,
            "Partner_encoded":1,"Dependents_encoded":1,
            "PhoneService_encoded":1,"PaperlessBilling_encoded":0,
            "AnnualRevenue":540.0,"LoyaltyScore":0.67
        }
        result = predict_single(customer)
        assert 0.0 <= result["probability"] <= 1.0, \
            "Probability must be between 0 and 1"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_ROOT, "models", "churn_model.pkl")),
        reason="Model not trained yet"
    )
    def test_predict_risk_labels(self):
        from src.predict import predict_single
        customer = {"SeniorCitizen":0,"tenure":1,"MonthlyCharges":99.0,
                    "TotalCharges":99.0,"gender_encoded":1,
                    "Partner_encoded":0,"Dependents_encoded":0,
                    "PhoneService_encoded":1,"PaperlessBilling_encoded":1,
                    "AnnualRevenue":1188.0,"LoyaltyScore":0.01}
        result = predict_single(customer)
        assert result["risk"] in {"Low","Medium","High"}, \
            f"Invalid risk label: {result['risk']}"


# ════════════════════════════════════════════════════════════
# 7. RECOMMEND TESTS
# ════════════════════════════════════════════════════════════

class TestRecommend:

    def test_high_risk_month_to_month_gets_contract_offer(self):
        from src.recommend import recommend_offer
        row = pd.Series({
            "ChurnProbability": 0.72, "ChurnRisk": "High",
            "Contract": "Month-to-month", "InternetService": "DSL",
            "PaymentMethod": "Mailed check", "SeniorCitizen": 0,
            "tenure": 5, "MonthlyCharges": 55.0
        })
        offer = recommend_offer(row)
        assert offer["id"] == "CONTRACT_UPGRADE"

    def test_very_high_risk_gets_retention_call(self):
        from src.recommend import recommend_offer
        row = pd.Series({
            "ChurnProbability": 0.92, "ChurnRisk": "High",
            "Contract": "Month-to-month", "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check", "SeniorCitizen": 0,
            "tenure": 2, "MonthlyCharges": 95.0
        })
        offer = recommend_offer(row)
        assert offer["id"] == "RETENTION_CALL"

    def test_senior_citizen_gets_senior_plan(self):
        from src.recommend import recommend_offer
        row = pd.Series({
            "ChurnProbability": 0.68, "ChurnRisk": "High",
            "Contract": "One year", "InternetService": "DSL",
            "PaymentMethod": "Mailed check", "SeniorCitizen": 1,
            "tenure": 20, "MonthlyCharges": 40.0
        })
        offer = recommend_offer(row)
        assert offer["id"] == "SENIOR_PLAN"

    def test_electronic_check_gets_autopay_cashback(self):
        from src.recommend import recommend_offer
        row = pd.Series({
            "ChurnProbability": 0.55, "ChurnRisk": "Medium",
            "Contract": "One year", "InternetService": "DSL",
            "PaymentMethod": "Electronic check", "SeniorCitizen": 0,
            "tenure": 18, "MonthlyCharges": 50.0
        })
        offer = recommend_offer(row)
        assert offer["id"] == "AUTOPAY_CASHBACK"

    def test_offer_has_required_fields(self):
        from src.recommend import recommend_offer
        row = pd.Series({
            "ChurnProbability": 0.30, "ChurnRisk": "Low",
            "Contract": "Two year", "InternetService": "DSL",
            "PaymentMethod": "Credit card (automatic)", "SeniorCitizen": 0,
            "tenure": 60, "MonthlyCharges": 35.0
        })
        offer = recommend_offer(row)
        for field in ["id","title","description","discount","action","priority"]:
            assert field in offer, f"Offer missing field: {field}"

    def test_generate_recommendations_adds_columns(self, featured_df):
        from src.recommend import generate_recommendations
        # Add dummy churn columns
        df = featured_df.copy()
        df["ChurnProbability"] = 0.5
        df["ChurnRisk"] = "Medium"
        result = generate_recommendations(df)
        for col in ["OfferID","OfferTitle","OfferDescription","OfferDiscount","OfferAction"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_generate_recommendations_row_count(self, featured_df):
        from src.recommend import generate_recommendations
        df = featured_df.copy()
        df["ChurnProbability"] = 0.5
        df["ChurnRisk"] = "Medium"
        result = generate_recommendations(df)
        assert len(result) == len(df), "Row count should not change"


# ════════════════════════════════════════════════════════════
# 8. AUTH TESTS
# ════════════════════════════════════════════════════════════

class TestAuth:

    def test_hash_and_verify_password(self):
        """Hashed password should verify correctly."""
        import hashlib, os as _os
        def hash_pw(pw):
            salt = _os.urandom(32).hex()
            key  = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 260000)
            return f"{salt}:{key.hex()}"
        def verify_pw(pw, stored):
            salt, key_hex = stored.split(":", 1)
            key = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 260000)
            return key.hex() == key_hex

        hashed = hash_pw("admin123")
        assert verify_pw("admin123", hashed) is True
        assert verify_pw("wrongpass", hashed) is False

    def test_users_json_exists(self):
        users_path = os.path.join(_ROOT, "users.json")
        assert os.path.exists(users_path), \
            "users.json not found — run main.py to initialise"

    def test_admin_user_exists(self):
        users_path = os.path.join(_ROOT, "users.json")
        if os.path.exists(users_path):
            with open(users_path) as f:
                users = json.load(f)
            assert "Div-dev" in users, "Default admin user 'Div-dev' should exist"
            assert users["Div-dev"]["role"] == "admin"

    def test_add_and_delete_user(self):
        from src.auth import add_user, delete_user, _load_users, _save_users
        # Add test user
        ok, msg = add_user("testuser_pytest", "test1234", "user", "Test User", "Div-dev")
        assert ok is True, f"add_user failed: {msg}"
        # Verify exists
        users = _load_users()
        assert "testuser_pytest" in users
        # Delete test user
        ok, msg = delete_user("testuser_pytest", "Div-dev")
        assert ok is True, f"delete_user failed: {msg}"
        # Verify removed
        users = _load_users()
        assert "testuser_pytest" not in users


# ════════════════════════════════════════════════════════════
# 9. CONFIG TESTS
# ════════════════════════════════════════════════════════════

class TestConfig:

    def test_config_imports(self):
        from src.config import (RAW_DATA, PROCESSED_DATA, BACKUP_DATA,
                                 DB_URL, TABLE_CLEANED, CHUNK_SIZE,
                                 TENURE_BINS, TENURE_LABELS)
        assert isinstance(RAW_DATA, str)
        assert isinstance(DB_URL, str)
        assert len(TENURE_BINS) == 5
        assert len(TENURE_LABELS) == 4

    def test_tenure_bins_sorted(self):
        from src.config import TENURE_BINS
        assert TENURE_BINS == sorted(TENURE_BINS), "TENURE_BINS must be sorted ascending"

    def test_paths_are_strings(self):
        from src.config import RAW_DATA, PROCESSED_DATA, BACKUP_DATA
        for path in [RAW_DATA, PROCESSED_DATA, BACKUP_DATA]:
            assert isinstance(path, str) and len(path) > 0
