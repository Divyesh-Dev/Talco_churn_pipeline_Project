# ============================================================
# train_model.py — Train churn prediction ML model
# Run once:  python -m src.train_model
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .config import PROCESSED_DATA
from .utils import log

# ── Paths ─────────────────────────────────────────────────────
_SRC  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
MODELS_DIR      = os.path.join(_ROOT, "models")
MODEL_PATH      = os.path.join(MODELS_DIR, "churn_model.pkl")
FEATURES_PATH   = os.path.join(MODELS_DIR, "feature_names.pkl")
METRICS_PATH    = os.path.join(MODELS_DIR, "model_metrics.json")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")

# ── Features used for training ────────────────────────────────
# These are the numeric / encoded columns available after transform
FEATURE_COLS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "gender_encoded",
    "Partner_encoded", "Dependents_encoded",
    "PhoneService_encoded", "PaperlessBilling_encoded",
    "AnnualRevenue", "LoyaltyScore",
]

TARGET_COL = "Churn"


def load_training_data() -> pd.DataFrame:
    if not os.path.exists(PROCESSED_DATA):
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA}.\n"
            "Run the ETL pipeline first: python main.py --no-load"
        )
    df = pd.read_csv(PROCESSED_DATA)
    log("TRAIN", f"Loaded training data: {df.shape[0]:,} rows")
    return df


def prepare_features(df: pd.DataFrame):
    """Extract feature matrix X and target vector y."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log("TRAIN", f"⚠ Missing feature cols (will skip): {missing}")

    X = df[available].copy()
    y = df[TARGET_COL].astype(int)

    # Drop any remaining nulls
    mask = X.notnull().all(axis=1) & y.notnull()
    X, y = X[mask], y[mask]

    log("TRAIN", f"Feature matrix: {X.shape}  |  Churn rate: {y.mean():.2%}")
    return X, y, available


def train(X, y):
    """Train three models, pick best by ROC-AUC, return all metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            random_state=42
        ),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced",
                random_state=42, C=0.5
            ))
        ]),
    }

    results = {}
    best_name, best_model, best_auc = None, None, 0.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba),  4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        results[name] = metrics

        log("TRAIN", (
            f"{name}: Acc={metrics['accuracy']:.3f}  "
            f"F1={metrics['f1']:.3f}  AUC={metrics['roc_auc']:.3f}"
        ))

        if metrics["roc_auc"] > best_auc:
            best_auc   = metrics["roc_auc"]
            best_name  = name
            best_model = model

    log("TRAIN", f"Best model → {best_name} (AUC={best_auc:.3f}) ✓")
    return best_model, best_name, results, X_test, y_test


def get_feature_importance(model, feature_names: list) -> dict:
    """Extract feature importances (works for tree models)."""
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf")
            if hasattr(clf, "coef_"):
                imp = abs(clf.coef_[0])
            else:
                return {}
        else:
            return {}

        importance = dict(zip(feature_names, [round(float(v), 4) for v in imp]))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        return {}


def save_artifacts(model, feature_names: list, metrics: dict,
                   best_name: str, importance: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,        MODEL_PATH)
    joblib.dump(feature_names, FEATURES_PATH)

    full_metrics = {
        "best_model":        best_name,
        "feature_importance": importance,
        "model_results":     metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(full_metrics, f, indent=2)

    log("TRAIN", f"Model saved      → {MODEL_PATH}")
    log("TRAIN", f"Feature names    → {FEATURES_PATH}")
    log("TRAIN", f"Metrics saved    → {METRICS_PATH}")


def run_training():
    log("TRAIN", "=" * 50)
    log("TRAIN", "  Churn Prediction Model Training")
    log("TRAIN", "=" * 50)

    df             = load_training_data()
    X, y, features = prepare_features(df)
    model, name, all_metrics, X_test, y_test = train(X, y)
    importance     = get_feature_importance(model, features)

    log("TRAIN", "\nFeature Importances:")
    for feat, val in list(importance.items())[:8]:
        bar = "█" * int(val * 40)
        log("TRAIN", f"  {feat:<30} {val:.4f}  {bar}")

    save_artifacts(model, features, all_metrics, name, importance)

    log("TRAIN", "=" * 50)
    log("TRAIN", "  Training Complete ✓")
    log("TRAIN", "=" * 50)
    return model, features


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_training()
