# ============================================================
# main.py — Master ETL + ML pipeline runner
# Usage:
#   python main.py              → full ETL + ML pipeline
#   python main.py --no-load    → skip DB load (CSV output only)
#   python main.py --no-train   → skip model training (use existing model)
#   python main.py --spark      → also run PySpark pipeline
# ============================================================

import argparse
import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from src.extract    import extract_data
from src.transform  import run_transform
from src.load       import load_data
from src.utils      import log


def parse_args():
    p = argparse.ArgumentParser(description="Telecom Churn ETL + ML Pipeline")
    p.add_argument("--spark",    action="store_true", help="Also run PySpark pipeline")
    p.add_argument("--no-load",  action="store_true", help="Skip database load step")
    p.add_argument("--no-train", action="store_true", help="Skip model training (use saved model)")
    return p.parse_args()


def main():
    args = parse_args()

    log("PIPELINE", "=" * 58)
    log("PIPELINE", "  Telecom Customer Churn ETL + ML — Starting")
    log("PIPELINE", "=" * 58)

    # Phase 1: Extract
    log("PIPELINE", "Phase 1 → Extract")
    df_raw = extract_data()

    # Phase 2: Transform
    log("PIPELINE", "Phase 2 → Transform + Feature Engineering")
    df_clean = run_transform(df_raw)
    log("PIPELINE", f"Transformed shape: {df_clean.shape}")

    # Phase 3: Train ML Model
    if not args.no_train:
        log("PIPELINE", "Phase 3 → Train Churn Prediction Model")
        from src.train_model import run_training
        run_training()
    else:
        log("PIPELINE", "Phase 3 → Training SKIPPED (--no-train flag)")

    # Phase 4: Predict + Recommend
    log("PIPELINE", "Phase 4 → Predict Churn Probabilities + Generate Offers")
    try:
        from src.predict   import predict_churn
        from src.recommend import generate_recommendations

        df_scored = predict_churn(df_clean)
        df_final  = generate_recommendations(df_scored)

        out_path = os.path.join("data", "processed", "churn_scored.csv")
        df_final.to_csv(out_path, index=False)
        log("PIPELINE", f"Scored data saved -> {out_path}")

    except FileNotFoundError as e:
        log("PIPELINE", f"Warning - Skipping prediction: {e}")
        df_final = df_clean

    # Phase 5: Load to DB
    if args.no_load:
        log("PIPELINE", "Phase 5 → Load SKIPPED (--no-load flag)")
    else:
        log("PIPELINE", "Phase 5 → Load to PostgreSQL")
        load_data(df_final, df_raw)

    # Phase 6: PySpark (optional)
    if args.spark:
        log("PIPELINE", "Phase 6 → PySpark Pipeline")
        from src.spark_etl import run_spark_pipeline
        run_spark_pipeline()

    log("PIPELINE", "=" * 58)
    log("PIPELINE", "  Pipeline Complete - run: streamlit run app.py")
    log("PIPELINE", "=" * 58)


if __name__ == "__main__":
    main()
