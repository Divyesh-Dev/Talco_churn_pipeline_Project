# ============================================================
# spark_etl.py — PySpark-based ETL (scalable / enterprise layer)
# ============================================================
# This mirrors the Pandas pipeline but uses PySpark for distributed
# processing. Run independently or call run_spark_pipeline() from main.py.
# ============================================================

from src.config import RAW_DATA, PYSPARK_MASTER
from src.utils import log
import os


def get_spark():
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .master(PYSPARK_MASTER)
        .appName("CustomerChurnETL")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log("PYSPARK", f"SparkSession started — master: {PYSPARK_MASTER}")
    return spark


def spark_extract(spark, filepath: str = RAW_DATA):
    log("PYSPARK", f"Reading: {filepath}")
    sdf = spark.read.csv(filepath, header=True, inferSchema=True)
    log("PYSPARK", f"Schema:")
    sdf.printSchema()
    log("PYSPARK", f"Row count: {sdf.count():,}")
    return sdf


def spark_transform(sdf):
    from pyspark.sql.functions import col, when, pandas_udf
    import pyspark.sql.functions as F

    log("PYSPARK", "Transforming …")

    # Fix TotalCharges (spaces → 0.0)
    sdf = sdf.withColumn(
        "TotalCharges",
        F.when(F.trim(col("TotalCharges")) == "", 0.0)
         .otherwise(col("TotalCharges").cast("double"))
    )

    # Encode Churn
    sdf = sdf.withColumn(
        "ChurnFlag",
        when(col("Churn") == "Yes", 1).otherwise(0)
    )

    # Encode gender
    sdf = sdf.withColumn(
        "gender_encoded",
        when(col("gender") == "Male", 1).otherwise(0)
    )

    # Revenue category
    sdf = sdf.withColumn(
        "RevenueCategory",
        when(col("MonthlyCharges") < 35, "Low")
        .when(col("MonthlyCharges") < 70, "Medium")
        .otherwise("High")
    )

    # Tenure group
    sdf = sdf.withColumn(
        "tenure_group",
        when(col("tenure") <= 12, "0-1 Year")
        .when(col("tenure") <= 24, "1-2 Years")
        .when(col("tenure") <= 48, "2-4 Years")
        .otherwise("4-6 Years")
    )

    log("PYSPARK", "Transformation complete ✓")
    return sdf


def spark_save(sdf, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    sdf.write.csv(output_path, header=True, mode="overwrite")
    log("PYSPARK", f"Saved → {output_path}")


def run_spark_pipeline():
    """
    Full PySpark ETL:  CSV → transform → save as CSV partitions.
    """
    spark = get_spark()
    sdf = spark_extract(spark)
    sdf = spark_transform(sdf)
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed", "spark_output"
    )
    spark_save(sdf, out_path)
    spark.stop()
    log("PYSPARK", "Pipeline finished ✓")
