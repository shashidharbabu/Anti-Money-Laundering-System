import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, sum, max, min,
    hour, countDistinct, expr, to_timestamp
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
INPUT_CSV      = "code/HI-Small_Trans.csv"
PATTERNS_CSV   = "code/laundering_transactions.csv"
OUTPUT_DIR     = "code/ml_features"
TIMESTAMP_FMT  = "yyyy/MM/dd HH:mm:ss"  # adjust if needed

# ──────────────────────────────────────────────────────────────────────────────
# Initialize Spark
# ──────────────────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("AML Feature Engineering") \
    .getOrCreate()

# ──────────────────────────────────────────────────────────────────────────────
# 1) Load & clean raw transactions
# ──────────────────────────────────────────────────────────────────────────────
df_raw = spark.read.csv(INPUT_CSV, header=True, inferSchema=True)

df = (
    df_raw
    .selectExpr(
        "`From Bank` as From_Bank",
        "Account2 as From_Account",
        "`To Bank` as To_Bank",
        "Account4 as To_Account",
        "`Amount Received` as Amount_Received",
        "`Receiving Currency` as Receiving_Currency",
        "`Amount Paid` as Amount_Paid",
        "`Payment Currency` as Payment_Currency",
        "`Payment Format` as Payment_Format",
        "`Is Laundering` as Is_Laundering",
        "`Timestamp` as Timestamp_str"
    )
    .dropna(subset=["From_Account", "Amount_Paid", "Timestamp_str"])
    .withColumn("Timestamp", to_timestamp("Timestamp_str", TIMESTAMP_FMT))
    .drop("Timestamp_str")
    .withColumn("Hour", hour("Timestamp"))
)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Feature Engineering (per account)
# ──────────────────────────────────────────────────────────────────────────────
features = (
    df
    .groupBy("From_Account")
    .agg(
        count("*").alias("TxnCount"),
        sum("Amount_Paid").alias("TotalAmount"),
        avg("Amount_Paid").alias("AvgAmount"),
        max("Amount_Paid").alias("MaxAmount"),
        min("Amount_Paid").alias("MinAmount"),
        countDistinct("Payment_Currency").alias("NumCurrencies"),
        countDistinct("Payment_Format").alias("NumFormats"),
        expr("percentile_approx(Hour, 0.5)").alias("HourBucket")
    )
)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Load laundering pattern labels
# ──────────────────────────────────────────────────────────────────────────────
pattern_df = (
    spark.read
    .csv(PATTERNS_CSV, header=True, inferSchema=True)
    .select(col("From_Bank").alias("From_Account"),
            col("isLaundering").cast("int").alias("IsLaundering"))
)

labels = (
    pattern_df
    .groupBy("From_Account")
    .agg(max("IsLaundering").alias("IsLaundering"))
)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Join features with labels
# ──────────────────────────────────────────────────────────────────────────────
final_df = (
    features
    .join(labels, on="From_Account", how="left")
    .fillna({"IsLaundering": 0})
)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Save to ML-ready CSV
# ──────────────────────────────────────────────────────────────────────────────
final_df.coalesce(1) \
    .write.option("header", True).mode("overwrite") \
    .csv(OUTPUT_DIR)

print(f"\n✅ Feature engineering complete. ML dataset at {OUTPUT_DIR}")

# ──────────────────────────────────────────────────────────────────────────────
# Optional: Keep Spark UI alive for DAG screenshot
# ──────────────────────────────────────────────────────────────────────────────
ui_url = spark.sparkContext.uiWebUrl or "http://localhost:4040"
print(f"\n▶ Spark Web UI available at: {ui_url}")
input("🔸 Press ENTER after screenshot to stop Spark…")
spark.stop()
