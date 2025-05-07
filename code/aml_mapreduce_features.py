# aml_mapreduce_features.py
# PySpark MapReduce-style feature engineering for AML dataset

from pyspark.sql import SparkSession

# ----------------------------------------------------------------------------
# 1. Spark Session Setup
# ----------------------------------------------------------------------------
spark = (
    SparkSession.builder
        .appName("AML MapReduce Feature Engineering")
        .master("local[*]")
        .config("spark.hadoop.hadoop.security.authentication", "simple")
        .config("spark.hadoop.hadoop.security.authorization", "false")
        .getOrCreate()
)

# ----------------------------------------------------------------------------
# 2. Load AML Transaction Data
# ----------------------------------------------------------------------------
df_raw = spark.read.csv("code/HI-Small_Trans.csv", header=True, inferSchema=True)

df = df_raw.selectExpr(
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
    "`Timestamp` as Timestamp"
).filter("From_Account IS NOT NULL AND Amount_Paid IS NOT NULL")


# ----------------------------------------------------------------------------
# 3. RDD MapReduce: Total, Count, Average per From_Account
# ----------------------------------------------------------------------------
rdd = df.rdd.map(lambda row: (row["From_Account"], (row["Amount_Paid"], 1)))

# Reduce: sum amounts and counts
reduced = rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))

# Final map: (Account, Total, Count, Average)
final_rdd = reduced.map(lambda kv: (kv[0], kv[1][0], kv[1][1], kv[1][0] / kv[1][1]))

# ----------------------------------------------------------------------------
# 4. Convert to DataFrame and Save
# ----------------------------------------------------------------------------
schema = "Account STRING, TotalAmount DOUBLE, TxnCount INT, AvgAmount DOUBLE"
result_df = spark.createDataFrame(final_rdd, schema)

# Show results for debug
result_df.show(20, truncate=False)

# Save for ML (can upload to Snowflake or use in pandas)
result_df.coalesce(1).write.option("header", True).csv("code/hi_small_features", mode="overwrite")
print("✅ Feature engineering completed and saved to 'code/hi_small_features'.")

# ----------------------------------------------------------------------------
# 5. Keep Spark UI Alive (Optional for DAG screenshot)
# ----------------------------------------------------------------------------
ui_url = spark.sparkContext.uiWebUrl or "http://localhost:4040"
print(f"\n✅ Job finished. Spark UI is live at {ui_url}")
input("🔹 Press ENTER to stop Spark and exit…")
print("🛑 Shutting down Spark …")
spark.stop()