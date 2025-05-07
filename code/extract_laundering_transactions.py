from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ──────────────────────────────────────────────────────────────────────────────
# 1. Initialize Spark
# ──────────────────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("ExtractLaunderingLabels") \
    .getOrCreate()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Load the patterns text file
# ──────────────────────────────────────────────────────────────────────────────
df = spark.read.text("code/HI-Small_Patterns.txt")

# Step 1: Extract Pattern_Type from BEGIN lines
df = df.withColumn(
    "Pattern_Type",
    F.when(
        F.col("value").rlike("BEGIN LAUNDERING ATTEMPT - (.+)"),
        F.regexp_extract("value", "BEGIN LAUNDERING ATTEMPT - (.+)", 1)
    )
)

# Step 2: Forward fill Pattern_Type across the laundering block
window_spec = Window.orderBy(F.monotonically_increasing_id()).rowsBetween(Window.unboundedPreceding, 0)
df = df.withColumn("Pattern_Type", F.last("Pattern_Type", ignorenulls=True).over(window_spec))

# Step 3: Filter out END lines
df = df.filter(~F.col("value").contains("END LAUNDERING ATTEMPT"))

# Step 4: Filter actual transaction lines using a timestamp regex
transactions = df.filter(F.col("value").rlike(r"\d{4}/\d{2}/\d{2}"))

# Step 5: Split value column into fields
columns = [
    "Timestamp", "From_Bank", "From_Account", "To_Bank", "To_Account",
    "Amount_Received", "Receiving_Currency", "Amount_Paid",
    "Payment_Currency", "Payment_Format"
]

for i, col_name in enumerate(columns):
    transactions = transactions.withColumn(col_name, F.split("value", ",").getItem(i))

# Step 6: Add label
transactions = transactions.withColumn("isLaundering", F.lit(1).cast("int"))

# Step 7: Select useful columns
result_df = transactions.select("Timestamp", "From_Bank", "Pattern_Type", "isLaundering")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Save to CSV
# ──────────────────────────────────────────────────────────────────────────────
result_df.coalesce(1).write.option("header", True).mode("overwrite").csv("code/laundering_transactions_temp")

# Rename output file for consistency
import os, shutil

output_dir = "code/laundering_transactions_temp"
part_file = [f for f in os.listdir(output_dir) if f.startswith("part")][0]
shutil.move(f"{output_dir}/{part_file}", "code/laundering_transactions.csv")
shutil.rmtree(output_dir)

print("\n✅ laundering_transactions.csv has been created in the code/ directory.")
spark.stop()
