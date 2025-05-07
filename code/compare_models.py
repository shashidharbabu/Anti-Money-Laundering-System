# compare_models.py
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# XGBoost4J-Spark
from xgboost.spark import SparkXGBClassifier

# ──────────────────────────────────────────────────────────────────────────────
# 1) Start Spark
# ──────────────────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Compare-GBT-vs-XGBoost") \
    .getOrCreate()

# ──────────────────────────────────────────────────────────────────────────────
# 2) Load ML dataset
# ──────────────────────────────────────────────────────────────────────────────
df = spark.read.csv("code/ml_features/final_ml_dataset.csv", header=True, inferSchema=True)

from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

# cast HourBucket (and any other numeric features read as string) to Double
df = df.withColumn("HourBucket", col("HourBucket").cast(DoubleType()))
null_buckets = df.filter(col("HourBucket").isNull()).count()

print("################################################")

print(f"🚩 Null HourBucket rows: {null_buckets} / {df.count()}")
print("################################################")

# numeric_feats = [
#     "TxnCount","TotalAmount","AvgAmount","MaxAmount","MinAmount",
#     "NumCurrencies","NumFormats","HourBucket"
# ]
# for feat in numeric_feats:
#     df = df.withColumn(feat, col(feat).cast(DoubleType()))


# ──────────────────────────────────────────────────────────────────────────────
# 3) Prepare label & features
# ──────────────────────────────────────────────────────────────────────────────
label_indexer = StringIndexer(inputCol="IsLaundering", outputCol="label", handleInvalid="keep")
feature_cols   = [
    "TxnCount", "TotalAmount", "AvgAmount", "MaxAmount", "MinAmount",
    "NumCurrencies", "NumFormats", "HourBucket"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")


# … after casting & optional null‐dropping …
total_count = df.count()
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(f"Total rows after cast/drop: {total_count}")

train, test = df.randomSplit([0.8, 0.2], seed=42)
print(f"  ↳ train rows: {train.count()}, test rows: {test.count()}")


print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


# ──────────────────────────────────────────────────────────────────────────────
# 4a) Pipeline for GBTClassifier
# ──────────────────────────────────────────────────────────────────────────────
# ─── Spark MLlib GBT stage ─────────────────────────────────────────────────────
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=50)
pipeline_gbt = Pipeline(stages=[label_indexer, assembler, gbt])

# ─── XGBoost4J-Spark Python wrapper stage ──────────────────────────────────────
xgb = SparkXGBClassifier(
    label_col="label",
    features_col="features",
    eta=0.1,                  # learning rate
    max_depth=6,     
    objective="binary:logistic",
    num_round=100,             # number of boosting rounds
    num_workers=4              # parallelism
)
pipeline_xgb = Pipeline(stages=[label_indexer, assembler, xgb])


# ──────────────────────────────────────────────────────────────────────────────
# 5) Split data
# ──────────────────────────────────────────────────────────────────────────────
train, test = df.randomSplit([0.8, 0.2], seed=42)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Train both models
# ──────────────────────────────────────────────────────────────────────────────
model_gbt = pipeline_gbt.fit(train)
model_xgb = pipeline_xgb.fit(train)

# ──────────────────────────────────────────────────────────────────────────────
# 7) Predict & evaluate
# ──────────────────────────────────────────────────────────────────────────────
preds_gbt = model_gbt.transform(test)
preds_xgb = model_xgb.transform(test)

evaluator = BinaryClassificationEvaluator(
    rawPredictionCol="probability", labelCol="label", metricName="areaUnderROC"
)
auc_gbt = evaluator.evaluate(preds_gbt)
auc_xgb = evaluator.evaluate(preds_xgb)

print(f"\n👉 GBTClassifier AUC    : {auc_gbt:.4f}")
print(f"👉 XGBoostClassifier AUC: {auc_xgb:.4f}")


# Load (later)
# model = joblib.load("xgb_aml_model.pkl")


# ──────────────────────────────────────────────────────────────────────────────
# 8) Keep Spark UI alive for DAG screenshot
# ──────────────────────────────────────────────────────────────────────────────
ui_url = spark.sparkContext.uiWebUrl or "http://localhost:4040"
print(f"\n▶ Spark Web UI available at: {ui_url}")
input("🔸 Press ENTER once you’ve captured the model-training DAG…")

spark.stop()
