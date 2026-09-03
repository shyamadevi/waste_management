from pyspark.sql import SparkSession

# =========================
# STEP 1: START SPARK
# =========================
spark = SparkSession.builder.appName("Healthcare Big Data").getOrCreate()

# =========================
# STEP 2: LOAD DATA
# =========================
patients = spark.read.csv("patients.csv", header=True, inferSchema=True)
admissions = spark.read.csv("admissions.csv", header=True, inferSchema=True)
diagnoses = spark.read.csv("diagnoses.csv", header=True, inferSchema=True)

print("Patients:", patients.count())
print("Admissions:", admissions.count())
print("Diagnoses:", diagnoses.count())

# =========================
# STEP 3: MERGE DATA
# =========================
df = admissions.join(patients, on="patient_id", how="inner")
df = df.join(diagnoses, on="admission_id", how="inner")

print("Merged Rows:", df.count())

# =========================
# STEP 4: CLEAN DATA
# =========================

# Drop unnecessary + problematic columns
df = df.drop(
    "diag_desc",
    "icd10_code",
    "diag_id",
    "admission_id",
    "patient_id",
    "admit_date",
    "discharge_date",
    "hospital_id"
)

# Fill missing values
df = df.fillna({"insurance_type": "Unknown"})

# =========================
# STEP 5: ENCODE CATEGORICAL DATA
# =========================

from pyspark.ml.feature import StringIndexer

categorical_cols = [
    "gender",
    "admit_type",
    "ward_type",
    "discharge_type",
    "insurance_type",
    "diag_category",
    "state"
]

for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_index")
    df = indexer.fit(df).transform(df)

# Drop original categorical columns
df = df.drop(*categorical_cols)

# =========================
# STEP 6: PREPARE FEATURES
# =========================

from pyspark.ml.feature import VectorAssembler

feature_cols = [c for c in df.columns if c != "readmitted_7d"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# =========================
# STEP 7: TRAIN MODEL
# =========================

from pyspark.ml.classification import RandomForestClassifier

train_data, test_data = df.randomSplit([0.8, 0.2])

rf = RandomForestClassifier(labelCol="readmitted_7d", featuresCol="features")

model = rf.fit(train_data)

predictions = model.transform(test_data)

# =========================
# STEP 8: EVALUATE MODEL
# =========================

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="readmitted_7d")
accuracy = evaluator.evaluate(predictions)

print("Model Accuracy (Spark):", accuracy)

# =========================
# STEP 9: SAVE OUTPUT
# =========================

df.toPandas().to_csv("spark_final.csv", index=False)

print("✅ Spark project completed successfully!")
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol="readmitted_7d", predictionCol="prediction", metricName="accuracy"
)
print("Accuracy:", evaluator.evaluate(predictions))

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="readmitted_7d", predictionCol="prediction", metricName="f1"
)
print("F1 Score:", evaluator_f1.evaluate(predictions))