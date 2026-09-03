from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, when, to_date

# =========================
# STEP 1: START SPARK
# =========================
spark = SparkSession.builder.appName("Healthcare Dashboard").getOrCreate()

# =========================
# STEP 2: LOAD DATA
# =========================
patients = spark.read.csv("patients.csv", header=True, inferSchema=True)
admissions = spark.read.csv("admissions.csv", header=True, inferSchema=True)
diagnoses = spark.read.csv("diagnoses.csv", header=True, inferSchema=True)
billing = spark.read.csv("billing.csv", header=True, inferSchema=True)
hospitals = spark.read.csv("hospitals.csv", header=True, inferSchema=True)

# =========================
# STEP 3: FIX DUPLICATE COLUMN NAMES
# =========================
patients = patients.withColumnRenamed("state", "patient_state")
hospitals = hospitals.withColumnRenamed("state", "hospital_state")

# =========================
# STEP 4: FORMAT DATES
# =========================
admissions = admissions.withColumn("admit_date", to_date("admit_date"))
admissions = admissions.withColumn("discharge_date", to_date("discharge_date"))

if "next_admit_date" in admissions.columns:
    admissions = admissions.withColumn("next_admit_date", to_date("next_admit_date"))

# =========================
# STEP 5: JOIN ALL TABLES
# =========================
df = admissions.join(patients, "patient_id") \
               .join(diagnoses, "admission_id") \
               .join(billing, "admission_id") \
               .join(hospitals, "hospital_id")

print("Merged Rows:", df.count())

# =========================
# STEP 6: CREATE READMISSION
# =========================
if "next_admit_date" in df.columns:
    df = df.withColumn(
        "readmitted_7d",
        when(datediff(col("next_admit_date"), col("discharge_date")) <= 7, 1).otherwise(0)
    )
else:
    df = df.withColumn(
        "readmitted_7d",
        when(col("discharge_type") != "Home", 1).otherwise(0)
    )

# =========================
# STEP 7: DEBUG (SEE REAL COLUMNS)
# =========================
print("AVAILABLE COLUMNS:\n", df.columns)

# =========================
# STEP 8: SELECT CORRECT COLUMNS
# =========================
df = df.select(
    col("patient_id"),
    col("admission_id"),

    col("name").alias("doctor_name"),  # from hospitals

    col("gender"),
    col("patient_state"),
    col("hospital_state"),

    col("admit_type"),
    col("ward_type"),
    col("discharge_type"),
    col("insurance_type"),

    col("diag_desc").alias("disease_type"),   # from diagnoses
    col("diag_rank"),                         # optional useful field

    col("readmitted_7d"),

    col("total_cost_inr").alias("cost"),      # from billing
    col("los_days").alias("waiting_time")     # from admissions
)

# =========================
# STEP 9: HANDLE NULLS
# =========================
df = df.fillna({
    "insurance_type": "Unknown",
    "disease_type": "Unknown"
})

# =========================
# STEP 10: SAVE FILE
# =========================
df.coalesce(1).toPandas().to_csv("final_dashboard.csv", index=False)
print("✅ SUCCESS: Dataset ready for Tableau!")