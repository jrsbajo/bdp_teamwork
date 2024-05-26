from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder
from pyspark.ml.classification import LinearSVC, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# set up a sufix to save the file
sufix = "01"

# =========================================
# 0. Configuring Spark
# =========================================

# Configure Spark
conf = SparkConf() \
    .setAppName("ML") \
    .setMaster("yarn")

# Create SparkContext
sc = SparkContext(conf=conf)

# Create SparkSession from existing SparkContext
spark = SparkSession.builder.config(conf=sc.getConf()).getOrCreate()


# =========================================
# 1. Loading data and preprocessing
# =========================================
# Load data =============
path = "/teamwork/input/data.parquet"
df = spark.read.parquet(f'{path}')

df = df.repartition(6)

filtered_data = df.filter(col("isFlaggedFraud") == 1)
filtered_data = df.subtract(filtered_data)
filtered_data = filtered_data.drop("isFlaggedFraud", "nameOrig", "nameDest").dropna()

# Cast columns to numeric types
numeric_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
for col_name in numeric_cols:
    filtered_data = filtered_data.withColumn(col_name, col(col_name).cast(DoubleType()))

# Assuming `filtered_data` is your DataFrame and `type` is the column to preprocess
stringIndexer = StringIndexer(inputCol="type", outputCol="type_index", handleInvalid="skip")
model = stringIndexer.fit(filtered_data)
indexed_data = model.transform(filtered_data)

# Apply OneHotEncoder
encoder = OneHotEncoder(inputCols=["type_index"], outputCols=["type_encoded"])
encoded_data = encoder.fit(indexed_data).transform(indexed_data)

# Drop the original categorical column if desired
encoded_data = encoded_data.drop("type", "type_index")

# Assemble features
assembler = VectorAssembler(inputCols=["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "type_encoded"], outputCol="features")
scaled_data = assembler.transform(encoded_data)

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
final_df = scaler.fit(scaled_data).transform(scaled_data)
final_df = final_df.withColumn("isFraud", col("isFraud").cast("integer"))


# =========================================
# 2. Split Data
# =========================================
train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)


# =========================================
# 3. Build and Train Models
# =========================================

# RANDOM FOREST
# -------------
# Create a RandomForest model
rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="isFraud", seed=42)

# Train model
model_rf = rf.fit(train_data)

# SVM
# ---
# Create an SVM model
svm = LinearSVC(labelCol="isFraud", featuresCol="scaledFeatures", maxIter=10)

# Train model
model_svm = svm.fit(train_data)


# =========================================
# 4. Evaluate the Models
# =========================================

# Evaluate models
evaluator = BinaryClassificationEvaluator(labelCol="isFraud")

auc_randomforest = evaluator.evaluate(model_rf.transform(test_data))
auc_svm = evaluator.evaluate(model_svm.transform(test_data))
print("Random Forest AUC: ", auc_randomforest)
print("SVM AUC: ", auc_svm)

# Create an RDD with the AUC and save it
auc_rdd = sc.parallelize([auc_randomforest, auc_svm])
auc_rdd.map(lambda x: str(x)).saveAsTextFile(f"/teamwork/output/ml_auc_{sufix}")


# # Save the trained model
# model_path = "/modelos/rf"
# model_rf.save(model_path)

# model_path = "/modelos/svm"
# model_svm.save(model_path)
