from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType


# ************************************************
# Data streaming implementation for the query 5:
# TRANSACTION VOLUME OVER TIME
# ************************************************


# Create Spark session
spark = SparkSession \
    .builder \
    .appName("TransactionStreaming") \
    .getOrCreate()

# Define data input schema
schema = StructType([
    StructField("step", IntegerType(), True),
    StructField("type", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("nameOrig", StringType(), True),
    StructField("oldbalanceOrg", DoubleType(), True),
    StructField("newbalanceOrig", DoubleType(), True),
    StructField("nameDest", StringType(), True),
    StructField("oldbalanceDest", DoubleType(), True),
    StructField("newbalanceDest", DoubleType(), True),
    StructField("isFraud", IntegerType(), True),
    StructField("isFlaggedFraud", IntegerType(), True)
])

# Read data streaming
transactions = spark \
    .readStream \
    .schema(schema) \
    .parquet("/streaming/input")

# Set threshold
large_amount_threshold = 200000

# Filtering transactions due to 
large_non_fraud_transactions = transactions.filter((col("amount") > large_amount_threshold) & (col("isFraud") == 0))


# Uncomment to SAVE processed data INTO A FILE
# query = large_non_fraud_transactions \
#     .select("type", "amount") \
#     .writeStream \
#     .outputMode("append") \
#     .format("csv") \
#     .option("path", "/streaming/output") \
#     .option("checkpointLocation", "/streaming/checkpoint") \
#     .start()
# ===

# Print data
query = large_non_fraud_transactions \
    .select("type", "amount") \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .trigger(processingTime='10 seconds') \
    .start()


query.awaitTermination()
