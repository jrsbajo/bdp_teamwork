from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# Create Spark configuration and context
conf = SparkConf()
sc = SparkContext(conf=conf)

# Create Spark session
spark = SparkSession.builder.getOrCreate()

sufix = ".parquet"

# Load data =============
path = "/teamwork/input/data.parquet"
df = spark.read.parquet(f'{path}')


# 1 =====================
# Group by 'step' and 'type', then count the number of transactions in each group
transaction_volume_over_time = df.groupBy("step", "type").count().orderBy(F.asc("step"))

# Show the results
# transaction_volume_over_time.rdd.map(lambda r: ','.join([str(c) for c in r])).saveAsTextFile(f"/teamwork/output/spark_ex1{sufix}")
transaction_volume_over_time.write.parquet(f"/teamwork/output/spark_ex1{sufix}")


# 2 =====================
# Filter for fraudulent transactions
fraud_transactions = df.filter(df.isFraud == 1)

# Group by 'type' and count the number of fraudulent transactions for each type
fraud_count_by_type = fraud_transactions.groupBy("type").count()

# Show the results
fraud_count_by_type.write.parquet(f"/teamwork/output/spark_ex2{sufix}")


# 3 =====================
# Filter for fraudulent transactions
fraud_transactions = df.filter(df.isFraud == 1)

# Calculate the average amount of fraudulent transactions
average_fraud_transaction_value = fraud_transactions.agg(F.mean("amount").alias("avg_fraud_amount"))

# Show the results
average_fraud_transaction_value.write.parquet(f"/teamwork/output/spark_ex3{sufix}")


# 4 =====================
# Group by 'nameOrig', sum the 'amount', and order by the sum in descending order
top_customers_by_total_amount = df.groupBy("nameOrig").agg(F.sum("amount").alias("total_amount")).orderBy(F.desc("total_amount"))

# Show the top 10 customers
top_customers_by_total_amount.write.parquet(f"/teamwork/output/spark_ex4{sufix}")


# 5 =====================
# Define a threshold for what you consider a large transaction
large_amount_threshold = 200000

# Filter for transactions with large amounts that are not marked as fraud
large_non_fraud_transactions = df.filter((df.amount > large_amount_threshold) & (df.isFraud == 0))

# Select relevant columns and show some of the results
large_non_fraud_transactions.select("type", "amount").write.parquet(f"/teamwork/output/spark_ex5{sufix}")
