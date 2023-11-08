# Databricks notebook source
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# COMMAND ----------

spark = SparkSession.builder\
        .appName("data processing app")\
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1')\
        .getOrCreate()

# COMMAND ----------

df = spark.read.table("processed_sales")
df.limit(5).toPandas()

# COMMAND ----------

# Aggregated Table

aggregated_df = df.groupBy("date_sold", "country", "tea_type")\
    .agg(
    F.sum("price_lower").alias("total_sales"),
    F.avg("price_lower").alias("average_sales"),
    F.min("price_lower").alias("min_sales"),
    F.max("price_lower").alias("max_sales"),
    F.count("title").alias("quantity")
)

aggregated_df.write \
  .format("parquet") \
  .mode("overwrite") \
  .partitionBy("country", "tea_type") \
  .saveAsTable("sales_aggregation_table")

# COMMAND ----------

# Shipping Info Table

shipping_df = df.withColumn("has_shipping", F.when(col("shipping") > 0, "Yes").otherwise("No"))

shipping_df = shipping_df.groupBy("date_sold", "country", "tea_type")\
    .agg(
    F.sum("shipping").alias("total_shipping"),
    F.avg("shipping").alias("average_shipping"),
    F.min("shipping").alias("min_shipping"),
    F.max("shipping").alias("max_shipping"),
    F.count(F.when(col("has_shipping") == "Yes", 1)).alias("orders_with_shipping")
)
    
shipping_df.write \
  .format("parquet") \
  .mode("overwrite") \
  .partitionBy("country", "tea_type") \
  .saveAsTable("shipping_table")

# COMMAND ----------

# Seller Info Table

seller_df = df.groupBy("date_sold", "country", "tea_type","seller_name","seller_sold_items","seller_rating")\
    .agg(
    F.sum("price_lower").alias("total_sales"),
    F.avg("price_lower").alias("average_sales"),
    F.min("price_lower").alias("min_sales"),
    F.max("price_lower").alias("max_sales"),
    F.count("title").alias("quantity")
)
    
seller_df.write \
  .format("parquet") \
  .mode("overwrite") \
  .partitionBy("country", "tea_type","seller_name") \
  .saveAsTable("seller_table")

# COMMAND ----------


