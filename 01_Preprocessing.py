# Databricks notebook source
from pyspark.sql import SparkSession

databasename= "demand-froecasting-ebay"
collection_name = "raw"

# COMMAND ----------

spark = SparkSession.builder\
        .appName("data enhancing app")\
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1')\
        .getOrCreate()

# COMMAND ----------

df = spark.read.format("com.mongodb.spark.sql.DefaultSource")\
    .option("database", databasename)\
    .option("collection", collection_name)\
    .load()

# COMMAND ----------

df.show()

# COMMAND ----------


