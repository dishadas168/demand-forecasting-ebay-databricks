# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, DateType, BooleanType

# COMMAND ----------

databasename= "demand-froecasting-ebay"
collection_name = "raw"

# COMMAND ----------

spark = SparkSession.builder\
        .appName("data enhancing app")\
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1')\
        .getOrCreate()

# COMMAND ----------

custom_schema = StructType([ \
    StructField("_id",StringType(),False), \
    StructField("country",StringType(),True), \
    StructField("date_extracted",DateType(),True), \
    StructField("date_sold", StringType(), True), \
    StructField("location", StringType(), True), \
    StructField("page", IntegerType(), True), \
    StructField("price", StringType(), True), \
    StructField("purchase_option", StringType(), True), \
    StructField("scraped", BooleanType(), True), \
    StructField("seller", StringType(), True), \
    StructField("shipping", StringType(), True), \
    StructField("subtitle", StringType(), True), \
    StructField("tea_type", StringType(), True), \
    StructField("title", StringType(), True), \
    StructField("url", StringType(), True), \
  ])

# COMMAND ----------

df = spark.read\
    .format("com.mongodb.spark.sql.DefaultSource")\
    .option("database", databasename)\
    .option("collection",collection_name)\
    .option("pipeline", [{'$match': {"date_sold": "Sold  Oct 30, 2023"}}])\
    .schema(custom_schema)\
    .load()

df.show(n=5)

# COMMAND ----------

df.write.mode('append').format("parquet").saveAsTable("raw_sales")
