# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, DateType, BooleanType

import calendar
from datetime import datetime
from dateutil import tz

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

def get_scrape_date(date_string):

    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/Chicago')
    date_string = date_string.split(".")[0]
    date_format = "%Y-%m-%dT%H:%M:%S"
    dt_obj = datetime.strptime(date_string, date_format)
    dt_obj = dt_obj.replace(tzinfo=from_zone)
    dt_obj = dt_obj.astimezone(to_zone)
    month = calendar.month_abbr[int(dt_obj.month)]

    scrape_date = f"Sold  {month} {dt_obj.day}, {dt_obj.year}"
    return scrape_date

dbutils.widgets.text("date","")
date_param = dbutils.widgets.get("date")
scrape_date = get_scrape_date(date_param)
# scrape_date = "Sold  Nov 6, 2023"

df = spark.read\
    .format("com.mongodb.spark.sql.DefaultSource")\
    .option("database", databasename)\
    .option("collection",collection_name)\
    .option("pipeline", [{'$match': {"date_sold": scrape_date}}])\
    .schema(custom_schema)\
    .load()

df.show(n=5)

# COMMAND ----------

df.write.mode('append').saveAsTable("raw_sales")

# COMMAND ----------


