# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col, split, regexp_replace
from pyspark.sql.types import DateType, FloatType, StringType

from datetime import datetime
import calendar
from dateutil import tz

# COMMAND ----------

spark = SparkSession.builder\
        .appName("data processing app")\
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1')\
        .getOrCreate()
spark.conf.set("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")

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
# scrape_date="Sold  Nov 6, 2023"

df = spark.read.table("raw_sales")
df = df.filter(col('date_sold') == scrape_date)
df.limit(5).toPandas()

# COMMAND ----------

df.describe().toPandas()

# COMMAND ----------

df.select(*[
    (
        count(when((isnan(c) | col(c).isNull()), c)) if t not in ("timestamp", "date", "boolean")
        else count(when(col(c).isNull(), c))
    ).alias(c)
    for c, t in df.dtypes if c in df.columns
]).toPandas()

# COMMAND ----------

df = df.na.drop(subset=["country","date_sold","tea_type","url","subtitle","seller","price","shipping"])
df = df.drop("location","subtitle")
df = df.fillna(" ", subset=["purchase_option"])

# COMMAND ----------

def standardize_date(string):
  date_format = "%b %d, %Y"
  string = str(string).replace("Sold  ", "")
  print(string)
  string = datetime.strptime(string, date_format)
  return string

std_date_func = udf(lambda ds : standardize_date(ds), DateType())

df = df.withColumn("date_sold", std_date_func(df.date_sold))
df = df.sort(df.date_sold.desc())

# COMMAND ----------

def standardize_shipping(string):
  if string != string:
    string = '0'
  elif string.startswith("Free") or string.startswith("Shipping") or string=="Freight":
    string = '0'
  else:
    string = string.split(" ")[0].split("$")[1]
  return float(string)

std_shipping_func = udf(lambda s : standardize_shipping(s), FloatType())

df = df.withColumn("shipping", std_shipping_func(df.shipping))

# COMMAND ----------

def strip_price(string):
  return string.replace("$","").replace(",","").split("to")

def standardize_pricing_lower(string):
  string = strip_price(string)
  return float(string[0])

def standardize_pricing_upper(string):
  string = strip_price(string)
  return float(string[1] if len(string)>1 else string[0])

std_pricing_lower_func = udf(lambda p : standardize_pricing_lower(p), FloatType())
std_pricing_upper_func = udf(lambda p : standardize_pricing_upper(p), FloatType())

df = df.withColumn("price_lower", std_pricing_lower_func(df.price))
df = df.withColumn("price_upper", std_pricing_upper_func(df.price))
df = df.drop("price")

# COMMAND ----------

def standardize_purchase_option(string):
    if string in ["Best offer accepted","or Best Offer"]:
        string = "Best Offer"
    return string

std_purchase_option_func = udf(lambda p : standardize_purchase_option(p), StringType())

df = df.withColumn("purchase_option", std_purchase_option_func(df.purchase_option))

# COMMAND ----------

replace = r'[(),%]'

df = df.withColumn('seller_name', split(df['seller'], ' ').getItem(0))\
       .withColumn('seller_sold_items', regexp_replace(split(df['seller'], ' ')\
                    .getItem(1),replace,"")\
                    .cast("int"))\
       .withColumn('seller_rating', regexp_replace(split(df['seller'], ' ')\
                    .getItem(2), replace, "")\
                    .cast("float"))
df = df.drop("seller")

# COMMAND ----------

df.limit(5).toPandas()

# COMMAND ----------

df.schema

# COMMAND ----------

df.write.option('path','dbfs:/user/hive/warehouse/').mode('append').format("parquet").saveAsTable("processed_sales")

# COMMAND ----------


