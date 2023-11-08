# Databricks notebook source
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import struct, col, monotonically_increasing_id

import datetime

# COMMAND ----------

spark = SparkSession.builder\
        .appName("data processing app")\
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1')\
        .getOrCreate()
spark.conf.set("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")

# COMMAND ----------

df = spark.read.table("processed_sales")
df.limit(5).toPandas()

# COMMAND ----------

logged_model = 'runs:/7c326a288f5f428392a2506855866faf/model'
loaded_model = mlflow.pyfunc.load_model(model_uri=logged_model)


# COMMAND ----------

future_df = loaded_model._model_impl.python_model.make_future_dataframe()
future_df.tail(7)

# COMMAND ----------

future_df = spark.createDataFrame(future_df)
future_df = future_df.withColumnRenamed("ds", "date_sold")
future_df.show()

# COMMAND ----------

# Predict on a Spark DataFrame.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')
future_df=future_df.withColumn('predictions', loaded_model(struct(*map(col, future_df.columns))))

# COMMAND ----------

fd = future_df.toPandas()

# COMMAND ----------

fd.tail()

# COMMAND ----------

# Predict future with the default horizon
forecast_pd = loaded_model._model_impl.python_model.model().predict(future_df)

# COMMAND ----------

forecast_pd.tail()

# COMMAND ----------

predict_cols = ["date_sold", "predictions"]
fd = fd.reset_index()
display(fd[predict_cols].tail(7))

# COMMAND ----------

fd.write \
  .format("parquet") \
  .mode("overwrite") \
  .partitionBy("country", "tea_type") \
  .saveAsTable("7day_forecast_table")
