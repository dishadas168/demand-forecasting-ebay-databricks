# Databricks notebook source
# MAGIC %md
# MAGIC # ARIMA training
# MAGIC - This is an auto-generated notebook.
# MAGIC - To reproduce these results, attach this notebook to a cluster with runtime version **14.1.x-cpu-ml-scala2.12**, and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/623356438753392).
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.

# COMMAND ----------

import mlflow
import databricks.automl_runtime
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder\
        .appName("data processing app")\
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1')\
        .getOrCreate()

# COMMAND ----------



target_col = "price_lower"
time_col = "date_sold"
unit = "day"

id_cols = ["country", "tea_type"]

horizon = 7

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

import mlflow
import os
import uuid
import shutil
import pandas as pd
import pyspark.pandas as ps

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
print(input_temp_dir)
os.makedirs(input_temp_dir)

# # Download the artifact and read it into a pandas DataFrame
input_data_path = mlflow.artifacts.download_artifacts(run_id="b86af23b312b418fbc612853c968478c", artifact_path="data", dst_path=input_temp_dir)

input_file_path = os.path.join(input_data_path, "training_data")
input_file_path = "file://" + input_file_path

df_loaded = ps.from_pandas(pd.read_parquet(input_file_path))

Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate data by `id_col` and `time_col`
# MAGIC Group the data by `id_col` and `time_col`, and take average if there are multiple `target_col` values in the same group.

# COMMAND ----------

group_cols = [time_col] + id_cols

df_aggregated = df_loaded \
  .groupby(group_cols) \
  .agg(y=(target_col, "avg")) \
  .reset_index()

df_aggregated = df_aggregated.assign(ts_id=lambda x:x["country"].astype(str)+"-"+x["tea_type"].astype(str))


# Filter out the time series with too few data. Models won't be trained for the timeseries
# with these identities. Please provide more data for these timeseries.
df_aggregated = df_aggregated.loc[~df_aggregated["ts_id"].isin(['Unknown-Earl Grey', 'United States-Mate', 'Taiwan-Matcha', 'China-Boba Tea', 'Poland-Ceylon', 'China-English Breakfast', 'Korea, Republic of-Black', 'Argentina-Black', 'Argentina-White', 'Not Specified-English Breakfast', 'United States-Yellow', 'Taiwan-Herbal/Tisane', 'United States-Darjeeling', 'Poland-Iced Tea', 'India-Oolong', 'China-Chai', 'Canada-Iced Tea', 'China-Darjeeling', 'Not Specified-Da Hong Pao', 'United States-Pu-erh', 'Poland-Yerba Mate', 'Canada-Oolong', 'Thailand-White', 'Not Specified-Iced Tea', 'China-Assam', 'Korea, Republic of-Pu-erh', 'Unknown-Ceylon', 'Not Specified-Tie Guan Yin', 'Canada-Ceylon', 'Not Specified-Gunpowder Tea', 'United States-Nettle Leaf', 'China-Chamomile Tea', 'India-Ceylon', 'China-Ceylon', 'United States-Rose Petal', 'Not Specified-Chamomile Tea', 'Canada-Green', 'India-Pu-erh', 'Sri Lanka-Darjeeling', 'Korea, Republic of-Breakfast Tea', 'Taiwan-Pu-erh', 'United Kingdom-Oolong', 'Not Specified-Matcha', 'Canada-Earl Grey', 'Argentina-Not Specified', 'China-Gunpowder Tea', 'Unknown-Assortment', 'Not Specified-Earl Grey', 'Poland-Breakfast Tea', 'Taiwan-Assam', 'Sri Lanka-Oolong', 'Not Specified-Green', 'Poland-Chai', 'Poland-Rooibos & Honeybush', 'Not Specified-Herbal/Tisane', 'Not Specified-Turkish Apple Tea', 'Japan-Matcha', 'Not Specified-Chai', 'Not Specified-Assortment', 'India-English Breakfast', 'United States-Ceylon', 'China-Breakfast Tea', 'India-Iced Tea', 'Taiwan-Earl Grey', 'Bulgaria-Nettle Leaf', 'Unknown-Yerba Mate', 'Taiwan-Assortment', 'Japan-Green', 'Korea, Republic of-Chamomile Tea', 'Unknown-Matcha', 'United States-Yorkshire Tea', 'United Kingdom-White'])]
df_aggregated.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train ARIMA model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/623356438753392)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment

# COMMAND ----------

# Define the search space of seasonal period m
seasonal_periods = [1, 7]

# COMMAND ----------



# COMMAND ----------

from pyspark.sql.types import *

df_schema = df_aggregated.to_spark().schema
result_columns = id_cols + ["pickled_model", "start_time", "end_time", "mse",
                  "rmse", "mae", "mape", "mdape", "smape", "coverage"]
result_schema = StructType(
  [StructField(id_col, df_schema[id_col].dataType) for id_col in id_cols] + [
  StructField("pickled_model", BinaryType()),
  StructField("start_time", TimestampType()),
  StructField("end_time", TimestampType()),
  StructField("mse", FloatType()),
  StructField("rmse", FloatType()),
  StructField("mae", FloatType()),
  StructField("mape", FloatType()),
  StructField("mdape", FloatType()),
  StructField("smape", FloatType()),
  StructField("coverage", FloatType())
  ])

def arima_training(history_pd):
  from databricks.automl_runtime.forecast.pmdarima.training import ArimaEstimator

  arima_estim = ArimaEstimator(horizon=horizon,
                               frequency_unit=unit,
                               metric="smape",
                               seasonal_periods=seasonal_periods,
                               num_folds=2)

  results_pd = arima_estim.fit(history_pd)
  results_pd[id_cols] = history_pd[id_cols]
  results_pd["start_time"] = pd.Timestamp(history_pd["ds"].min())
  results_pd["end_time"] = pd.Timestamp(history_pd["ds"].max())
 
  return results_pd[result_columns]

def train_with_fail_safe(df):
  try:
    return arima_training(df)
  except Exception as e:
    print(f"Encountered an exception while training timeseries: {repr(e)}")
    return pd.DataFrame(columns=result_columns)

# COMMAND ----------

import mlflow
from databricks.automl_runtime.forecast.pmdarima.model import MultiSeriesArimaModel, mlflow_arima_log_model

with mlflow.start_run(experiment_id="623356438753392", run_name="Arima") as mlflow_run:
  mlflow.set_tag("estimator_name", "ARIMA")

  df_aggregated = df_aggregated.rename(columns={time_col: "ds"})

  arima_results = (df_aggregated.to_spark().repartition(sc.defaultParallelism, "ts_id") \
    .groupby("ts_id").applyInPandas(train_with_fail_safe, result_schema)).cache().pandas_api()
  arima_results = arima_results.to_pandas()
  arima_results["ts_id"] = arima_results[id_cols].astype(str).agg('-'.join, axis=1)
  arima_results["ts_id_tuple"] = arima_results[id_cols].apply(tuple, axis=1)
   
  # Check whether every time series's model is trained
  ts_models_trained = set(arima_results["ts_id"].unique().tolist())
  ts_ids = set(df_aggregated["ts_id"].unique().tolist())

  if len(ts_models_trained) == 0:
    raise Exception("Trial unable to train models for any identities. Please check the training cell for error details")

  if ts_ids != ts_models_trained:
    mlflow.log_param("partial_model", True)
    print(f"WARNING: Models not trained for the following identities: {ts_ids.difference(ts_models_trained)}")
 
  # Log metrics to mlflow
  metric_names = ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]
  avg_metrics = arima_results[metric_names].mean().to_frame(name="mean_metrics").reset_index()
  avg_metrics["index"] = "val_" + avg_metrics["index"].astype(str)
  avg_metrics.set_index("index", inplace=True)
  mlflow.log_metrics(avg_metrics.to_dict()["mean_metrics"])

  # Save the model to mlflow
  arima_results = arima_results.set_index("ts_id_tuple")
  pickled_model = arima_results["pickled_model"].to_dict()
  start_time = arima_results["start_time"].to_dict()
  end_time = arima_results["end_time"].to_dict()
  arima_model = MultiSeriesArimaModel(pickled_model, horizon, unit, start_time, end_time, time_col, id_cols)

  # Generate sample input dataframe
  sample_input = df_loaded.tail(5).to_pandas()
  sample_input[time_col] = pd.to_datetime(sample_input[time_col])
  sample_input.drop(columns=[target_col], inplace=True)

  mlflow_arima_log_model(arima_model, sample_input=sample_input)

# COMMAND ----------

avg_metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze the predicted results

# COMMAND ----------

# Load the model
run_id = mlflow_run.info.run_id
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Patch pandas version in logged model
# MAGIC
# MAGIC Ensures that model serving uses the same version of pandas that was used to train the model.

# COMMAND ----------

import mlflow
import os
import shutil
import tempfile
import yaml

run_id = mlflow_run.info.run_id

# Set up a local dir for downloading the artifacts.
tmp_dir = tempfile.mkdtemp()

client = mlflow.tracking.MlflowClient()

# Fix conda.yaml
conda_file_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model/conda.yaml", dst_path=tmp_dir)
with open(conda_file_path) as f:
  conda_libs = yaml.load(f, Loader=yaml.FullLoader)
pandas_lib_exists = any([lib.startswith("pandas==") for lib in conda_libs["dependencies"][-1]["pip"]])
if not pandas_lib_exists:
  print("Adding pandas dependency to conda.yaml")
  conda_libs["dependencies"][-1]["pip"].append(f"pandas=={pd.__version__}")

  with open(f"{tmp_dir}/conda.yaml", "w") as f:
    f.write(yaml.dump(conda_libs))
  client.log_artifact(run_id=run_id, local_path=conda_file_path, artifact_path="model")

# Fix requirements.txt
venv_file_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model/requirements.txt", dst_path=tmp_dir)
with open(venv_file_path) as f:
  venv_libs = f.readlines()
venv_libs = [lib.strip() for lib in venv_libs]
pandas_lib_exists = any([lib.startswith("pandas==") for lib in venv_libs])
if not pandas_lib_exists:
  print("Adding pandas dependency to requirements.txt")
  venv_libs.append(f"pandas=={pd.__version__}")

  with open(f"{tmp_dir}/requirements.txt", "w") as f:
    f.write("\n".join(venv_libs))
  client.log_artifact(run_id=run_id, local_path=venv_file_path, artifact_path="model")

shutil.rmtree(tmp_dir)

# COMMAND ----------

future_df = loaded_model._model_impl.python_model.make_future_dataframe()

# COMMAND ----------

# Predict future with the default horizon
forecast_pd = loaded_model._model_impl.python_model.predict_timeseries()

# COMMAND ----------

from databricks.automl_runtime.forecast.pmdarima.utils import plot

# Choose a random id from `ts_id` for plot
forecast_pd["ts_id"] = forecast_pd[id_cols].astype(str).agg('-'.join, axis=1)
forecast_pd["ts_id_tuple"] = forecast_pd[id_cols].apply(tuple, axis=1)
id_ = set(forecast_pd.index.to_list()).pop()
ts_id = forecast_pd["ts_id"].loc[id_]
ts_id_tuple = forecast_pd["ts_id_tuple"].loc[id_]
forecast_pd_plot = forecast_pd[forecast_pd["ts_id"] == ts_id]
history_pd_plot = df_aggregated[df_aggregated["ts_id"] == ts_id].to_pandas()
# When visualizing, we ignore the first d (differencing order) points of the prediction results
# because it is impossible for ARIMA to predict the first d values
d = loaded_model._model_impl.python_model.model(ts_id_tuple).order[1]
fig = plot(history_pd_plot[d:], forecast_pd_plot[d:])
fig

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show the predicted results

# COMMAND ----------

predict_cols = id_cols + ["ds", "yhat"]
forecast_pd = forecast_pd.reset_index()
display(forecast_pd[predict_cols].tail(horizon))
