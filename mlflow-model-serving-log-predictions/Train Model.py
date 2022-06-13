# Databricks notebook source
# MAGIC %md
# MAGIC ## Example: Log Predictions of an MLflow Model
# MAGIC 
# MAGIC This notebook demonstrates how predictions from an MLflow Model, created on a Model Serving V1 cluster, can be persisted onto the Databricks File System (DBFS).
# MAGIC 
# MAGIC The high-level steps of this notebook are the following:
# MAGIC 1. Load the training and test dataset. The `wine-quality` dataset from `/databricks-datasets` is used for this.
# MAGIC 2. Create a `pyfunc.PythonModel` wrapper class, which will wrap our actual model and will save the predictions each time `predict` is called.
# MAGIC 3. Train our model, wrap in the previously created wrapper class and store it in MLflow
# MAGIC 4. Register the model into the MLflow Model Registry
# MAGIC 5. Turn-on the Model Serving cluster
# MAGIC 6. Make calls to the REST API Endpoint of the model
# MAGIC 7. Verify that the predictions get stored.

# COMMAND ----------

# MAGIC %md **Import required packages**

# COMMAND ----------

import os
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
import uuid

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# COMMAND ----------

# MAGIC %md ### 1. Data Loading and Preparation

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/wine-quality/

# COMMAND ----------

white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=';')
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=';')

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0
 
data = pd.concat([red_wine, white_wine], axis=0)
 
# Remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

data

# COMMAND ----------

X = data.drop(["quality"], axis=1)
y = data.quality
 
# Split out the training data
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)
 
# Split the remaining data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

# COMMAND ----------

# MAGIC %md ### 2. Create the Wrapper Class

# COMMAND ----------

PREDICTIONS_STORAGE_LOCATION = "/dbfs/tmp/predictions/"  # Local DBFS file path for the predictions storage location


class PredictionSaverModelWrapper(mlflow.pyfunc.PythonModel):
  
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    predictions = self.model.predict_proba(model_input)[:,1]
    
    # The below three lines of code are used to persist
    # any predictions made by the model.
    df = pd.DataFrame(predictions, columns=["probability"])
    random_id = str(uuid.uuid4())
    df.to_csv(f"{PREDICTIONS_STORAGE_LOCATION}{random_id}.csv", index=False)
    
    return predictions

# COMMAND ----------

dbutils.fs.mkdirs(PREDICTIONS_STORAGE_LOCATION.replace("/dbfs", ""))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3. Train & Save the Model
# MAGIC 
# MAGIC Below we train a simple model on our dataset, wrap it into our `ModelWrapper` class and save it into MLflow.

# COMMAND ----------

with mlflow.start_run() as run:
  n_estimators = 5

  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)
  wrappedModel = PredictionSaverModelWrapper(model)
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel)

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/random_forest_model")

# COMMAND ----------

# MAGIC %md Briefly testing whether it works:

# COMMAND ----------

loaded_model.predict(X_test)

# COMMAND ----------

display(dbutils.fs.ls(PREDICTIONS_STORAGE_LOCATION.replace("/dbfs", "")))

# COMMAND ----------

# MAGIC %md #### 4. Register the Model in the Model Registry

# COMMAND ----------

model_uri = f"runs:/{run.info.run_id}/random_forest_model"

model_version = mlflow.register_model(model_uri, "example_model", await_registration_for=30)

# COMMAND ----------

# MAGIC %md ### 5. Turn on the Model Serving Cluster
# MAGIC 
# MAGIC In the menu on the left, go to Models -> *example_model* -> Serving -> Enable Serving.
# MAGIC 
# MAGIC Wait till the Model Serving cluster is up and running.

# COMMAND ----------

# MAGIC %md #### 6. Call the Model endpoint

# COMMAND ----------

workspace_host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()
model_url = f'https://{workspace_host_name}/model/{model_version.name}/{model_version.version}/invocations'

print(model_url)

def score_model(dataset):
  
  headers = {'Authorization': f'Bearer {dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()}'}
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  response = requests.request(method='POST', headers=headers, url=model_url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()


score_model(X_test)

# COMMAND ----------

# MAGIC %md ### 7. Verify that predictions get stored.

# COMMAND ----------

path_on_dbfs = PREDICTIONS_STORAGE_LOCATION.replace("/dbfs", "")

display(dbutils.fs.ls(path_on_dbfs))

# COMMAND ----------

# MAGIC %md From here on we can start analysing our predictions.

# COMMAND ----------

df = spark.read.csv(path_on_dbfs, schema="probability double", header=True)

display(df)

# COMMAND ----------

# MAGIC %md Or if we want to incrementally process the predictions, we can make use of <a href="https://docs.microsoft.com/de-de/azure/databricks/ingestion/auto-loader/" target="_blank">Databricks Auto Loader</a>:

# COMMAND ----------

df = (spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("cloudFiles.schemaLocation", "/tmp/predictions.schema")
        .option("cloudFiles.schemaHints", "probability double")
        .load(path_on_dbfs)
      )

# COMMAND ----------

display(df)
