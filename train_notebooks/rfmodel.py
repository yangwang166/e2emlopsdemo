# Databricks notebook source
# MAGIC %md # Modeling Training

# COMMAND ----------

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow.sklearn

# Define paths
modelPath = "random-forest-model"

# Train and log model
df = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

rf = RandomForestRegressor(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)

with mlflow.start_run(run_name="RF Model") as run: 
  mlflow.sklearn.log_model(rf, modelPath)
  runID = run.info.run_id

print("Run completed with ID {} and path {}".format(runID, modelPath))
model_uri = f"runs:/{run.info.run_id}/{modelPath}"

# COMMAND ----------

# MAGIC %md # Registry model to CMR (Run on Prod)

# COMMAND ----------

# MAGIC %md
# MAGIC Config secret scope for the CMR's token
# MAGIC 
# MAGIC databricks --profile mlopsprod secrets create-scope --scope cmr
# MAGIC 
# MAGIC databricks --profile mlopsprod secrets put --scope cmr --key cmr-token
# MAGIC * Create PAT from CMR workspace
# MAGIC 
# MAGIC databricks --profile mlopsprod secrets put --scope cmr --key cmr-host
# MAGIC * https://adb-2605268180347428.8.azuredatabricks.net/
# MAGIC 
# MAGIC databricks --profile mlopsprod secrets put --scope cmr --key cmr-workspace-id
# MAGIC * 2605268180347428

# COMMAND ----------

import json
context = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())
orgid = context.get('tags').get('orgId')
print("OrdId: ", orgid)

if orgid == '3059735186595105':
  print("In Prod, save model")
  scope = 'cmr'
  secret_prefix = 'cmr'
  registry_uri = 'databricks://' + scope + ':' + secret_prefix

  # Instantiate an MlflowClient pointing to the local tracking server and a remote registry server
  from mlflow.tracking.client import MlflowClient
  client = MlflowClient(tracking_uri=None, registry_uri=registry_uri)

  model_name = "rfmodel"
  try:
    model = client.create_registered_model(model_name)
  except:
    print(f"Model {model_name} Exist in Central Model Registry")

  from mlflow.tracking.artifact_utils import get_artifact_uri
  source = get_artifact_uri(run_id=runID, artifact_path=modelPath)
  client.create_model_version(name=model_name, source=source, run_id=runID)
else:
  print("Not in Prod, do nothing")

# COMMAND ----------

# Add new feature

# COMMAND ----------

# Add new feature jira2

# COMMAND ----------

# Add new feature jira3

# COMMAND ----------

# Add new feature jira4

# COMMAND ----------

# 20210816 Add new feature jira5

# COMMAND ----------

# Add new feature jira6

# COMMAND ----------

# Add new feature jira7
