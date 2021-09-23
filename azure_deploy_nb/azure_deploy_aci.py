# Databricks notebook source
# %pip install azureml-sdk

# COMMAND ----------

import mlflow
import pandas as pd

workspace_name = "yw-aml-ws-1"
workspace_location = "australiaeast"
resource_group = "yw-aml-rs-1"
subscription_id = "6369c148-f8a9-4fb5-8a9d-ac1b2c8e756e"

# COMMAND ----------

# TODO: svc token to secret scope

# COMMAND ----------

import os
import azureml
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

svc_pr_password = "password"

svc_pr = ServicePrincipalAuthentication(
    tenant_id="9f37a392-f0ae-4280-9796-f1864a10effc",
    service_principal_id="c0e7de4d-5b09-4920-b26e-636b6d889aca",
    service_principal_password=svc_pr_password)


workspace3 = Workspace(
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name,
    auth=svc_pr
    )

print("Found workspace {} at location {}".format(workspace3.name, workspace3.location))

# COMMAND ----------

workspace3.get_details()

# COMMAND ----------

model_name="rfmodel"
scope = 'cmr'
secret_prefix = 'cmr'
model_version_uri_remote = "models://{scope}:{secret_prefix}@databricks/{model_name}/1".format(scope=scope,secret_prefix=secret_prefix,model_name=model_name)

# COMMAND ----------

model_version_uri_remote

# COMMAND ----------

modeltest = mlflow.pyfunc.load_model(model_version_uri_remote)

# COMMAND ----------

df = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")
modeltest.predict(df.drop(["price"], axis=1).iloc[[0]])

# COMMAND ----------

# MAGIC %md ### Make sure the service is not deployed

# COMMAND ----------

# TODO: Delete service if service exisits

# COMMAND ----------

import mlflow.azureml
from azureml.core.webservice import AciWebservice

# Set deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

#Create an Azure Container Instance webservice for an MLflow model
azure_service, azure_model = mlflow.azureml.deploy(model_uri=model_version_uri_remote,
                                                   model_name="rfmodel10",
                                                   service_name="rfmodelservice1",
                                                   workspace=workspace3,
                                                   deployment_config=deployment_config,
                                                   synchronous=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Create a sample query input.

# COMMAND ----------

sample = df.drop(["price"], axis=1).iloc[[0]]
query_input = sample.to_json(orient='split')
query_input = eval(query_input)
query_input.pop('index', None)
query_input

# COMMAND ----------

# MAGIC %md
# MAGIC Create a wrapper function to make queries to a URI.

# COMMAND ----------

import requests
import json

def query_endpoint(scoring_uri, inputs, service_key=None):
  headers = {
    "Content-Type": "application/json",
  }
  if service_key is not None:
    headers["Authorization"] = "Bearer {service_key}".format(service_key=service_key)
    
  print("Sending batch prediction request with inputs: {}".format(inputs))
  response = requests.post(scoring_uri, data=json.dumps(inputs), headers=headers)
  preds = json.loads(response.text)
  print("Received response: {}".format(preds))
  return preds

# COMMAND ----------

# MAGIC %md
# MAGIC Query the endpoint using the scoring URI.

# COMMAND ----------

dev_prediction = query_endpoint(scoring_uri=azure_service.scoring_uri, inputs=query_input)

# COMMAND ----------


