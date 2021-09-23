# Databricks notebook source
import mlflow
from mlflow.utils.rest_utils import http_request
import json

def client():
  return mlflow.tracking.client.MlflowClient()

host_creds = client()._tracking_client.store.get_host_creds()

def mlflow_call_endpoint(endpoint, method, body='{}'):
  if method == 'GET':
    response = http_request(
        host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
  else:
    response = http_request(
        host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
  print("Response: {}".format(response.json()))

# COMMAND ----------

# MAGIC %md # Create Webhook for slack

# COMMAND ----------

# Support events:
#  MODEL_VERSION_CREATED
#  TRANSITION_REQUEST_CREATED
#  MODEL_VERSION_TRANSITIONED_STAGE
#  COMMENT_CREATED
#  MODEL_VERSION_TAG_SET
#  REGISTERED_MODEL_CREATED

json_slack = {"model_name": "rfmodel", "events": ["MODEL_VERSION_CREATED", "TRANSITION_REQUEST_CREATED", "MODEL_VERSION_TRANSITIONED_STAGE"], "http_url_spec" : {"url": "https://hooks.slack.com/services/your_token"}, "status":"Active"}
mlflow_call_endpoint("registry-webhooks/create", "POST", body=json.dumps(json_slack))

# COMMAND ----------

# MAGIC %md # Create Webhook for Jenkins

# COMMAND ----------

json_obj = {"model_name": "rfmodel", "events": ["MODEL_VERSION_TRANSITIONED_STAGE"], "http_url_spec" : {"url": "https://willwy.com/generic-webhook-trigger/invoke?token=jenkins-mlflow-webhook", "enable_ssl_verification": "false"}, "status":"ACTIVE"}
mlflow_call_endpoint("registry-webhooks/create", "POST", body=json.dumps(json_obj))

# COMMAND ----------

# MAGIC %md # List

# COMMAND ----------

json_list = {"model_name": "rfmodel"}
mlflow_call_endpoint("registry-webhooks/list", "GET", body=json.dumps(json_list))

# COMMAND ----------

# MAGIC %md # Delete

# COMMAND ----------

json_del = {"id": "d0489dadcfa745bd89be9593444f35d2"}
mlflow_call_endpoint("registry-webhooks/delete", "DELETE", body=json.dumps(json_del))

# COMMAND ----------

# MAGIC %md # Test

# COMMAND ----------

json_test = {"id": "30af004ff721424c9852c291e31950df"}
mlflow_call_endpoint("registry-webhooks/test", "POST", body=json.dumps(json_test))


# COMMAND ----------

# Webhook bug: https://databricks.atlassian.net/browse/ML-15769

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -k https://40.126.247.85:8443/generic-webhook-trigger/invoke?token=jenkins-bitbucket-webhook2

# COMMAND ----------

# MAGIC %sh
# MAGIC curl https://40.126.247.85:8443/generic-webhook-trigger/invoke?token=jenkins-bitbucket-webhook2

# COMMAND ----------



