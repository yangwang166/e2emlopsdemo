# Databricks notebook source
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

modeltest = mlflow.pyfunc.load_model(model_version_uri_remote)

# COMMAND ----------

df = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")
modeltest.predict(df.drop(["price"], axis=1).iloc[[0]])

# COMMAND ----------

from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core import Image

aks_cluster_name = "prod-ml"

aks_target = ComputeTarget(workspace3, aks_cluster_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that you have defined an AKS cluster that is up and running, confirm that it is in `Succeeded` status.

# COMMAND ----------

aks_target.get_status()

# COMMAND ----------

# MAGIC %md
# MAGIC Define deploy function to AKS

# COMMAND ----------

import sys
import os
import subprocess
import logging
import uuid

from packaging.version import Version

from mlflow import get_tracking_uri, get_registry_uri
from mlflow import pyfunc
from mlflow import register_model as mlflow_register_model
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import get_unique_resource_id
from mlflow.utils.annotations import experimental
from mlflow.utils.file_utils import TempDir, _copy_file_or_tree, _copy_project
from mlflow.version import VERSION as mlflow_version

_logger = logging.getLogger(__name__)
from pathlib import Path

def deploy2(
    model_uri,
    workspace,
    deployment_config=None,
    service_name=None,
    model_name=None,
    tags=None,
    mlflow_home=None,
    synchronous=True,
    deployment_target=None
):
    from azureml.core.model import Model as AzureModel, InferenceConfig
    from azureml.core import Environment as AzureEnvironment
    from azureml.core import VERSION as AZUREML_VERSION
    from azureml.core.webservice import AciWebservice

    absolute_model_path = _download_artifact_from_uri(model_uri)

    model_pyfunc_conf, model = _load_pyfunc_conf_with_model(model_path=absolute_model_path)
    model_python_version = model_pyfunc_conf.get(pyfunc.PY_VERSION, None)
    run_id = None
    run_id_tag = None
    try:
        run_id = model.run_id
        run_id_tag = run_id
    except AttributeError:
        run_id = str(uuid.uuid4())
    if model_python_version is not None and Version(model_python_version) < Version("3.0.0"):
        raise MlflowException(
            message=(
                "Azure ML can only deploy models trained in Python 3 and above. See"
                " the following MLflow GitHub issue for a thorough explanation of this"
                " limitation and a workaround to enable support for deploying models"
                " trained in Python 2: https://github.com/mlflow/mlflow/issues/668"
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    tags = _build_tags(
        model_uri=model_uri,
        model_python_version=model_python_version,
        user_tags=tags,
        run_id=run_id_tag,
    )

    if service_name is None:
        service_name = _get_mlflow_azure_name(run_id)
    if model_name is None:
        model_name = _get_mlflow_azure_name(run_id)

    with TempDir(chdr=True) as tmp:
        model_directory_path = tmp.path("model")
        tmp_model_path = os.path.join(
            model_directory_path,
            _copy_file_or_tree(src=absolute_model_path, dst=model_directory_path),
        )

        registered_model = None
        azure_model_id = None

        # If we are passed a 'models' uri, we will attempt to extract a name and version which
        # can be used to retreive an AzureML Model. This will ignore stage based model uris,
        # which is alright until we have full deployment plugin support.
        #
        # If instead we are passed a 'runs' uri while the user is using the AzureML tracking
        # and registry stores, we will be able to register the model on their behalf using
        # the AzureML plugin, which will maintain lineage between the model and the run that
        # produced it. This returns an MLFlow Model object however, so we'll still need the
        # name and ID in order to retrieve the AzureML Model object which is currently
        # needed to deploy.
        if model_uri.startswith("models:/"):
            m_name = model_uri.split("/")[-2]
            m_version = int(model_uri.split("/")[-1])
            azure_model_id = "{}:{}".format(m_name, m_version)
        elif (
            model_uri.startswith("runs:/")
            and get_tracking_uri().startswith("azureml")
            and get_registry_uri().startswith("azureml")
        ):
            mlflow_model = mlflow_register_model(model_uri, model_name)
            azure_model_id = "{}:{}".format(mlflow_model.name, mlflow_model.version)

            _logger.info(
                "Registered an Azure Model with name: `%s` and version: `%s`",
                mlflow_model.name,
                azure_model_id,
            )

        # Attempt to retrieve an AzureML Model object which we intend to deploy
        if azure_model_id:
            try:
                registered_model = AzureModel(workspace, id=azure_model_id)
                _logger.info("Found registered model in AzureML with ID '%s'", azure_model_id)
            except Exception as e:
                _logger.info(
                    "Unable to find model in AzureML with ID '%s', will register the model.\n"
                    "Exception was: %s",
                    azure_model_id,
                    e,
                )

        # If we have not found a registered model by this point, we will register it on the users'
        # behalf. It is required for a Model to be registered in some way with Azure in order to
        # deploy to Azure, so this is expected for Azure users.
        if not registered_model:
            registered_model = AzureModel.register(
                workspace=workspace, model_path=tmp_model_path, model_name=model_name, tags=tags
            )

            _logger.info(
                "Registered an Azure Model with name: `%s` and version: `%s`",
                registered_model.name,
                registered_model.version,
            )

        # Create an execution script (entry point) for the image's model server. Azure ML requires
        # the container's execution script to be located in the current working directory during
        # image creation, so we create the execution script as a temporary file in the current
        # working directory.
        execution_script_path = tmp.path("execution_script.py")
        _create_execution_script(output_path=execution_script_path, azure_model=registered_model)

        environment = None
        if pyfunc.ENV in model_pyfunc_conf:
            environment = AzureEnvironment.from_conda_specification(
                _get_mlflow_azure_name(run_id),
                os.path.join(tmp_model_path, model_pyfunc_conf[pyfunc.ENV]),
            )
        else:
            environment = AzureEnvironment(_get_mlflow_azure_name(run_id))

        if mlflow_home is not None:
            path = tmp.path("dist")
            _logger.info("Bulding temporary MLFlow wheel in %s", path)
            wheel = _create_mlflow_wheel(mlflow_home, path)
            whl_url = AzureEnvironment.add_private_pip_wheel(
                workspace=workspace, file_path=wheel, exist_ok=True
            )
            environment.python.conda_dependencies.add_pip_package(whl_url)
        else:
            environment.python.conda_dependencies.add_pip_package(
                "mlflow=={}".format(mlflow_version)
            )

        # AzureML requires azureml-defaults to be installed to include
        # flask for the inference server.
        environment.python.conda_dependencies.add_pip_package(
            "azureml-defaults=={}".format(AZUREML_VERSION)
        )

        inference_config = InferenceConfig(
            entry_script=execution_script_path, environment=environment
        )

        if deployment_config is not None:
            if deployment_config.tags is not None:
                # We want more narrowly-scoped tags to win on merge
                tags.update(deployment_config.tags)
            deployment_config.tags = tags
        else:
            deployment_config = AciWebservice.deploy_configuration(tags=tags)

        # Finally, deploy the AzureML Model object to a webservice, and return back
        webservice = AzureModel.deploy(
            workspace=workspace,
            name=service_name,
            models=[registered_model],
            inference_config=inference_config,
            deployment_config=deployment_config,
            deployment_target=deployment_target
        )
        _logger.info("Deploying an Azure Webservice with name: `%s`", webservice.name)
        if synchronous:
            webservice.wait_for_deployment(show_output=True)
        return webservice, registered_model


def _build_tags(model_uri, model_python_version=None, user_tags=None, run_id=None):
    """
    :param model_uri: URI to the MLflow model.
    :param model_python_version: The version of Python that was used to train the model, if
                                 the model was trained in Python.
    :param user_tags: A collection of user-specified tags to append to the set of default tags.
    """
    tags = dict(user_tags) if user_tags is not None else {}
    tags["model_uri"] = model_uri
    if model_python_version is not None:
        tags["python_version"] = model_python_version
    if run_id is not None:
        tags["mlflow_run_id"] = run_id
    return tags


def _create_execution_script(output_path, azure_model):
    """
    Creates an Azure-compatibele execution script (entry point) for a model server backed by
    the specified model. This script is created as a temporary file in the current working
    directory.
    :param output_path: The path where the execution script will be written.
    :param azure_model: The Azure Model that the execution script will load for inference.
    :return: A reference to the temporary file containing the execution script.
    """
    execution_script_text = SCORE_SRC.format(
        model_name=azure_model.name, model_version=azure_model.version
    )

    with open(output_path, "w") as f:
        f.write(execution_script_text)


def _load_pyfunc_conf_with_model(model_path):
    """
    Loads the `python_function` flavor configuration for the specified model or throws an exception
    if the model does not contain the `python_function` flavor.
    :param model_path: The absolute path to the model.
    :return: The model's `python_function` flavor configuration and the model.
    """
    model_path = os.path.abspath(model_path)
    model = Model.load(os.path.join(model_path, MLMODEL_FILE_NAME))
    if pyfunc.FLAVOR_NAME not in model.flavors:
        raise MlflowException(
            message=(
                "The specified model does not contain the `python_function` flavor. This "
                " flavor is required for model deployment."
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )
    return model.flavors[pyfunc.FLAVOR_NAME], model


def _get_mlflow_azure_name(run_id):
    """
    :return: A unique name for an Azure resource indicating that the resource was created by
             MLflow
    """
    azureml_max_resource_length = 32
    resource_prefix = "mlflow-model-"
    azureml_name = resource_prefix + run_id
    return azureml_name[:azureml_max_resource_length]


def _create_mlflow_wheel(mlflow_dir, out_dir):
    """
    Create the wheel of MLFlow by using setup.py bdist_wheel in the outdir.
    :param mlflow_dir: The absolute path to base of the MLflow Repo to create a wheel from..
    :param out_dir: The absolute path to the outdir.
                    Will be created if it does not exist.
    :return: The absolute path to the wheel.
    """
    unresolved = Path(out_dir)
    unresolved.mkdir(parents=True, exist_ok=True)
    out_path = unresolved.resolve()
    subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel", "-d", out_path], cwd=mlflow_dir, check=True
    )
    files = list(out_path.glob("./*.whl"))
    if len(files) < 1:
        raise MlflowException(
            "Error creating MLFlow Wheel - couldn't"
            " find it in dir {} - found {}".format(out_path, files)
        )
    if len(files) > 1:
        raise MlflowException(
            "Error creating MLFlow Wheel - couldn't"
            " find it in dir {} - found several wheels {}".format(out_path, files)
        )
    return files[0]


SCORE_SRC = """
import pandas as pd
from azureml.core.model import Model
from mlflow.pyfunc import load_model
from mlflow.pyfunc.scoring_server import parse_json_input, _get_jsonable_obj
def init():
    global model
    model_path = Model.get_model_path(model_name="{model_name}", version={model_version})
    model = load_model(model_path)
def run(json_input):
    input_df = parse_json_input(json_input=json_input, orient="split")
    return _get_jsonable_obj(model.predict(input_df), pandas_orient="records")
"""

# COMMAND ----------

from azureml.core.webservice import AksWebservice

prod_webservice_deployment_config = AksWebservice.deploy_configuration( 
                                                                       autoscale_enabled=True, 
                                                                       autoscale_target_utilization=30,
                                                                       autoscale_min_replicas=1,
                                                                       autoscale_max_replicas=4)

#Create an Azure Container Instance webservice for an MLflow model
azure_service, azure_model = deploy2(model_uri=model_version_uri_remote,
                                                   model_name="rfmodel20",
                                                   service_name="rfmodelservice2",
                                                   workspace=workspace3,
                                                   deployment_config=prod_webservice_deployment_config,
                                                   synchronous=True,
                                                   deployment_target=aks_target)

# COMMAND ----------

prod_scoring_uri = azure_service.scoring_uri
prod_service_key = azure_service.get_keys()[0] if len(azure_service.get_keys()) > 0 else None

# COMMAND ----------

prod_scoring_uri

# COMMAND ----------

prod_service_key

# COMMAND ----------

# MAGIC %md ## Create query inputs, in batch

# COMMAND ----------

def get_queries():
  
  sample = df.drop(["price"], axis=1).iloc[:3,:]

  query_input = sample.to_json(orient='split')
  query_input = eval(query_input)
  query_input.pop('index', None)
  return query_input

query_input=get_queries()
query_input

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

prod_prediction = query_endpoint(scoring_uri=prod_scoring_uri, service_key=prod_service_key, inputs=query_input)
prod_prediction
