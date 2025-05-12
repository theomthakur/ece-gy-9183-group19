#!/usr/bin/env python3
"""
Startup script that fetches the latest model from MLflow and sets up Triton
before starting the Triton server.
"""
import os
import sys
import logging
import subprocess
import shutil
from pathlib import Path
import mlflow
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException
from requests.exceptions import HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - set these as environment variables in docker-compose
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://your-mlflow-ip:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "chest_xray_detector")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", None)  # Set to None to use latest version
TRITON_MODEL_DIR = "/models"  # Inside container

def get_model_from_mlflow():
    """Fetch the latest model from MLflow"""
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((MlflowException, HTTPError)),
        before_sleep=lambda retry_state: logger.info(f"Retrying download (attempt {retry_state.attempt_number})...")
    )
    def download_with_retry(artifact_uri, dst_path):
        return mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=dst_path)
    
    try:
        # Set up MLflow tracking
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"Connected to MLflow at {MLFLOW_TRACKING_URI}")
        
        # Get model version (either by alias or latest)
        if MODEL_ALIAS:
            logger.info(f"Looking for model {MODEL_NAME} with alias '{MODEL_ALIAS}'")
            version_by_alias = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
            model_version = version_by_alias.version
            logger.info(f"Found model version {model_version} with alias '{MODEL_ALIAS}'")
        else:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if not versions:
                raise ValueError(f"No versions found for model '{MODEL_NAME}'")
            latest_version = max([int(v.version) for v in versions])
            model_version = str(latest_version)
            logger.info(f"Using latest model version: {model_version}")
        
        # Get run ID associated with the model version
        model_version_info = client.get_model_version(MODEL_NAME, model_version)
        run_id = model_version_info.run_id
        logger.info(f"Model version {model_version} associated with run ID: {run_id}")
        
        # Create temporary download directory
        download_dir = "/tmp/downloaded_models"
        os.makedirs(download_dir, exist_ok=True)
        
        # Try to download ONNX model directly
        try:
            artifact_uri = f"runs:/{run_id}/model/best.onnx"
            logger.info(f"Downloading ONNX model from {artifact_uri}")
            local_path = download_with_retry(artifact_uri=artifact_uri, dst_path=download_dir)
            
            if os.path.isfile(local_path):
                logger.info(f"Successfully downloaded ONNX model file: {local_path}")
                return local_path, model_version
        except Exception as e:
            logger.warning(f"Error downloading specific ONNX model: {str(e)}")
        
        # If direct download failed, try downloading the model directory
        try:
            artifact_uri = f"runs:/{run_id}/model"
            logger.info(f"Downloading model directory from {artifact_uri}")
            model_dir = download_with_retry(artifact_uri=artifact_uri, dst_path=download_dir)
            
            # Search for ONNX files
            onnx_files = list(Path(model_dir).glob("**/*.onnx"))
            if onnx_files:
                model_path = str(onnx_files[0])
                logger.info(f"Found ONNX model file: {model_path}")
                return model_path, model_version
            else:
                logger.warning("No ONNX file found in model directory")
        except Exception as e:
            logger.warning(f"Error downloading model directory: {str(e)}")
        
        # If we got here, no ONNX model was found
        raise FileNotFoundError("No ONNX model file found in MLflow artifacts")
    
    except Exception as e:
        logger.error(f"Error getting model from MLflow: {str(e)}")
        raise

def setup_triton_model():
    """Set up the Triton model repository with the latest model from MLflow"""
    try:
        # Get model from MLflow
        model_path, model_version = get_model_from_mlflow()
        logger.info(f"Retrieved model from MLflow: {model_path} (version {model_version})")
        
        # Setup model directory structure for specific model
        model_triton_dir = os.path.join(TRITON_MODEL_DIR, MODEL_NAME, "1")
        os.makedirs(model_triton_dir, exist_ok=True)
        
        # Copy the model file with the expected name
        triton_model_path = os.path.join(model_triton_dir, "model.onnx")
        shutil.copy(model_path, triton_model_path)
        logger.info(f"Copied model to Triton directory: {triton_model_path}")
        
        # Check if config.pbtxt exists at the model level, if not create it
        config_path = os.path.join(TRITON_MODEL_DIR, MODEL_NAME, "config.pbtxt")
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                f.write(f'''name: "{MODEL_NAME}"
backend: "openvino"
default_model_filename: "model.onnx"
max_batch_size: 1
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [3, 640, 640]
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [19, 8400]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
''')
            logger.info(f"Created config.pbtxt at {config_path}")
        
        logger.info("Triton model setup completed successfully")
        return triton_model_path
    
    except Exception as e:
        logger.error(f"Failed to set up Triton model: {str(e)}")
        raise

def main():
    """Main function to setup model and start Triton server"""
    try:
        logger.info("Starting MLflow to Triton integration")
        triton_model_path = setup_triton_model()
        logger.info(f"Successfully set up Triton model at {triton_model_path}")
        
        # Start Triton server with the updated model repository
        logger.info("Starting Triton server...")
        triton_cmd = ["tritonserver", "--model-repository=/models"]
        subprocess.run(triton_cmd)
    except Exception as e:
        logger.error(f"Integration failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()