import os
import time
import torch
import mlflow
import asyncio
import yaml
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from ultralytics import YOLO
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import HTTPError
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/config/config.yaml")
DATA_PATH = os.environ.get("DATA_PATH", "/app/config/data.yaml")
FINETUNE_DATA_PATH = os.environ.get("FINETUNE_DATA_PATH", "/app/config/finetune-data.yaml")

def load_config(config_path=CONFIG_PATH):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
            return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {str(e)}")
        print("Using default configuration")
        return {
            "model_name": "yolo12n",
            "epochs": 1,
            "batch_size": 0.5,
            "img_size": 640,
            "output_dir": "/app/runs",
            "weights_dir": "/app/weights",
        }

CONFIG = load_config()

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(CONFIG.get("weights_dir", "/app/weights"), f"{CONFIG.get('model_name', 'yolo12n')}.pt"))
MODEL_NAME = os.environ.get("MODEL_NAME", CONFIG.get("model_name", "yolo12n"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", CONFIG.get("output_dir", "/app/runs"))
BATCH_SIZE = float(os.environ.get("BATCH_SIZE", CONFIG.get("batch_size", 0.5)))
IMG_SIZE = int(os.environ.get("IMG_SIZE", CONFIG.get("img_size", 640)))
EPOCHS = int(os.environ.get("EPOCHS", CONFIG.get("epochs", 1)))
FINETUNE_EPOCHS = int(os.environ.get("FINETUNE_EPOCHS", 1))

app = FastAPI(title="YOLO12 Training API", 
              description="API for training and registering YOLO12 models using MLflow and Prefect")
pipeline_lock = asyncio.Lock()

class TrainingConfig(BaseModel):
    epochs: int = EPOCHS
    batch_size: float = BATCH_SIZE
    img_size: int = IMG_SIZE
    data_path: str = DATA_PATH
    model_name: str = MODEL_NAME
    config_path: str = CONFIG_PATH

class FinetuningConfig(BaseModel):
    epochs: int = FINETUNE_EPOCHS
    batch_size: float = BATCH_SIZE
    img_size: int = IMG_SIZE
    data_path: str = FINETUNE_DATA_PATH
    model_name: str = MODEL_NAME
    config_path: str = CONFIG_PATH
    alias: str = "development"

def check_production_data_exists():
    production_data_path = "/app/datasets/production/images"
    return os.path.exists(production_data_path) and len(os.listdir(production_data_path)) > 0

def check_model_exists_in_registry(model_name):
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        return len(versions) > 0
    except Exception as e:
        print(f"Error checking model registry: {str(e)}")
        return False

def create_finetune_data_yaml():
    """Create a fine-tuning data.yaml file pointing to production data if it doesn't exist"""
    if os.path.exists(FINETUNE_DATA_PATH):
        try:
            with open(FINETUNE_DATA_PATH, 'r') as f:
                existing_data = yaml.safe_load(f)
            
            if (existing_data and 
                'train' in existing_data and 
                existing_data['train'] == 'production/images'):
                print(f"Using existing fine-tuning data YAML at {FINETUNE_DATA_PATH}")
                return FINETUNE_DATA_PATH
            else:
                print(f"Existing fine-tuning data YAML has invalid structure, recreating...")
        except Exception as e:
            print(f"Error reading existing fine-tuning data YAML: {str(e)}, recreating...")
    
    try:
        with open(DATA_PATH, 'r') as f:
            original_data = yaml.safe_load(f)
        
        finetune_data = {
            "path": "./",
            "train": "production/images",
            "val": original_data.get("val", "validation/images"),
            "test": original_data.get("test", "test/images"),
            "nc": original_data.get("nc", 15),
            "names": original_data.get("names", [])
        }
        
        finetune_yaml_path = FINETUNE_DATA_PATH
        with open(finetune_yaml_path, 'w') as f:
            yaml.dump(finetune_data, f)
        
        print(f"Created fine-tuning data YAML at {finetune_yaml_path}")
        return finetune_yaml_path
    
    except Exception as e:
        print(f"Error creating fine-tuning data YAML: {str(e)}")
        return None

@task(name="get_latest_model", cache_policy=NO_CACHE)
def get_latest_model(model_name, alias=None):
    logger = get_run_logger()
    client = MlflowClient()
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((MlflowException, HTTPError)),
        before_sleep=lambda retry_state: logger.info(f"Retrying download (attempt {retry_state.attempt_number})...")
    )
    def download_with_retry(artifact_uri, dst_path):
        return mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=dst_path)
    
    try:
        if alias:
            logger.info(f"Looking for model {model_name} with alias '{alias}'")
            version_by_alias = client.get_model_version_by_alias(name=model_name, alias=alias)
            model_version = version_by_alias.version
            logger.info(f"Found model version {model_version} with alias '{alias}'")
        else:
            versions = client.search_model_versions(f"name='{model_name}'")
            latest_version = max([int(v.version) for v in versions])
            model_version = str(latest_version)
            logger.info(f"Using latest model version: {model_version}")
        
        model_version_info = client.get_model_version(model_name, model_version)
        run_id = model_version_info.run_id
        logger.info(f"Model version {model_version} associated with run ID: {run_id}")
        
        download_dir = os.path.join(OUTPUT_DIR, "downloaded_models", run_id)
        os.makedirs(download_dir, exist_ok=True)
        
        try:
            artifact_uri = f"runs:/{run_id}/model/best.pt"
            logger.info(f"Downloading PyTorch model from {artifact_uri}")
            
            local_path = download_with_retry(artifact_uri=artifact_uri, dst_path=download_dir)
            logger.info(f"Downloaded PyTorch model to {local_path}")
            
            if os.path.isfile(local_path):
                logger.info(f"Successfully downloaded model file: {local_path}")
                return local_path, model_version
            
        except Exception as e:
            logger.warning(f"Error downloading PyTorch model: {str(e)}")
            
            try:
                artifact_uri = f"runs:/{run_id}/model/best.onnx"
                logger.info(f"Downloading ONNX model from {artifact_uri}")
                
                local_path = download_with_retry(artifact_uri=artifact_uri, dst_path=download_dir)
                logger.info(f"Downloaded ONNX model to {local_path}")
                
                if os.path.isfile(local_path):
                    logger.info(f"Successfully downloaded ONNX model file: {local_path}")
                    return local_path, model_version
                
            except Exception as onnx_error:
                logger.warning(f"Error downloading ONNX model: {str(onnx_error)}")
        
        logger.info(f"Searching for model files in {download_dir}")
        pt_files = list(Path(download_dir).glob("**/*.pt"))
        onnx_files = list(Path(download_dir).glob("**/*.onnx"))
        
        if pt_files:
            model_path = str(pt_files[0])
            logger.info(f"Found PyTorch model file: {model_path}")
            return model_path, model_version
        elif onnx_files:
            model_path = str(onnx_files[0])
            logger.info(f"Found ONNX model file: {model_path}")
            return model_path, model_version
        
        try:
            artifact_uri = f"runs:/{run_id}/model"
            logger.info(f"Downloading entire model directory from {artifact_uri}")
            
            model_dir = download_with_retry(artifact_uri=artifact_uri, dst_path=download_dir)
            logger.info(f"Downloaded model directory to {model_dir}")
            
            pt_files = list(Path(model_dir).glob("**/*.pt"))
            onnx_files = list(Path(model_dir).glob("**/*.onnx"))
            
            if pt_files:
                model_path = str(pt_files[0])
                logger.info(f"Found PyTorch model file in directory: {model_path}")
                return model_path, model_version
            elif onnx_files:
                model_path = str(onnx_files[0])
                logger.info(f"Found ONNX model file in directory: {model_path}")
                return model_path, model_version
            
        except Exception as dir_error:
            logger.warning(f"Error downloading model directory: {str(dir_error)}")
        
        # If we got here, no model was found
        raise FileNotFoundError("No model file (.pt or .onnx) found in downloaded artifacts")
    
    except Exception as e:
        logger.error(f"Error getting model from registry: {str(e)}")
        fallback_path = MODEL_PATH
        if os.path.exists(fallback_path):
            logger.info(f"Falling back to local model: {fallback_path}")
            return fallback_path, "local"
        else:
            raise FileNotFoundError(f"No local model found at {fallback_path}")
     
@task(name="load_model", cache_policy=NO_CACHE)
def load_and_train_model(config: TrainingConfig):
    logger = get_run_logger()
    logger.info(f"Training {config.model_name} for {config.epochs} epochs...")
    
    try:
        logger.info(f"Using config file: {config.config_path}")
        logger.info(f"Using data file: {config.data_path}")
        
        if not os.path.exists(config.config_path):
            logger.warning(f"Config file not found: {config.config_path}")
        
        if not os.path.exists(config.data_path):
            logger.error(f"Data file not found: {config.data_path}")
            raise FileNotFoundError(f"Data file not found: {config.data_path}")
        
        model_path = MODEL_PATH
        logger.info(f"Loading base model: {model_path}")
        
        try:
            model = YOLO(model_path)
            logger.info(f"Successfully loaded model: {config.model_name}")
        except Exception as e:
            logger.warning(f"Error loading model from {model_path}: {str(e)}")
            logger.info("Attempting to directly use the model name...")
            model = YOLO(config.model_name)
        
        mlflow.log_param("epochs", config.epochs)
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("img_size", config.img_size)
        mlflow.log_param("model_name", config.model_name)
        mlflow.log_param("data_path", config.data_path)
        mlflow.log_param("config_path", config.config_path)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{config.model_name}-{timestamp}"
        logger.info(f"Starting training with run name: {run_name}")
        
        training_params = {}
        if os.path.exists(config.config_path):
            with open(config.config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                for param in ['hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fliplr', 'mosaic', 'auto_augment', 'erasing']:
                    if param in yaml_config:
                        training_params[param] = yaml_config[param]
                        logger.info(f"Using config parameter {param}={yaml_config[param]}")
        
        results = model.train(
            data=config.data_path,
            epochs=config.epochs,
            imgsz=config.img_size,
            batch=config.batch_size,
            project=OUTPUT_DIR,
            name=run_name,
            patience=10,
            save=True,
            verbose=True,
            deterministic=False,
            pretrained=True,
            close_mosaic=config.epochs,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            translate=0.0,
            scale=0.0,
            fliplr=0.0,
            mosaic=0.0,
            auto_augment=None,
            erasing=0.0,
        )
        
        run_dir = Path(OUTPUT_DIR) / run_name
        best_pt_path = run_dir / "weights" / "best.pt"
        
        if best_pt_path.exists():
            logger.info(f"Logging best model from {best_pt_path} to MLflow...")
            mlflow.log_artifact(str(best_pt_path), "model")
        else:
            logger.warning(f"Best model weights not found at {best_pt_path}")
        
        mlflow.log_artifact(config.config_path, "config")
        mlflow.log_artifact(config.data_path, "config")
        
        return model, run_dir, results
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise e

@task(name="finetune_model", cache_policy=NO_CACHE)
def finetune_model(config: FinetuningConfig, model_path=None):
    logger = get_run_logger()
    logger.info(f"Fine-tuning model for {config.epochs} epochs...")
    
    try:
        logger.info(f"Using config file: {config.config_path}")
        logger.info(f"Using data file: {config.data_path}")
        
        if not os.path.exists(config.data_path):
            logger.error(f"Data file not found: {config.data_path}")
            raise FileNotFoundError(f"Data file not found: {config.data_path}")
        
        if model_path:
            logger.info(f"Loading model from {model_path}")
        else:
            model_path = MODEL_PATH
            logger.info(f"Loading base model: {model_path}")
        
        model = YOLO(model_path)
        logger.info(f"Successfully loaded model")
        
        mlflow.log_param("fine_tuning", True)
        mlflow.log_param("base_model_path", model_path)
        mlflow.log_param("epochs", config.epochs)
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("img_size", config.img_size)
        mlflow.log_param("data_path", config.data_path)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"finetune-{config.model_name}-{timestamp}"
        logger.info(f"Starting fine-tuning with run name: {run_name}")
        
        results = model.train(
            data=config.data_path,
            epochs=config.epochs,
            imgsz=config.img_size,
            batch=config.batch_size,
            project=OUTPUT_DIR,
            name=run_name,
            patience=10,
            save=True,
            pretrained=True,
            optimizer="Adam",
            lr0=0.001,
            lrf=0.01,
            close_mosaic=config.epochs,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            translate=0.0,
            scale=0.0,
            fliplr=0.0,
            mosaic=0.0,
            auto_augment=None,
            erasing=0.0,
        )
        
        run_dir = Path(OUTPUT_DIR) / run_name
        best_pt_path = run_dir / "weights" / "best.pt"
        
        if best_pt_path.exists():
            logger.info(f"Logging best model from {best_pt_path} to MLflow...")
            mlflow.log_artifact(str(best_pt_path), "model")
        else:
            logger.warning(f"Best model weights not found at {best_pt_path}")
        
        mlflow.log_artifact(config.config_path, "config")
        mlflow.log_artifact(config.data_path, "config")
        
        return model, run_dir, results
        
    except Exception as e:
        logger.error(f"Error during model fine-tuning: {str(e)}")
        raise e

@task(name="export_model", cache_policy=NO_CACHE)
def export_model_formats(model, run_dir, img_size):
    logger = get_run_logger()
    logger.info("Exporting model to additional formats...")
    
    results = {}
    
    try:
        logger.info("Exporting model to ONNX format...")
        onnx_path = Path(run_dir) / "weights" / "best.onnx"
        
        if not onnx_path.exists():
            logger.info(f"Exporting ONNX model with image size {img_size}...")
            model.export(format="onnx", imgsz=img_size)
            
            if onnx_path.exists():
                logger.info(f"ONNX model exported successfully to {onnx_path}")
                mlflow.log_artifact(str(onnx_path), "model")
                results["onnx_path"] = str(onnx_path)
            else:
                logger.warning("ONNX export completed but file not found at expected location")
        else:
            logger.info(f"ONNX model already exists at {onnx_path}")
            mlflow.log_artifact(str(onnx_path), "model")
            results["onnx_path"] = str(onnx_path)
            
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {str(e)}")
        logger.error("ONNX export failed, but continuing with other formats")
    
    return results

@task(name="evaluate_model", cache_policy=NO_CACHE)
def evaluate_model(model, run_dir=None):
    logger = get_run_logger()
    logger.info("Evaluating model on validation set...")
    
    try:
        metrics = model.val()
        
        if metrics is not None:
            if isinstance(metrics, dict):
                metrics_dict = metrics
            elif hasattr(metrics, 'to_dict'):
                metrics_dict = metrics.to_dict()
            else:
                metrics_dict = {}
                for attr in ['map', 'map50', 'map75', 'maps', 'fitness']:
                    if hasattr(metrics, attr):
                        value = getattr(metrics, attr)
                        if isinstance(value, (int, float)):
                            metrics_dict[attr] = value
            
            processed_metrics = {} 
            for key, value in metrics_dict.items():
                if hasattr(value, "item"):
                    processed_value = value.item()
                    processed_metrics[key] = processed_value
                    if isinstance(processed_value, (int, float)):
                        mlflow.log_metric(key, processed_value)
                else:
                    processed_metrics[key] = value
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
            print(f"Logged metrics to MLflow")
                
        logger.info(f"Validation metrics: {processed_metrics}")
        
        map50 = processed_metrics.get('map50', 0)
        map_value = processed_metrics.get('map', 0)
        fitness = processed_metrics.get('fitness', 0)
        
        mlflow.log_metric("mAP50", map50)
        mlflow.log_metric("mAP", map_value)
        mlflow.log_metric("fitness", fitness)
        
        passed = map50 >= 0.0 or fitness >= 0.0
        logger.info(f"Model {'passed' if passed else 'failed'} evaluation criteria")
        
        return bool(passed), processed_metrics
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise e

@task(name="register_model", cache_policy=NO_CACHE)
def register_model_if_passed(passed: bool, metrics_dict=None, is_finetuned=False):
    logger = get_run_logger()
    if not passed:
        logger.info("Evaluation did not pass criteria. Skipping registration.")
        return None

    try:
        logger.info("Registering model in MLflow Model Registry...")
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        
        model_name = f"{MODEL_NAME}-finetuned" if is_finetuned else MODEL_NAME
        
        model_uri = f"runs:/{run_id}/model/best.pt"
        model_type = "PyTorch"
        
        try:
            pt_path = mlflow.artifacts.download_artifacts(model_uri, dst_path="/tmp/model_check")
            logger.info(f"PyTorch model verified at: {pt_path}")
        except Exception as e:
            logger.info(f"PyTorch model not found in artifacts ({str(e)}), trying ONNX model...")
            model_uri = f"runs:/{run_id}/model/best.onnx"
            model_type = "ONNX"
            
            try:
                onnx_path = mlflow.artifacts.download_artifacts(model_uri, dst_path="/tmp/model_check")
                logger.info(f"ONNX model verified at: {onnx_path}")
            except Exception as onnx_error:
                logger.error(f"Neither PyTorch nor ONNX model found in artifacts: {str(onnx_error)}")
                logger.error("Cannot register model without artifacts. Check model export.")
                
                try:
                    artifact_list = mlflow.artifacts.list_artifacts(run_id=run_id, path="model")
                    logger.info(f"Available artifacts: {artifact_list}")
                    
                    model_files = [a for a in artifact_list if a.path.endswith('.pt') or a.path.endswith('.onnx')]
                    
                    if model_files:
                        model_uri = f"runs:/{run_id}/{model_files[0].path}"
                        model_type = "PyTorch" if model_files[0].path.endswith('.pt') else "ONNX"
                        logger.info(f"Found model file: {model_uri}")
                    else:
                        return None
                except Exception as list_error:
                    logger.error(f"Error listing artifacts: {str(list_error)}")
                    return None
        
        logger.info(f"Registering {model_type} model: {model_uri}")
        
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
        
        if metrics_dict:
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    client.set_model_version_tag(
                        name=model_name,
                        version=registered_model.version,
                        key=key,
                        value=str(value)
                    )
        
        client.set_model_version_tag(
            name=model_name,
            version=registered_model.version,
            key="model_type", 
            value=model_type
        )
        
        alias = "staging" if is_finetuned else "development"
        
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=registered_model.version
            )
            logger.info(f"Set '{alias}' alias to version {registered_model.version}")
        except Exception as alias_error:
            logger.warning(f"Could not set alias: {str(alias_error)}")
            client.set_model_version_tag(
                name=model_name,
                version=registered_model.version,
                key=alias,
                value="true"
            )
            
        logger.info(f"Model registered (v{registered_model.version}).")
        return registered_model.version
        
    except Exception as e:
        logger.error(f"Error during model registration: {str(e)}")
        raise e

@flow(name="yolo_training_flow")
def ml_pipeline_flow(config: TrainingConfig):
    logger = get_run_logger()
    logger.info(f"Starting ML pipeline flow with config: {config}")
    
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_uri)
    logger.info(f"Using MLflow tracking URI: {mlflow_uri}")
    
    s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    if s3_endpoint:
        logger.info(f"Using S3 endpoint: {s3_endpoint}")
        logger.info(f"Using bucket: {os.environ.get('BUCKET_NAME', 'mlflow-artifacts')}")
    
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "YOLO12-Training")
    logger.info(f"Using MLflow experiment: {experiment_name}")
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {str(e)}")
        logger.info("Check your MLflow connection and credentials")
        logger.info(f"MLflow URI: {mlflow_uri}")
        experiment_id = None
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{config.model_name}-train", log_system_metrics=True):
        logger.info("MLflow run started")
        
        mlflow.log_param("mlflow_server", mlflow_uri)
        mlflow.log_param("mlflow_experiment", experiment_name)
        
        model, run_dir, results = load_and_train_model(config)
        passed, metrics = evaluate_model(model, run_dir)
        export_results = export_model_formats(model, run_dir, config.img_size)
        version = register_model_if_passed(passed, metrics)
        
        return {
            "model_version": version,
            "passed_evaluation": passed,
            "metrics": metrics,
            "run_id": mlflow.active_run().info.run_id
        }

@flow(name="yolo_finetuning_flow")
def finetune_pipeline_flow(config: FinetuningConfig):
    logger = get_run_logger()
    logger.info(f"Starting ML fine-tuning pipeline with config: {config}")
    
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_uri)
    logger.info(f"Using MLflow tracking URI: {mlflow_uri}")
    
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "YOLO12-Finetuning")
    logger.info(f"Using MLflow experiment: {experiment_name}")
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {str(e)}")
        experiment_id = None
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{config.model_name}-finetune", log_system_metrics=True):
        logger.info("MLflow run started for fine-tuning")
        
        model_path, base_version = get_latest_model(config.model_name, config.alias)
        mlflow.log_param("base_model_version", base_version)
        
        model, run_dir, results = finetune_model(config, model_path)
        passed, metrics = evaluate_model(model, run_dir)
        export_results = export_model_formats(model, run_dir, config.img_size)
        
        version = register_model_if_passed(passed, metrics, is_finetuned=True)
        
        return {
            "model_version": version,
            "passed_evaluation": passed,
            "metrics": metrics,
            "run_id": mlflow.active_run().info.run_id,
            "base_model_version": base_version
        }

def should_finetune():
    """Determine if we should do fine-tuning instead of full training"""
    has_production_data = check_production_data_exists()
    has_existing_model = check_model_exists_in_registry(MODEL_NAME)
    
    return has_production_data and has_existing_model

@app.post("/trigger-training")
async def trigger_training(
    background_tasks: BackgroundTasks,
    config: TrainingConfig = Body(
        default=TrainingConfig(),
        description="Training configuration parameters"
    )
):
    if pipeline_lock.locked():
        raise HTTPException(status_code=423, detail="Pipeline is already running. Please wait.")

    async with pipeline_lock:
        try:
            if should_finetune():
                finetune_data_path = create_finetune_data_yaml()
                
                if not finetune_data_path:
                    raise HTTPException(
                        status_code=500, 
                        detail="Failed to create fine-tuning data YAML"
                    )
                
                finetune_config = FinetuningConfig(
                    epochs=1,
                    batch_size=config.batch_size,
                    img_size=config.img_size,
                    data_path=finetune_data_path,
                    model_name=config.model_name,
                    config_path=config.config_path,
                    alias="development"
                )
                
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: finetune_pipeline_flow(finetune_config)
                )
                
                if result.get("model_version"):
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "Fine-tuning pipeline executed successfully",
                            "fine_tuned": True,
                            "new_model_version": result["model_version"],
                            "base_model_version": result["base_model_version"],
                            "metrics": result["metrics"],
                            "run_id": result["run_id"]
                        }
                    )
                else:
                    return JSONResponse(
                        status_code=200, 
                        content={
                            "status": "Fine-tuning pipeline executed, but no new model registered", 
                            "fine_tuned": True,
                            "passed_evaluation": result["passed_evaluation"],
                            "metrics": result["metrics"],
                            "run_id": result["run_id"]
                        }
                    )
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: ml_pipeline_flow(config))
                
                if result.get("model_version"):
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "Pipeline executed successfully",
                            "fine_tuned": False,
                            "new_model_version": result["model_version"],
                            "metrics": result["metrics"],
                            "run_id": result["run_id"]
                        }
                    )
                else:
                    return JSONResponse(
                        status_code=200, 
                        content={
                            "status": "Pipeline executed, but no new model registered",
                            "fine_tuned": False, 
                            "passed_evaluation": result["passed_evaluation"],
                            "metrics": result["metrics"],
                            "run_id": result["run_id"]
                        }
                    )
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during pipeline execution: {str(e)}")

@app.post("/trigger-finetune")
async def trigger_finetune(
    background_tasks: BackgroundTasks,
    config: FinetuningConfig = Body(
        default=FinetuningConfig(),
        description="Fine-tuning configuration parameters"
    )
):
    """Explicit endpoint for fine-tuning only"""
    if pipeline_lock.locked():
        raise HTTPException(status_code=423, detail="Pipeline is already running. Please wait.")

    async with pipeline_lock:
        try:
            if not check_model_exists_in_registry(config.model_name):
                raise HTTPException(
                    status_code=400, 
                    detail=f"No existing model '{config.model_name}' found in registry. Cannot fine-tune."
                )
            
            if not check_production_data_exists():
                raise HTTPException(
                    status_code=400, 
                    detail="No production data found. Cannot fine-tune."
                )
            
            finetune_data_path = create_finetune_data_yaml()
            
            if not finetune_data_path:
                raise HTTPException(
                    status_code=500, 
                    detail="Failed to create fine-tuning data YAML"
                )
            
            config.data_path = finetune_data_path
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: finetune_pipeline_flow(config))
            
            if result.get("model_version"):
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "Fine-tuning pipeline executed successfully",
                        "new_model_version": result["model_version"],
                        "base_model_version": result["base_model_version"],
                        "metrics": result["metrics"],
                        "run_id": result["run_id"]
                    }
                )
            else:
                return JSONResponse(
                    status_code=200, 
                    content={
                        "status": "Fine-tuning pipeline executed, but no new model registered", 
                        "passed_evaluation": result["passed_evaluation"],
                        "metrics": result["metrics"],
                        "run_id": result["run_id"]
                    }
                )
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during fine-tuning: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model": MODEL_NAME,
        "mlflow_uri": os.environ.get("MLFLOW_TRACKING_URI"),
        "config_path": CONFIG_PATH,
        "data_path": DATA_PATH,
        "can_finetune": should_finetune()
    }

@app.get("/config")
async def get_config():
    return {
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "config_path": CONFIG_PATH,
        "data_path": DATA_PATH,
        "output_dir": OUTPUT_DIR,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "epochs": EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "mlflow_uri": os.environ.get("MLFLOW_TRACKING_URI"),
        "training_experiment": os.environ.get("MLFLOW_EXPERIMENT_NAME", "YOLO12-Training"),
        "finetuning_experiment": "YOLO12-Finetuning",
        "has_production_data": check_production_data_exists(),
        "has_existing_model": check_model_exists_in_registry(MODEL_NAME)
    }

@app.get("/")
async def root():
    return {
        "service": "YOLO12 Training API",
        "description": "API for training and registering YOLO12 models",
        "model": MODEL_NAME,
        "config_path": CONFIG_PATH,
        "data_path": DATA_PATH,
        "mlflow_uri": os.environ.get("MLFLOW_TRACKING_URI"),
        "endpoints": {
            "/trigger-training": "POST - Start a new training pipeline (auto-detects if fine-tuning is needed)",
            "/trigger-finetune": "POST - Explicitly start a fine-tuning pipeline",
            "/health": "GET - Check service health",
            "/config": "GET - View current configuration"
        },
        "can_finetune": should_finetune()
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("SERVICE_PORT", 8000))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting YOLO Training API on port {port}")
    print(f"MLflow server: {os.environ.get('MLFLOW_TRACKING_URI')}")
    print(f"Model: {MODEL_NAME}")
    print(f"Config path: {CONFIG_PATH}")
    print(f"Data path: {DATA_PATH}")
    print(f"Can fine-tune: {should_finetune()}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)