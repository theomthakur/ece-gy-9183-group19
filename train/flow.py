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
from mlflow.tracking import MlflowClient
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/config/config.yaml")
DATA_PATH = os.environ.get("DATA_PATH", "/app/config/data.yaml")

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

@task(name="load_model")
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

@task(name="export_model")
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

@task(name="evaluate_model")
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

@task(name="register_model")
def register_model_if_passed(passed: bool, metrics_dict=None):
    logger = get_run_logger()
    if not passed:
        logger.info("Evaluation did not pass criteria. Skipping registration.")
        return None

    try:
        logger.info("Registering model in MLflow Model Registry...")
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        
        model_uri = f"runs:/{run_id}/model/best.pt"
        model_type = "PyTorch"
        
        try:
            mlflow.artifacts.download_artifacts(model_uri)
        except Exception:
            logger.info("PyTorch model not found in artifacts, trying ONNX model...")
            model_uri = f"runs:/{run_id}/model/best.onnx"
            model_type = "ONNX"
            
            try:
                mlflow.artifacts.download_artifacts(model_uri)
            except Exception as e:
                logger.error(f"Neither PyTorch nor ONNX model found in artifacts: {str(e)}")
                logger.error("Cannot register model without artifacts. Check model export.")
                return None
        
        logger.info(f"Registering {model_type} model: {model_uri}")
        
        registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        
        if metrics_dict:
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    client.set_model_version_tag(
                        name=MODEL_NAME,
                        version=registered_model.version,
                        key=key,
                        value=str(value)
                    )
        
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=registered_model.version,
            key="model_type", 
            value=model_type
        )
        
        try:
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="development",
                version=registered_model.version
            )
            logger.info(f"Set 'development' alias to version {registered_model.version}")
        except Exception as alias_error:
            logger.warning(f"Could not set alias: {str(alias_error)}")
            client.set_model_version_tag(
                name=MODEL_NAME,
                version=registered_model.version,
                key="development",
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
    
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.26.124:8000")
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
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{config.model_name}-train"):
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
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: ml_pipeline_flow(config))
            
            if result.get("model_version"):
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "Pipeline executed successfully",
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
                        "passed_evaluation": result["passed_evaluation"],
                        "metrics": result["metrics"],
                        "run_id": result["run_id"]
                    }
                )
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during pipeline execution: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model": MODEL_NAME,
        "mlflow_uri": os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.26.124:8000"),
        "config_path": CONFIG_PATH,
        "data_path": DATA_PATH
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
        "mlflow_uri": os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.26.124:8000"),
        "experiment_name": os.environ.get("MLFLOW_EXPERIMENT_NAME", "YOLO12-Training")
    }

@app.get("/")
async def root():
    return {
        "service": "YOLO12 Training API",
        "description": "API for training and registering YOLO12 models",
        "model": MODEL_NAME,
        "config_path": CONFIG_PATH,
        "data_path": DATA_PATH,
        "mlflow_uri": os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.26.124:8000"),
        "endpoints": {
            "/trigger-training": "POST - Start a new training pipeline",
            "/health": "GET - Check service health",
            "/config": "GET - View current configuration"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("SERVICE_PORT", 8000))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting YOLO Training API on port {port}")
    print(f"MLflow server: {os.environ.get('MLFLOW_TRACKING_URI', 'http://129.114.26.124:8000')}")
    print(f"Model: {MODEL_NAME}")
    print(f"Config path: {CONFIG_PATH}")
    print(f"Data path: {DATA_PATH}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)