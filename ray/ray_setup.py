import os
import yaml
import time
import argparse
from pathlib import Path
import ray
from ray import tune
from ray.train import ScalingConfig, CheckpointConfig, RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint
from ray.tune.integration.mlflow import MLflowLoggerCallback
from dotenv import load_dotenv

load_dotenv()

def setup_ray_cluster():
    try:
        ray.init(address="ray://ray-head:6379", ignore_reinit_error=True)
        print("Connected to existing Ray cluster")
    except Exception as e:
        print(f"Could not connect to Ray cluster: {e}")
        print("Initializing Ray locally")
        ray.init(ignore_reinit_error=True)
    
    print("Available resources:")
    print(ray.cluster_resources())

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def create_scaling_config(num_workers=None):
    if num_workers is None:
        num_gpus = int(ray.cluster_resources().get("GPU", 0))
        num_workers = max(1, num_gpus)
    
    return ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
        resources_per_worker={
            "CPU": 4,
            "GPU": 1,
        }
    )

def create_checkpoint_config():
    return CheckpointConfig(
        num_to_keep=5,
        checkpoint_score_attribute="map50",
        checkpoint_score_order="max",
    )

def create_tune_config(config, num_samples=10):
    param_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([8, 16, 32, 64]),
        "img_size": tune.choice([512, 640, 768]),
    }
    
    scheduler = ASHAScheduler(
        max_t=config["epochs"],
        grace_period=5,
        reduction_factor=2,
    )
    
    return param_space, scheduler, num_samples

def setup_mlflow_callback():
    return MLflowLoggerCallback(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "YOLO12-RayTune"),
        save_artifact=True,
    )

def parse_args():

    parser = argparse.ArgumentParser(description="YOLO Training with Ray")
    parser.add_argument("--config", type=str, default="/app/config/config.yaml", 
                        help="Path to the config file")
    parser.add_argument("--data", type=str, default="/app/config/data.yaml", 
                        help="Path to the data.yaml file")
    parser.add_argument("--experiment", type=str, help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, help="Custom run name")
    parser.add_argument("--num_workers", type=int, default=None, 
                        help="Number of Ray workers to use")
    parser.add_argument("--num_samples", type=int, default=10, 
                        help="Number of hyperparameter trials")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup_ray_cluster()
    
    print(f"Ray initialized. Dashboard URL: http://localhost:8265")
    
    config = load_config(args.config)
    
    if args.experiment:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment
    
    print(f"MLflow Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI', 'Not set')}")
    print(f"MinIO Endpoint: {os.environ.get('AWS_ENDPOINT_URL', 'Not set')}")
    print("Ray cluster setup complete")