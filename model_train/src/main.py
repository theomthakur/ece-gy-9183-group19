import os
import sys
from pathlib import Path
import yaml
import argparse
from dotenv import load_dotenv

from model_trainer import download_yolo_weights, main

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO12 Fine-tuning with Docker")
    parser.add_argument(
        "--config", 
        type=str, 
        default="/app/config/config.yaml", 
        help="Path to the config file"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="/app/config/data.yaml", 
        help="Path to the data.yaml file"
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--run_name", 
        type=str, 
        help="Optional custom run name (default is modelname-timestamp)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
        
    if not os.path.exists(args.data):
        print(f"Error: Data YAML file not found at {args.data}")
        sys.exit(1)
    
    if args.experiment:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment
        print(f"Using experiment name: {args.experiment}")
    
    if args.run_name:
        os.environ["CUSTOM_RUN_NAME"] = args.run_name
        print(f"Using custom run name: {args.run_name}")
    
    print(f"MLflow Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI', 'Not set')}")
    print(f"MinIO Endpoint: {os.environ.get('MLFLOW_S3_ENDPOINT_URL', 'Not set')}")
    print(f"MinIO Bucket: {os.environ.get('BUCKET_NAME', 'mlflow-artifacts')}")
    
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        print(f"Successfully connected to MLflow. Found {len(experiments)} experiments.")
    except Exception as e:
        print(f"Warning: Could not connect to MLflow: {e}")
        print("Check your network connection and credentials.")
    
    main(args.config, args.data)