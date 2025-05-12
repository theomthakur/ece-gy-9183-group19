import os
import argparse
import ray
from datetime import datetime
from pathlib import Path

from ray_setup import setup_ray_cluster, load_config
from ray_trainer import run_distributed_training, create_yolo_trainer
from ray_tune import run_hyperparameter_tuning
from ray import train
from ray.train import ScalingConfig, CheckpointConfig, RunConfig
from ray.tune.integration.mlflow import MLflowLoggerCallback

def main():
    parser = argparse.ArgumentParser(description="YOLO Training with Ray")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--data", type=str, required=True, help="Path to the data.yaml file")
    parser.add_argument("--experiment", type=str, help="MLflow experiment name")
    parser.add_argument("--mode", type=str, choices=["train", "retrain", "tune"], default="train", 
                        help="Training mode: 'train' for new model, 'retrain' to continue training a model, 'tune' for hyperparameter tuning")
    parser.add_argument("--model_path", type=str, help="Path to model weights for retraining (required if mode is 'retrain')")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of trials for tuning")
    parser.add_argument("--max_concurrent", type=int, default=None, help="Max concurrent trials")
    
    args = parser.parse_args()
    
    setup_ray_cluster()
    
    if args.experiment:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment
    
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    config["output_dir"] = "/app/runs"
    
    if args.mode == "retrain" and not args.model_path:
        raise ValueError("--model_path must be specified when mode is 'retrain'")
    
    if args.mode in ["train", "retrain"]:
        print(f"Starting {'re' if args.mode == 'retrain' else ''}training...")
        

        if args.mode == "retrain":
            config["pretrained_weights"] = args.model_path
            print(f"Retraining using model: {args.model_path}")
        
        num_gpus = int(ray.cluster_resources().get("GPU", 0))
        num_workers = max(1, num_gpus)
        
        scaling_config = ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
            resources_per_worker={"CPU": 4, "GPU": 1}
        )
        
        checkpoint_config = CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="map50",
            checkpoint_score_order="max",
        )
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_name = f"{config['model_name']}-{timestamp}"
        
        if args.mode == "retrain":
            run_name = f"retrain-{Path(args.model_path).stem}-{timestamp}"
        
        mlflow_callback = MLflowLoggerCallback(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "YOLO-Ray-Train"),
            save_artifact=True,
        )
        
        run_config = RunConfig(
            name=run_name,
            callbacks=[mlflow_callback],
            storage_path=config.get("output_dir", "./runs"),
        )
        
        result, best_checkpoint = run_distributed_training(
            config, 
            args.data, 
            scaling_config, 
            checkpoint_dir="s3://mlflow-artifacts/ray-checkpoints",
            run_config=run_config
        )
        
        print(f"Training completed: {result.metrics}")
    
    elif args.mode == "tune":
        print("Starting hyperparameter tuning...")
        
        max_concurrent = args.max_concurrent
        if max_concurrent is None:
            num_gpus = int(ray.cluster_resources().get("GPU", 0))
            max_concurrent = max(1, num_gpus)
        
        results, best_config = run_hyperparameter_tuning(
            config,
            args.data,
            num_samples=args.num_samples,
            max_concurrent=max_concurrent
        )
        
        print(f"Tuning completed. Best config: {best_config}")
    
    print("All processes completed successfully!")

if __name__ == "__main__":
    main()
