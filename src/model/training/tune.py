import os
import sys
import argparse
import mlflow
import json
import yaml
import time
from pathlib import Path
from typing import Dict, Any, Optional
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.train import RunConfig, ScalingConfig, CheckpointConfig

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.data_processing import load_dataset, download_yolo_weights
from src.train_ray import setup_ray_cluster, train_func, setup_mlflow

def train_objective(config_dict: Dict[str, Any]):
    """
    Objective function for Ray Tune hyperparameter optimization.
    
    Args:
        config_dict: Dictionary with configuration
    """
    # This is needed to avoid circular imports
    from ray import train
    
    # Run the training function
    train_func(config_dict)

def tune_hyperparameters(config: Config):
    """
    Tune hyperparameters with Ray Tune.
    
    Args:
        config: Configuration object
    """
    # Connect to Ray cluster
    setup_ray_cluster(config)
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Prepare dataset (only once, not on each worker)
    yaml_path, class_names = load_dataset(
        config['data_csv'],
        config['images_dir'], 
        config['yolo_data_dir'],
        config['val_split'],
        config['seed']
    )
    
    # Add yaml_path to config for workers
    config['yaml_path'] = yaml_path
    
    # Convert Config object to dictionary for Ray
    config_dict = config.as_dict()
    
    # Define search space for hyperparameters
    search_space = {
        # Learning rate (log scale)
        "lr0": tune.loguniform(1e-5, 1e-2),
        
        # Batch size
        "batch_size": tune.choice([8, 16, 32, 64]),
        
        # Augmentation parameters
        "hsv_h": tune.uniform(0.0, 0.3),
        "hsv_s": tune.uniform(0.0, 0.7),
        "hsv_v": tune.uniform(0.0, 0.5),
        "translate": tune.uniform(0.0, 0.5),
        "scale": tune.uniform(0.0, 0.5),
        "fliplr": tune.uniform(0.0, 0.5),
        "mosaic": tune.uniform(0.0, 1.0),
        
        # Fixed parameters (not tuned)
        "epochs": config['epochs'],
        "img_size": config['img_size'],
        "model_name": config['model_name'],
        "pretrained": config['pretrained'],
        "patience": config['patience'],
        "yaml_path": yaml_path,
        "output_dir": config['output_dir'],
        "workers": config['workers'],
        "device": config['device'],
        "save_freq": config['save_freq'],
        "use_gpu": config['use_gpu']
    }
    
    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # Log base parameters
        mlflow.log_params({
            k: v for k, v in config.get_mlflow_params().items() 
            if k not in search_space
        })
        
        # Log dataset info
        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_param("classes", json.dumps(class_names))
        
        # Log dataset info
        with open(yaml_path, 'r') as f:
            dataset_info = yaml.safe_load(f)
            mlflow.log_param("dataset_info", json.dumps(dataset_info))
        
        # Define search algorithm
        search_alg = OptunaSearch()
        search_alg = ConcurrencyLimiter(
            search_alg, max_concurrent=config['num_workers']
        )
        
        # Define scheduler
        scheduler = ASHAScheduler(
            metric="val_metrics_fitness",
            mode="max",
            max_t=config['epochs'],
            grace_period=5,  # Minimum number of epochs before early stopping
            reduction_factor=2
        )
        
        # Set up the tuner
        resources_per_trial = {
            "cpu": config['cpus_per_worker'],
            "gpu": config['gpus_per_worker'] if config['use_gpu'] else 0
        }
        
        # Create a callback to log to MLflow
        def mlflow_callback(trial_id: str, result: Dict[str, Any]):
            # Create a nested run for each trial
            with mlflow.start_run(run_id=run_id, nested=True) as nested_run:
                nested_run_id = nested_run.info.run_id
                
                # Log parameters
                params = {k: v for k, v in result.items() 
                          if k in search_space and not k.startswith("config/")}
                mlflow.log_params(params)
                
                # Log metrics
                metrics = {k: v for k, v in result.items() 
                          if isinstance(v, (int, float)) and not k.startswith("config/")}
                mlflow.log_metrics(metrics)
                
                print(f"Logged trial {trial_id} to MLflow run {nested_run_id}")
        
        # Run the hyperparameter tuning
        print(f"Starting hyperparameter tuning with Ray Tune...")
        start_time = time.time()
        
        tuner = tune.Tuner(
            train_objective,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=config['tune_trials'],
                metric="val_metrics_fitness",
                mode="max",
                max_concurrent_trials=config['num_workers'],
                reuse_actors=False
            ),
            run_config=RunConfig(
                name="yolo_tuning",
                storage_path=os.path.join(config['output_dir'], "tune"),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=2,
                    checkpoint_frequency=config['save_freq'],
                    checkpoint_at_end=True
                ),
                callbacks=[mlflow_callback]
            )
        )
        
        results = tuner.fit()
        
        tuning_time = time.time() - start_time
        print(f"Tuning completed in {tuning_time:.2f} seconds")
        
        # Get best trial
        best_trial = results.get_best_result(
            metric="val_metrics_fitness", mode="max"
        )
        
        print("Best hyperparameters found:")
        for param, value in best_trial.config.items():
            if param in search_space and param not in config_dict:
                print(f"  {param}: {value}")
        
        print(f"Best value for val_metrics_fitness: {best_trial.metrics['val_metrics_fitness']}")
        
        # Log best metrics to parent run
        mlflow.log_metric("best_val_fitness", best_trial.metrics['val_metrics_fitness'])
        mlflow.log_metric("tuning_time", tuning_time)
        
        # Log best hyperparameters to parent run
        for param, value in best_trial.config.items():
            if param in search_space and param not in config_dict:
                mlflow.log_param(f"best_{param}", value)
        
        # Get the best checkpoint
        best_checkpoint = best_trial.checkpoint
        if best_checkpoint:
            # Log checkpoint to MLflow
            checkpoint_path = best_checkpoint.path
            print(f"Best checkpoint: {checkpoint_path}")
            
            # Find the best model file
            best_model_files = list(Path(checkpoint_path).glob("*.pt"))
            if best_model_files:
                best_model = str(best_model_files[0])
                mlflow.log_artifact(best_model, "best_model")
                print(f"Logged best model to MLflow: {best_model}")
                
                # Copy the best model to a standard location
                import shutil
                final_model_path = os.path.join(config['output_dir'], "best_tuned.pt")
                shutil.copy(best_model, final_model_path)
                print(f"Copied best model to {final_model_path}")
        
        print(f"All metrics and artifacts logged to MLflow run: {run_id}")
        
        return best_trial.config, best_trial.metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tune YOLO hyperparameters with Ray Tune")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ray_address', type=str, help='Ray cluster address (local, auto, or specific address)')
    parser.add_argument('--num_workers', type=int, help='Number of Ray workers')
    parser.add_argument('--tune_trials', type=int, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--mlflow_tracking_uri', type=str, help='MLflow tracking URI')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Initialize configuration
    config = Config(args.config)
    
    # Update configuration with command line arguments
    arg_dict = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
    if arg_dict:
        config.update(arg_dict)
    
    # Tune hyperparameters
    best_config, best_metrics = tune_hyperparameters(config)
    
    # Print summary
    print("\nTuning Summary:")
    print("Best hyperparameters:")
    for param, value in best_config.items():
        print(f"  {param}: {value}")
    
    print("\nBest metrics:")
    for metric, value in best_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()