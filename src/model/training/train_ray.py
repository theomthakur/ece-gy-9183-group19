import os
import sys
import argparse
import mlflow
import time
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

import ray
from ray import train
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer
from ray.train.torch import TorchConfig

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.yolo import load_dataset, download_yolo_weights

def train_func(config_dict: Dict[str, Any]):
    """
    Training function for Ray Train.
    This function runs on each worker.
    
    Args:
        config_dict: Dictionary with configuration
    """
    # Import locally to avoid issues with Ray serialization
    from ultralytics import YOLO
    import torch
    
    # Get the rank of the current worker
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    
    print(f"Starting worker {rank} of {world_size}")
    
    # Create a local config object
    config = Config()
    config.update(config_dict)
    
    # Setup device - Ray Train already sets CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available() and config['use_gpu']:
        device = f"cuda:{0}"  # Use the first GPU assigned by Ray
        print(f"Worker {rank} using {device}")
    else:
        device = "cpu"
        print(f"Worker {rank} using CPU")
    
    # Load YOLO model
    print(f"Worker {rank} loading model: {config['model_name']}")
    model_path = config['model_name']
    if rank == 0:  # Only the first worker needs to download weights
        model_path = download_yolo_weights(config['model_name'], config['weights_dir'])
    
    # Wait for rank 0 to download the model if needed
    if world_size > 1:
        ray.train.report({"status": "waiting_for_model"})
        time.sleep(5)  # Small delay to ensure model is downloaded
    
    try:
        model = YOLO(model_path)
        print(f"Worker {rank} successfully loaded model")
    except Exception as e:
        print(f"Worker {rank} error loading model: {e}")
        print("Attempting to directly use the model name...")
        model = YOLO(config['model_name'])
    
    # Get data path
    yaml_path = config['yaml_path']
    
    # Check for existing checkpoint
    checkpoint = train.get_checkpoint()
    starting_epoch = 0
    
    if checkpoint:
        checkpoint_dir = checkpoint.path
        print(f"Worker {rank} loading checkpoint from {checkpoint_dir}")
        
        # Find the best checkpoint file
        checkpoint_files = list(Path(checkpoint_dir).glob("*.pt"))
        if checkpoint_files:
            best_checkpoint = str(checkpoint_files[0])
            model = YOLO(best_checkpoint)
            
            # Try to extract the epoch from checkpoint filename
            try:
                filename = Path(best_checkpoint).stem
                if "_" in filename:
                    starting_epoch = int(filename.split("_")[1])
                    print(f"Worker {rank} resuming from epoch {starting_epoch}")
            except:
                pass
    
    # Start training
    print(f"Worker {rank} starting training from epoch {starting_epoch}")
    
    # Calculate remaining epochs
    epochs = config['epochs'] - starting_epoch
    
    try:
        # Train the model
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=config['img_size'],
            batch=config['batch_size'],
            device=device,
            workers=config['workers'],
            project=config['output_dir'],
            name=f'worker_{rank}',
            deterministic=True,  # Set to True for reproducibility
            pretrained=config['pretrained'],
            patience=config['patience'],
            save=True,
            save_period=config['save_freq'],  # Save model every N epochs
            exist_ok=True,
            resume=starting_epoch > 0,
            
            # Augmentation parameters
            hsv_h=config['hsv_h'],
            hsv_s=config['hsv_s'],
            hsv_v=config['hsv_v'],
            translate=config['translate'],
            scale=config['scale'],
            fliplr=config['fliplr'],
            mosaic=config['mosaic'],
            erasing=config['erasing'],
        )
        
        # Extract metrics
        metrics = {}
        
        # Add training metrics
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                metrics[key.replace('/', '_')] = value
        
        # Evaluate the model on validation set
        if rank == 0:  # Only worker 0 needs to evaluate
            print(f"Worker {rank} running validation...")
            val_results = model.val(data=yaml_path)
            
            # Add validation metrics
            for key, value in val_results.results_dict.items():
                if isinstance(value, (int, float)):
                    metrics[f"val_{key.replace('/', '_')}"] = value
        
        # Get the best model checkpoint path
        best_model_path = os.path.join(config['output_dir'], f'worker_{rank}', 'weights', 'best.pt')
        
        # Only save checkpoint from worker 0 in distributed mode, or from any worker in single-worker mode
        if rank == 0 or world_size == 1:
            if os.path.exists(best_model_path):
                # Create a checkpoint directory
                checkpoint_dir = os.path.join(config['checkpoint_dir'], f"epoch_{config['epochs']}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Copy the checkpoint
                import shutil
                checkpoint_file = os.path.join(checkpoint_dir, f"best_{rank}.pt")
                shutil.copy(best_model_path, checkpoint_file)
                
                # Save the checkpoint
                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(metrics, checkpoint=checkpoint)
                print(f"Worker {rank} saved checkpoint and reported metrics")
            else:
                # If no best model was saved, just report metrics
                train.report(metrics)
                print(f"Worker {rank} reported metrics (no checkpoint)")
        else:
            # Other workers just report metrics
            train.report(metrics)
            print(f"Worker {rank} reported metrics")
    
    except Exception as e:
        print(f"Worker {rank} error during training: {e}")
        train.report({"error": str(e)})

def setup_ray_cluster(config: Config):
    """
    Connect to Ray cluster or start a local Ray instance.
    
    Args:
        config: Configuration object
    """
    # Check if Ray is already initialized
    if not ray.is_initialized():
        if config['ray_address'] == 'local':
            # Start a local Ray instance
            ray.init(include_dashboard=True)
            print("Started local Ray instance")
        elif config['ray_address'] == 'auto':
            # Automatically detect Ray cluster
            ray.init()
            print("Connected to Ray cluster")
        else:
            # Connect to a specific Ray address
            ray.init(address=config['ray_address'])
            print(f"Connected to Ray cluster at {config['ray_address']}")

def setup_mlflow(config: Config):
    """
    Setup MLflow tracking.
    
    Args:
        config: Configuration object
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(config['experiment_name'])
        if experiment is None:
            experiment_id = mlflow.create_experiment(config['experiment_name'])
            print(f"Created new experiment '{config['experiment_name']}' with ID: {experiment_id}")
        else:
            print(f"Using existing experiment '{config['experiment_name']}' with ID: {experiment.experiment_id}")
            mlflow.set_experiment(config['experiment_name'])
    except Exception as e:
        print(f"Error setting up MLflow: {e}")
        print("Continuing without MLflow tracking...")

def train_with_ray(config: Config):
    """
    Train model using Ray.
    
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
    
    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # Log parameters
        mlflow.log_params(config.get_mlflow_params())
        
        # Additional parameters to log
        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_param("classes", json.dumps(class_names))
        
        # Log dataset info
        with open(yaml_path, 'r') as f:
            dataset_info = yaml.safe_load(f)
            mlflow.log_param("dataset_info", json.dumps(dataset_info))
        
        # Setup Ray Train for distributed training
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=config_dict,
            scaling_config={
                "num_workers": config['num_workers'],
                "use_gpu": config['use_gpu'],
                "resources_per_worker": {
                    "cpu": config['cpus_per_worker'],
                    "gpu": config['gpus_per_worker'] if config['use_gpu'] else 0
                },
            },
            run_config=train.RunConfig(
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=2,
                    checkpoint_frequency=config['save_freq'],
                    checkpoint_at_end=True
                ),
                storage_path=config['checkpoint_dir'],
                name="yolo_training"
            )
        )
        
        # Start training
        print(f"Starting Ray training with {config['num_workers']} workers...")
        start_time = time.time()
        
        result = trainer.fit()
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Process results
        metrics = result.metrics
        print("Training results:", metrics)
        
        # Log metrics to MLflow
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        
        # Log final training time
        mlflow.log_metric("training_time", training_time)
        
        # Get the best checkpoint
        best_checkpoint = result.checkpoint
        if best_checkpoint:
            # Log checkpoint to MLflow
            checkpoint_path = best_checkpoint.path
            print(f"Best checkpoint: {checkpoint_path}")
            
            # Find the best model file
            best_model_files = list(Path(checkpoint_path).glob("*.pt"))
            if best_model_files:
                best_model = str(best_model_files[0])
                mlflow.log_artifact(best_model, "model")
                print(f"Logged best model to MLflow: {best_model}")
                
                # Copy the best model to a standard location
                import shutil
                final_model_path = os.path.join(config['output_dir'], "best.pt")
                shutil.copy(best_model, final_model_path)
                print(f"Copied best model to {final_model_path}")
        
        print(f"All metrics and artifacts logged to MLflow run: {run_id}")
        
        return metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO model with Ray and MLflow")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--ray_address', type=str, help='Ray cluster address (local, auto, or specific address)')
    parser.add_argument('--num_workers', type=int, help='Number of Ray workers')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--cpus_per_worker', type=int, help='CPUs per Ray worker')
    parser.add_argument('--gpus_per_worker', type=float, help='GPUs per Ray worker')
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
    
    # Train model
    metrics = train_with_ray(config)
    
    # Print summary
    print("\nTraining Summary:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()