import os
import sys
import argparse
import mlflow
import mlflow.pytorch
from pathlib import Path
from ultralytics import YOLO
import yaml
import json
import time
from typing import Dict, Any, Optional

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.data_processing import load_dataset, download_yolo_weights

def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """
    Setup MLflow tracking.
    
    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: MLflow experiment name
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
        else:
            print(f"Using existing experiment '{experiment_name}' with ID: {experiment.experiment_id}")
            mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Error setting up MLflow: {e}")
        print("Continuing without MLflow tracking...")

def train_model(config: Config) -> Dict[str, Any]:
    """
    Train YOLO model with MLflow tracking.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with training results
    """
    # Setup MLflow
    setup_mlflow(config['mlflow_tracking_uri'], config['experiment_name'])
    
    # Prepare dataset
    yaml_path, class_names = load_dataset(
        config['data_csv'],
        config['images_dir'], 
        config['yolo_data_dir'],
        config['val_split'],
        config['seed']
    )
    
    # Check for YOLO weights or prepare for automatic download
    model_path = download_yolo_weights(config['model_name'], config['weights_dir'])
    
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
        
        # Load YOLO model
        print(f"Loading {config['model_name']} model...")
        try:
            model = YOLO(model_path)
            print(f"Successfully loaded model: {config['model_name']}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to directly use the model name...")
            model = YOLO(config['model_name'])
        
        # Start training
        print(f"Starting training for {config['epochs']} epochs...")
        start_time = time.time()
        
        try:
            # Train the model
            results = model.train(
                data=yaml_path,
                epochs=config['epochs'],
                imgsz=config['img_size'],
                batch=config['batch_size'],
                device=config['device'],
                workers=config['workers'],
                project=config['output_dir'],
                name='finetune',
                deterministic=True,  # Set to True for reproducibility
                pretrained=config['pretrained'],
                patience=config['patience'],
                save=True,
                verbose=True,
                
                # Augmentation parameters
                hsv_h=config['hsv_h'],
                hsv_s=config['hsv_s'],
                hsv_v=config['hsv_v'],
                translate=config['translate'],
                scale=config['scale'],
                fliplr=config['fliplr'],
                mosaic=config['mosaic'],
                erasing=config['erasing'],
                
                # Save checkpoints for MLflow
                save_period=config['save_freq'],
                exist_ok=True
            )
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Log metrics
            metrics = {
                "training_time": training_time,
                "final_precision": results.results_dict.get('metrics/precision(B)', 0),
                "final_recall": results.results_dict.get('metrics/recall(B)', 0),
                "final_map50": results.results_dict.get('metrics/mAP50(B)', 0),
                "final_map50_95": results.results_dict.get('metrics/mAP50-95(B)', 0),
                "final_fitness": results.results_dict.get('fitness', 0)
            }
            
            # Log all metrics from results
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key.replace('/', '_'), value)
            
            # Evaluate the model separately on validation set
            print("Running validation...")
            val_results = model.val(data=yaml_path)
            print(f"Validation complete.")
            
            # Add validation metrics
            for key, value in val_results.results_dict.items():
                if isinstance(value, (int, float)):
                    metrics[f"val_{key.replace('/', '_')}"] = value
                    mlflow.log_metric(f"val_{key.replace('/', '_')}", value)
            
            # Log best model
            best_model_path = os.path.join(config['output_dir'], 'finetune', 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                mlflow.log_artifact(best_model_path, "model")
                print(f"Logged best model to MLflow")
            
            # Log confusion matrix if available
            confusion_matrix_path = os.path.join(config['output_dir'], 'finetune', 'confusion_matrix.png')
            if os.path.exists(confusion_matrix_path):
                mlflow.log_artifact(confusion_matrix_path, "evaluation")
            
            # Log results plots
            results_path = os.path.join(config['output_dir'], 'finetune')
            for plot_file in os.listdir(results_path):
                if plot_file.endswith('.png') and 'confusion_matrix' not in plot_file:
                    plot_path = os.path.join(results_path, plot_file)
                    mlflow.log_artifact(plot_path, "plots")
            
            # Log final training metrics
            mlflow.log_metric("training_time", training_time)
            
            print(f"Training complete. Model saved to {config['output_dir']}/finetune")
            print(f"All metrics and artifacts logged to MLflow run: {run_id}")
            
            return metrics
            
        except Exception as training_error:
            print(f"Error during training: {training_error}")
            mlflow.log_param("error", str(training_error))
            return {"error": str(training_error)}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO model with MLflow tracking")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--data_csv', type=str, help='Path to CSV file with annotations')
    parser.add_argument('--images_dir', type=str, help='Directory with images')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--model_name', type=str, help='YOLO model name')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--device', type=str, help='Training device (cuda, cpu)')
    parser.add_argument('--mlflow_tracking_uri', type=str, help='MLflow tracking URI')
    parser.add_argument('--experiment_name', type=str, help='MLflow experiment name')
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
    metrics = train_model(config)
    
    # Print summary
    print("\nTraining Summary:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()