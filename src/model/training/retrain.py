import os
import sys
import argparse
import mlflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from train_ray import train_with_ray, setup_mlflow
from train import train_model


def retrain_model(config: Config, production_data_csv: str, production_images_dir: str):

    # Update config with production data paths
    config.update({
        'data_csv': production_data_csv,
        'images_dir': production_images_dir,
        'yolo_data_dir': os.path.join(config['output_dir'], 'production_dataset'),
        'experiment_name': f"{config['experiment_name']}_retrain"
    })
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # Log parameters
        mlflow.log_params(config.get_mlflow_params())
        mlflow.log_param("retrain", True)
        mlflow.log_param("production_data_csv", production_data_csv)
        mlflow.log_param("production_images_dir", production_images_dir)
        
        # Use Ray for distributed training
        if config['num_workers'] > 1:
            print(f"Retraining with Ray using {config['num_workers']} workers...")
            metrics = train_with_ray(config)
        else:
            # Use standard training
            print("Retraining with standard training...")
            metrics = train_model(config)
        
        print(f"Retraining complete. Metrics logged to MLflow run: {run_id}")
        
        return metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Retrain YOLO model on production data")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--production_data_csv', type=str, required=True, help='Path to CSV file with production annotations')
    parser.add_argument('--production_images_dir', type=str, required=True, help='Directory with production images')
    parser.add_argument('--model_path', type=str, help='Path to pre-trained model to fine-tune')
    parser.add_argument('--output_dir', type=str, help='Directory to save retraining results')
    parser.add_argument('--epochs', type=int, help='Number of epochs for retraining')
    parser.add_argument('--batch_size', type=int, help='Batch size for retraining')
    parser.add_argument('--ray_address', type=str, help='Ray cluster address (local, auto, or specific address)')
    parser.add_argument('--num_workers', type=int, help='Number of Ray workers')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--mlflow_tracking_uri', type=str, help='MLflow tracking URI')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    config = Config(args.config)
    
    arg_dict = {k: v for k, v in vars(args).items() 
                if v is not None and k not in ['config', 'production_data_csv', 'production_images_dir']}
    
    if arg_dict:
        config.update(arg_dict)
    
    if args.model_path:
        config['model_name'] = args.model_path
        config['pretrained'] = True
    
    metrics = retrain_model(config, args.production_data_csv, args.production_images_dir)
    
    print("\nRetraining Summary:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
