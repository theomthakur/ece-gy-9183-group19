import os
import sys
import argparse
import mlflow
import json
import yaml
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.training.config import Config
from src.yolo import create_data_yaml
from ultralytics import YOLO

def evaluate_model(model_path: str, yaml_path: str, output_dir: str, 
                  batch_size: int = 16, device: str = 'cuda'):
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Running evaluation on data from {yaml_path}...")
    results = model.val(
        data=yaml_path,
        batch=batch_size,
        device=device,
        project=output_dir,
        name='evaluation',
        exist_ok=True,
        verbose=True
    )
    
    print("Evaluation complete.")
    
    metrics = {}
    for key, value in results.results_dict.items():
        if isinstance(value, (int, float)):
            metrics[key.replace('/', '_')] = value
    
    return metrics

def evaluate_on_test_set(model_path: str, test_csv: str, test_images_dir: str, 
                        output_dir: str, batch_size: int = 16, device: str = 'cuda'):

    df = pd.read_csv(test_csv)
    print(f"Loaded {len(df)} annotations from {test_csv}")
    
    class_names = sorted(df['class_name'].unique())
    print(f"Found {len(class_names)} classes: {class_names}")
    
    test_dir = os.path.join(output_dir, 'test_dataset')
    os.makedirs(test_dir, exist_ok=True)
    
    images_dir = os.path.join(test_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    unique_image_ids = df['image_id'].unique()
    print(f"Copying {len(unique_image_ids)} test images...")
    
    for img_id in unique_image_ids:
        image_path = os.path.join(test_images_dir, f"{img_id}")
        
        if os.path.exists(f"{image_path}.jpg"):
            image_path = f"{image_path}.jpg"
            extension = ".jpg"
        elif os.path.exists(f"{image_path}.png"):
            image_path = f"{image_path}.png"
            extension = ".png"
        else:
            print(f"Warning: Image file {img_id} not found. Skipping.")
            continue
        
        dest_img_path = os.path.join(images_dir, f"{img_id}{extension}")
        shutil.copy(image_path, dest_img_path)
    
    print(f"Copied {len(os.listdir(images_dir))} test images")
    
    yaml_path = create_data_yaml(test_dir, class_names)
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    data['val'] = 'images'
    data['test'] = 'images'
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f"Updated {yaml_path} for test evaluation")
    
    metrics = evaluate_model(model_path, yaml_path, output_dir, batch_size, device)
    
    return metrics

def compare_models(model_paths: List[str], yaml_path: str, output_dir: str, 
                  batch_size: int = 16, device: str = 'cuda'):
    
    results = {}
    
    for i, model_path in enumerate(model_paths):
        model_name = Path(model_path).stem
        print(f"Evaluating model {i+1}/{len(model_paths)}: {model_name}")
        
        metrics = evaluate_model(
            model_path, 
            yaml_path, 
            os.path.join(output_dir, model_name),
            batch_size,
            device
        )
        
        results[model_name] = metrics
    
    comparison = pd.DataFrame(results).transpose()
    
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison.to_csv(comparison_path)
    print(f"Saved model comparison to {comparison_path}")
    
    if 'metrics_mAP50-95(B)' in comparison.columns:
        best_model = comparison['metrics_mAP50-95(B)'].idxmax()
        print(f"Best model by mAP50-95: {best_model}")
    else:
        print("Could not determine best model (mAP50-95 not found)")
    
    return results

def log_evaluation_to_mlflow(metrics: Dict[str, Any], model_path: str, 
                           tracking_uri: str, experiment_name: str, run_name: str):

    mlflow.set_tracking_uri(tracking_uri)
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
    else:
        print(f"Using existing experiment '{experiment_name}' with ID: {experiment.experiment_id}")
        mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        mlflow.log_param("model_path", model_path)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, "model")
            print(f"Logged model to MLflow: {model_path}")
        
        print(f"Logged evaluation results to MLflow run: {run_id}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--yaml_path', type=str, help='Path to data.yaml file')
    parser.add_argument('--test_csv', type=str, help='Path to CSV file with test annotations')
    parser.add_argument('--test_images_dir', type=str, help='Directory with test images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run evaluation on (cuda, cpu)')
    parser.add_argument('--compare_models', nargs='+', help='List of model paths to compare')
    parser.add_argument('--mlflow_tracking_uri', type=str, help='MLflow tracking URI')
    parser.add_argument('--experiment_name', type=str, default='yolo-evaluation', help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default='evaluation', help='MLflow run name')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.compare_models:
        if not args.yaml_path:
            print("Error: --yaml_path is required for model comparison")
            return
        
        print(f"Comparing {len(args.compare_models)} models...")
        results = compare_models(
            args.compare_models,
            args.yaml_path,
            args.output_dir,
            args.batch_size,
            args.device
        )
        
        print("\nComparison Summary:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value}")
        
        return
    
    if args.test_csv and args.test_images_dir:
        print("Evaluating on test set...")
        metrics = evaluate_on_test_set(
            args.model_path,
            args.test_csv,
            args.test_images_dir,
            args.output_dir,
            args.batch_size,
            args.device
        )

    elif args.yaml_path:
        print("Evaluating on validation set...")
        metrics = evaluate_model(
            args.model_path,
            args.yaml_path,
            args.output_dir,
            args.batch_size,
            args.device
        )
    else:
        print("Error: Either --yaml_path or both --test_csv and --test_images_dir are required")
        return
    
    print("\nEvaluation Summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    if args.mlflow_tracking_uri:
        print("Logging to MLflow...")
        log_evaluation_to_mlflow(
            metrics,
            args.model_path,
            args.mlflow_tracking_uri,
            args.experiment_name,
            args.run_name
        )

if __name__ == "__main__":
    main()