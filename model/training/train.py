import os
import argparse
import yaml
import logging
import time
import shutil
import mlflow
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[Any, Any]:

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_mlflow(experiment_name: str, run_name: Optional[str] = None, 
                tracking_uri: Optional[str] = None) -> str:

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment(experiment_name)
    
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"YOLOv11_train_{timestamp}"
        
    run = mlflow.start_run(run_name=run_name)
    return run.info.run_id


def train_yolo(config: Dict[Any, Any], output_dir: str, 
               resume_from: Optional[str] = None) -> str:

    # Initialize device settings
    device = config.get('device', '')  # Empty string means all available GPUs
    
    # Initialize model
    if resume_from:
        logger.info(f"Resuming training from checkpoint: {resume_from}")
        model = YOLO(resume_from)
    else:
        logger.info("Initializing new YOLOv11-L model")
        if config.get('pretrained', True):
            model = YOLO('yolov11l.pt')  # Base YOLOv11-L model
        else:
            model = YOLO('yolov11l.yaml')  # Model architecture without weights
    
    # Create training configuration
    train_cfg = {

        'task': 'detect',
        'data': config['data_yaml_path'],
        'epochs': config['epochs'],
        'patience': config['patience'],
        'batch': config['batch_size'],
        'imgsz': config['img_size'],
        'optimizer': config['optimizer'],
        'lr0': config['learning_rate'],
        'lrf': config['final_lr_factor'],
        'momentum': config['momentum'],
        'weight_decay': config['weight_decay'],
        'warmup_epochs': config['warmup_epochs'],
        'warmup_momentum': config['warmup_momentum'],
        'box': config['box_loss_weight'],
        'cls': config['cls_loss_weight'],
        
        # Data augmentation
        'cache': config.get('cache', True),

        # Additional training settings
        'cos_lr': config.get('cos_lr', True),  # Cosine LR scheduler
        'overlap_mask': config.get('overlap_mask', True),  # Overlapping masks
        'mask_ratio': config.get('mask_ratio', 4),  # Mask downsampling ratio
        'single_cls': config.get('single_cls', False),  # Train as single-class dataset
        'rect': config.get('rect', False),  # Rectangular training (batch shape maintained)
        'multi_scale': config.get('multi_scale', False),  # Multi-scale training
        'fraction': config.get('fraction', 1.0),  # Dataset fraction to train on
        'amp': config.get('amp', True),  # Automatic mixed precision
        'nbs': config.get('nbs', 64),  # Nominal batch size
        'val': config.get('val', True),  # Validate during training
    }
    
    # Create directory for config
    os.makedirs("./configs", exist_ok=True)
    cfg_file = f"./configs/train_config_{int(time.time())}.yaml"
    with open(cfg_file, 'w') as f:
        yaml.dump(train_cfg, f)
    
    # Log configuration
    logger.info(f"Training configuration saved to {cfg_file}")
    mlflow.log_artifact(cfg_file)
    mlflow.log_params(train_cfg)
    
    # Start training
    logger.info("Starting training...")
    results = model.train(
        cfg=cfg_file,
        project="chest-xray-detection",
        name=os.path.basename(output_dir),
        exist_ok=True,
        pretrained=config.get('pretrained', True),
        device=device,
        verbose=True,
        deterministic=config.get('deterministic', False),
        seed=config.get('seed', 0),
    )
    
    # Log metrics
    metrics = {
        "mAP50": results.results_dict.get('metrics/mAP50(B)', 0),
        "mAP50-95": results.results_dict.get('metrics/mAP50-95(B)', 0),
        "precision": results.results_dict.get('metrics/precision(B)', 0),
        "recall": results.results_dict.get('metrics/recall(B)', 0),
        "f1": 2 * results.results_dict.get('metrics/precision(B)', 0) * results.results_dict.get('metrics/recall(B)', 0) / 
              (results.results_dict.get('metrics/precision(B)', 1e-6) + results.results_dict.get('metrics/recall(B)', 1e-6) + 1e-6),
        "train_box_loss": results.results_dict.get('train/box_loss', 0),
        "train_cls_loss": results.results_dict.get('train/cls_loss', 0),
        "train_dfl_loss": results.results_dict.get('train/dfl_loss', 0),
        "val_box_loss": results.results_dict.get('val/box_loss', 0),
        "val_cls_loss": results.results_dict.get('val/cls_loss', 0),
        "val_dfl_loss": results.results_dict.get('val/dfl_loss', 0),
        "fitness": results.results_dict.get('fitness', 0),
    }
    
    # Log metrics to MLflow
    mlflow.log_metrics(metrics)
    
    # Log confusion matrix and other plots as artifacts
    if hasattr(results, 'save_dir') and os.path.exists(results.save_dir):
        for plot_file in os.listdir(results.save_dir):
            if plot_file.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                mlflow.log_artifact(os.path.join(results.save_dir, plot_file))
    
    # Get path to best weights
    best_weights_path = os.path.join(results.save_dir, "weights/best.pt")
    last_weights_path = os.path.join(results.save_dir, "weights/last.pt")
    
    # Copy weights to output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(best_weights_path):
        output_best_path = os.path.join(output_dir, "best.pt")
        shutil.copy(best_weights_path, output_best_path)
        logger.info(f"Best weights saved to {output_best_path}")
        mlflow.log_artifact(output_best_path, "models")
    else:
        logger.warning(f"Best weights not found at {best_weights_path}")
    
    if os.path.exists(last_weights_path):
        output_last_path = os.path.join(output_dir, "last.pt")
        shutil.copy(last_weights_path, output_last_path)
        logger.info(f"Last weights saved to {output_last_path}")
        mlflow.log_artifact(output_last_path, "models")
    
    # Export model for deployment
    logger.info("Exporting model for deployment...")
    
    # Export to ONNX format
    if config.get('export_onnx', True):
        try:
            onnx_path = os.path.join(output_dir, "model.onnx")
            model.export(format="onnx", imgsz=config['img_size'])
            if os.path.exists(model.onnx_path):
                shutil.copy(model.onnx_path, onnx_path)
                logger.info(f"ONNX model exported to {onnx_path}")
                mlflow.log_artifact(onnx_path, "deployment_models")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
    
    # Log model to MLflow model registry
    if config.get('log_model_to_registry', True):
        try:
            # Package the model in a format MLflow can understand
            mlflow.pytorch.log_model(
                model.model.float(),  # Get the PyTorch model from YOLO wrapper
                "mlflow_model",
                registered_model_name=config.get('model_registry_name', "YOLOv11L_ChestXray")
            )
            logger.info("Model logged to MLflow model registry")
        except Exception as e:
            logger.error(f"Failed to log model to registry: {e}")
    
    return os.path.join(output_dir, "best.pt")


def main():
    parser = argparse.ArgumentParser(description="YOLOv11-L Training Script")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to training configuration YAML"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./models/production",
        help="Directory to save model outputs"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--mlflow-uri", 
        type=str, 
        default=None,
        help="MLflow tracking server URI"
    )
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default="YOLOv11-CXR-Production",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name", 
        type=str, 
        default=None,
        help="MLflow run name (default: auto-generated)"
    )
    parser.add_argument(
        "--gpu", 
        type=str, 
        default="",
        help="Specific GPU indices to use (e.g. '0,1,2')"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set GPU if specified
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        config['device'] = [int(idx) for idx in args.gpu.split(',')]
        logger.info(f"Using GPUs: {args.gpu}")
    
    # Setup MLflow
    run_id = setup_mlflow(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tracking_uri=args.mlflow_uri
    )
    logger.info(f"MLflow run ID: {run_id}")
    
    # Save run ID to output directory for reference
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "mlflow_run_id.txt"), "w") as f:
        f.write(run_id)
    
    # Print system information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Start training
    try:
        best_model_path = train_yolo(
            config=config,
            output_dir=args.output_dir,
            resume_from=args.resume
        )
        logger.info(f"Training completed successfully! Best model saved at: {best_model_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        mlflow.end_run(status="FAILED")
        exit(1)
    
    # End MLflow run
    mlflow.end_run()


if __name__ == "__main__":
    main()