import os
import yaml
import time
import mlflow
from datetime import datetime
from pathlib import Path

import ray
from ray import train
from ray.train import ScalingConfig, CheckpointConfig, RunConfig
from ray.train.torch import TorchTrainer
import torch

from ultralytics import YOLO

def train_func(config):

    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    
    yaml_path = config["data_yaml"]
    model_name = config["model_name"]
    
    print(f"Worker {rank}/{world_size} starting training with YOLO model {model_name}")
    
    try:
        if os.path.exists(model_name):
            model = YOLO(model_name)
        else:
            model = YOLO(model_name)
        print(f"Worker {rank}: Successfully loaded model {model_name}")
    except Exception as e:
        print(f"Worker {rank}: Error loading model: {e}")
        raise e
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = config.get("run_name", f"{model_name}-{timestamp}-worker{rank}")
    
    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 16)
    img_size = config.get("img_size", 640)
    lr = config.get("lr", 0.01)
    
    hsv_h = config.get("hsv_h", 0.0)
    hsv_s = config.get("hsv_s", 0.0)
    hsv_v = config.get("hsv_v", 0.0)
    translate = config.get("translate", 0.0)
    scale = config.get("scale", 0.0)
    fliplr = config.get("fliplr", 0.0)
    mosaic = config.get("mosaic", 0.0)
    
    if rank == 0 and not config.get("tune_run", False):
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "YOLO12-Ray-Train")
        
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
            
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_id)
        except Exception as e:
            print(f"Worker {rank}: MLflow error: {e}")
    
    output_dir = config.get("output_dir", "./runs")
    
    print(f"Worker {rank}: Starting training for {epochs} epochs")
    start_time = time.time()
    
    try:
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=f"cuda:" if torch.cuda.is_available() else "cpu",
            workers=config.get("workers", 8),
            project=output_dir,
            name=run_name,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            translate=translate,
            scale=scale,
            fliplr=fliplr,
            mosaic=mosaic,
            close_mosaic=epochs,
            patience=config.get("patience", 10),
            save=True,
            lr0=lr,
            verbose=True,
        )
        
        training_duration = (time.time() - start_time) / 60
        print(f"Worker {rank}: Training completed in {training_duration:.2f} minutes")
        
        print(f"Worker {rank}: Evaluating model")
        metrics = model.val()
        
        if hasattr(metrics, "to_dict"):
            metrics_dict = metrics.to_dict()
        else:
            metrics_dict = metrics
            
        run_dir = os.path.join(output_dir, run_name)
        
        train.report({
            "map": metrics_dict.get("map", 0.0),
            "map50": metrics_dict.get("map50", 0.0),
            "map75": metrics_dict.get("map75", 0.0),
            "training_time_minutes": training_duration
        })
        
        best_model_path = os.path.join(run_dir, "weights", "best.pt")
        if os.path.exists(best_model_path):
            with train.checkpoint.CheckpointContext() as checkpoint_ctx:
                checkpoint_ctx.add_file(best_model_path, path_in_checkpoint="best.pt")
                
        if rank == 0:
            try:
                print(f"Worker {rank}: Exporting model to ONNX")
                model.export(format="onnx", imgsz=img_size)
            except Exception as e:
                print(f"Worker {rank}: Error exporting to ONNX: {e}")
        
        return metrics_dict
        
    except Exception as e:
        print(f"Worker {rank}: Error during training: {e}")
        train.report({"error": str(e)})
        raise e

def create_yolo_trainer(config, data_yaml, scaling_config, checkpoint_config=None, run_config=None):

    train_config = config.copy()
    train_config["data_yaml"] = data_yaml
    
    trainer = TorchTrainer(
        train_func,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        checkpoint_config=checkpoint_config,
        run_config=run_config,
    )
    
    return trainer

def run_distributed_training(config, data_yaml, scaling_config, checkpoint_config, run_config=None):
    trainer = create_yolo_trainer(
        config, 
        data_yaml, 
        scaling_config, 
        checkpoint_config, 
        run_config
    )
    
    result = trainer.fit()
    
    print("Training complete!")
    print(f"Best result: {result.metrics}")
    
    best_checkpoint = result.best_checkpoints[0][0]
    
    return result, best_checkpoint