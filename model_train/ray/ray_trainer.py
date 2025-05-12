import os
import yaml
import time
import mlflow
from datetime import datetime
from pathlib import Path

import ray
from ray import train
from ray.train import ScalingConfig, CheckpointConfig, RunConfig, FailureConfig
from ray.train.torch import TorchTrainer
import torch

from ultralytics import YOLO

def train_func(config):
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    
    yaml_path = config["data_yaml"]
    model_name = config["model_name"]
    pretrained_weights = config.get("pretrained_weights", None)
    
    is_retraining = pretrained_weights is not None
    model_info = f"{model_name} (retraining from {pretrained_weights})" if is_retraining else model_name
    
    print(f"Worker {rank}/{world_size} starting {'re' if is_retraining else ''}training with YOLO model {model_info}")
    
    try:
        if is_retraining and os.path.exists(pretrained_weights):
            print(f"Worker {rank}: Loading pretrained weights from {pretrained_weights}")
            model = YOLO(pretrained_weights)
        elif os.path.exists(model_name):
            model = YOLO(model_name)
        else:
            model = YOLO(model_name)
        print(f"Worker {rank}: Successfully loaded model {model_info}")
    except Exception as e:
        print(f"Worker {rank}: Error loading model: {e}")
        raise e
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = config.get("run_name", f"{'retrain-' if is_retraining else ''}{model_name}-{timestamp}-worker{rank}")
    
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
    
    print(f"Worker {rank}: Starting {'re' if is_retraining else ''}training for {epochs} epochs")
    start_time = time.time()
    
    try:
        checkpoint = train.get_checkpoint()
        resume_weights = None
        
        if checkpoint:
            print(f"Worker {rank}: Restoring from Ray checkpoint")
            with checkpoint.as_directory() as checkpoint_dir:
                resume_weights = os.path.join(checkpoint_dir, "best.pt")
                if os.path.exists(resume_weights):
                    print(f"Worker {rank}: Found checkpoint at {resume_weights}")
                else:
                    print(f"Worker {rank}: No checkpoint found at {resume_weights}")
                    resume_weights = None
        
        if resume_weights:
            print(f"Worker {rank}: Resuming training from Ray checkpoint {resume_weights}")
            model = YOLO(resume_weights)
        
        if is_retraining and not resume_weights:
            print(f"Worker {rank}: Retraining using weights from {pretrained_weights}")
        
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
        
        metrics_to_report = {
            "map": metrics_dict.get("map", 0.0),
            "map50": metrics_dict.get("map50", 0.0),
            "map75": metrics_dict.get("map75", 0.0),
            "training_time_minutes": training_duration,
            "is_retraining": 1 if is_retraining else 0
        }
        
        if is_retraining:
            metrics_to_report["pretrained_weights"] = pretrained_weights
        
        train.report(metrics_to_report)
        
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

def run_distributed_training(config, data_yaml, scaling_config, 
                           checkpoint_dir="s3://your-bucket/ray-checkpoints",
                           max_failures=2,
                           run_config=None):
    
    checkpoint_config = CheckpointConfig(
        num_to_keep=5,
        checkpoint_score_attribute="map50",
        checkpoint_score_order="max",
        checkpoint_frequency=1,
        _checkpoint_keep_all_ranks=False,
        _checkpoint_upload_from_tasks=True
    )
    
    if run_config is None:
        run_config = RunConfig(
            name=f"YOLO-{config['model_name']}-training",
            storage_path=checkpoint_dir, 
            failure_config=FailureConfig(max_failures=max_failures)
        )
    else:
        run_config = RunConfig(
            name=run_config.name,
            storage_path=checkpoint_dir,
            failure_config=FailureConfig(max_failures=max_failures)
        )
    
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

if __name__ == "__main__":
    scaling_config = ScalingConfig(
        num_workers=2,
        use_gpu=True,
        resources_per_worker={
            "CPU": 4,
            "GPU": 1,
        }
    )
    
    config = {
        "model_name": "yolo12n.pt",
        "epochs": 50,
        "batch_size": 16,
        "img_size": 640,
        "lr": 0.01,
        "workers": 4,
        "patience": 10,
    }
    
    data_yaml = "config/data.yaml"
    s3_checkpoint_path = f"s3://{os.environ.get('BUCKET_NAME', 'mlflow-artifacts')}/ray-checkpoints"
    
    result, best_checkpoint = run_distributed_training(
        config=config,
        data_yaml=data_yaml,
        scaling_config=scaling_config,
        checkpoint_dir=s3_checkpoint_path,
        max_failures=2
    )
    
    print(f"Training complete with best checkpoint at: {best_checkpoint}")
