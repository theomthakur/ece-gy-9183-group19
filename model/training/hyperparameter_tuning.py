import os
import argparse
import yaml
import logging
import mlflow
from pathlib import Path
from typing import Dict, Any

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.train import Checkpoint
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search import ConcurrencyLimiter
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[Any, Any]:
    """Load configuration from YAML file"""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def train_yolo(config: Dict[Any, Any], checkpoint_dir=None):

    mlflow.set_experiment("YOLOv11-L-Hyperparameter-Tuning")

    with mlflow.start_run(run_name=f"trial_{ray.train.get_context().get_trial_id()}"):

        mlflow.log_params(config)
        model = YOLO('yolov11l.pt')
        
        # Temporary YAML config for this trial
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
            'cache': config.get('cache', True),
        }
        
        # Temporary config file for this trial
        os.makedirs("./tmp", exist_ok=True)
        cfg_file = f"./tmp/trial_{ray.train.get_context().get_trial_id()}.yaml"
        with open(cfg_file, 'w') as f:
            yaml.dump(train_cfg, f)
        
        # Resume from checkpoint if available
        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            model = YOLO(checkpoint_path)
        
        results = model.train(
            cfg=cfg_file,
            project="chest-xray-detection",
            name=f"trial_{ray.train.get_context().get_trial_id()}",
            exist_ok=True,
            pretrained=True,
            device=config.get('device', 'mps')
        )
        
        val_map50 = results.results_dict.get('metrics/mAP50(B)', 0)
        val_map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
        
        mlflow.log_metrics({
            "mAP50": val_map50,
            "mAP50-95": val_map50_95,
            "precision": results.results_dict.get('metrics/precision(B)', 0),
            "recall": results.results_dict.get('metrics/recall(B)', 0),
            "val_loss": results.results_dict.get('val/box_loss', 0) + 
                       results.results_dict.get('val/cls_loss', 0) + 
                       results.results_dict.get('val/dfl_loss', 0)
        })
        
        # Save checkpoint for Ray Tune
        checkpoint_path = os.path.join(results.save_dir, "weights/best.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = Checkpoint.from_directory(Path(checkpoint_path).parent)
            session.report(
                {"mAP50": val_map50, "mAP50-95": val_map50_95},
                checkpoint=checkpoint
            )
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}")
            session.report({"mAP50": val_map50, "mAP50-95": val_map50_95})


def main():
    parser = argparse.ArgumentParser(description="YOLO12X Hyperparameter Tuning")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/hyperparameter_tuning.yaml",
        help="Path to hyperparameter tuning configuration"
    )
    parser.add_argument(
        "--ray-address", 
        type=str, 
        default=None,
        help="Ray cluster address (default: None for local)"
    )
    parser.add_argument(
        "--cpus-per-trial", 
        type=int, 
        default=8,
        help="CPUs per trial"
    )
    parser.add_argument(
        "--gpus-per-trial", 
        type=float, 
        default=1.0,
        help="GPUs per trial"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=20,
        help="Number of hyperparameter combinations to try"
    )
    parser.add_argument(
        "--max-concurrent-trials", 
        type=int, 
        default=4,
        help="Maximum number of concurrent trials"
    )
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.ray_address:
        ray.init(address=args.ray_address)
        logger.info(f"Connected to Ray cluster at {args.ray_address}")
    else:
        ray.init()
        logger.info("Started local Ray instance")
    
    search_space = {

        # Fixed parameters
        'data_yaml_path': config['data_yaml_path'],
        'epochs': config['epochs'],
        'patience': config['patience'],
        'img_size': config['img_size'],
        'cache': config.get('cache', True),
        
        # Search space for hyperparameters
        'batch_size': tune.choice(config['batch_size_options']),
        'optimizer': tune.choice(config['optimizer_options']),
        'learning_rate': tune.loguniform(config['learning_rate_range'][0], config['learning_rate_range'][1]),
        'final_lr_factor': tune.loguniform(config['final_lr_factor_range'][0], config['final_lr_factor_range'][1]),
        'momentum': tune.uniform(config['momentum_range'][0], config['momentum_range'][1]),
        'weight_decay': tune.loguniform(config['weight_decay_range'][0], config['weight_decay_range'][1]),
        'warmup_epochs': tune.choice(config['warmup_epochs_options']),
        'warmup_momentum': tune.uniform(config['warmup_momentum_range'][0], config['warmup_momentum_range'][1]),
        'box_loss_weight': tune.uniform(config['box_loss_weight_range'][0], config['box_loss_weight_range'][1]),
        'cls_loss_weight': tune.uniform(config['cls_loss_weight_range'][0], config['cls_loss_weight_range'][1]),
    }
    
    search_alg = BayesOptSearch(
        metric="mAP50-95",
        mode="max",
        random_search_steps=5,
    )
    
    search_alg = ConcurrencyLimiter(
        search_alg,
        max_concurrent=args.max_concurrent_trials
    )
    
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='mAP50-95',
        mode='max',
        max_t=config['epochs'],
        grace_period=config.get('grace_period', 5),
        reduction_factor=2
    )
    
    reporter = CLIReporter(
        metric_columns=["mAP50", "mAP50-95", "training_iteration"],
        parameter_columns=["learning_rate", "batch_size", "optimizer"]
    )
    
    tuner = tune.Tuner(
        tune.with_resources(
            train_yolo,
            resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial}
        ),

        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent_trials,
            reuse_actors=False
        ),

        run_config=ray.train.RunConfig(
            name="yolo12_vindr_cxr_tuning",
            progress_reporter=reporter,

            checkpoint_config=ray.train.CheckpointConfig(
                checkpoint_score_attribute="mAP50-95",
                checkpoint_score_order="max",
                num_to_keep=3
            ),

            storage_path=config.get("experiment_dir", "./ray_results"),
            failure_config=ray.train.FailureConfig(max_failures=3)
        ),

        param_space=search_space,
    )
    
    results = tuner.fit()
    
    best_trial = results.get_best_result(metric="mAP50-95", mode="max")
    logger.info(f"Best trial config: {best_trial.config}")
    logger.info(f"Best trial final mAP50: {best_trial.metrics['mAP50']}")
    logger.info(f"Best trial final mAP50-95: {best_trial.metrics['mAP50-95']}")
    
    # Save the best model
    best_checkpoint = best_trial.checkpoint
    if best_checkpoint:
        best_model_path = os.path.join(
            config.get("output_dir", "./models"), 
            "best_model_from_tuning.pt"
        )
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        

        best_checkpoint_path = best_checkpoint.path
        if os.path.isdir(best_checkpoint_path):
            
            model_files = [f for f in os.listdir(best_checkpoint_path) if f.endswith('.pt')]
            if model_files:
                import shutil
                shutil.copy(
                    os.path.join(best_checkpoint_path, model_files[0]), 
                    best_model_path
                )
                logger.info(f"Best model saved to {best_model_path}")
            else:
                logger.warning(f"No model files found in {best_checkpoint_path}")
        else:
            logger.warning(f"Expected directory but got: {best_checkpoint_path}")
    
    best_hparams = {k: v for k, v in best_trial.config.items()}
    best_hparams_path = os.path.join(
        config.get("output_dir", "./models"), 
        "best_hyperparameters.yaml"
    )
    with open(best_hparams_path, 'w') as f:
        yaml.dump(best_hparams, f)
    logger.info(f"Best hyperparameters saved to {best_hparams_path}")
    
    ray.shutdown()
    logger.info("Hyperparameter tuning completed!")


if __name__ == "__main__":
    main()