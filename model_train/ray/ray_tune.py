import os
import yaml
import mlflow
from datetime import datetime
from pathlib import Path

import ray
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
from ray.tune.integration.mlflow import MLflowLoggerCallback

from ray_trainer import train_func

def create_tune_search_space():

    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([8, 16, 32, 64]),
        "img_size": tune.choice([512, 640, 768]),
    }
    
    return search_space

def create_tune_scheduler(max_epochs, grace_period=5):
    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=grace_period,
        reduction_factor=2,
        metric="map50",
        mode="max",
    )
    
    return scheduler

def create_bayesopt_search(max_concurrent=4):

    algo = BayesOptSearch(
        metric="map50",
        mode="max",
        max_concurrent=max_concurrent,
        utility_kwargs={
            "kind": "ucb",
            "kappa": 2.5,
            "seed": 42,
        }
    )
    
    return algo

def run_hyperparameter_tuning(
    config, 
    data_yaml, 
    num_samples=10, 
    max_concurrent=None,
    resources_per_trial=None
):

    if max_concurrent is None:
        num_gpus = int(ray.cluster_resources().get("GPU", 0))
        max_concurrent = max(1, num_gpus)
    
    if resources_per_trial is None:
        resources_per_trial = {
            "cpu": 4,
            "gpu": 1,
        }
    
    base_config = config.copy()
    base_config["data_yaml"] = data_yaml
    base_config["tune_run"] = True
    
    search_space = create_tune_search_space()
    merged_config = {**base_config, **search_space}
    
    scheduler = create_tune_scheduler(max_epochs=config["epochs"])
    search_algo = create_bayesopt_search(max_concurrent=max_concurrent)
    
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "YOLO-RayTune"),
        save_artifact=True,
    )
    
    print(f"Starting hyperparameter tuning with {num_samples} trials")
    print(f"Using up to {max_concurrent} concurrent trials")
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_name = f"{config['model_name']}-tune-{timestamp}"
    
    tuner = tune.Tuner(
        tune.with_resources(
            train_func,
            resources=resources_per_trial
        ),
        param_space=merged_config,
        tune_config=tune.TuneConfig(
            metric="map50",
            mode="max",
            scheduler=scheduler,
            search_alg=search_algo,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
            reuse_actors=False,
        ),
        run_config=ray.train.RunConfig(
            name=experiment_name,
            callbacks=[mlflow_callback],
            storage_path=config.get("output_dir", "./runs"),
            checkpoint_config=ray.train.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="map50",
                checkpoint_score_order="max",
            ),
        ),
    )
    
    results = tuner.fit()
    
    print("Hyperparameter tuning complete!")
    print(f"Best trial: {results.get_best_result(metric='map50', mode='max')}")
    
    best_result = results.get_best_result(metric="map50", mode="max")
    best_config = best_result.config
    best_metrics = best_result.metrics
    
    print("Best hyperparameters:")
    for key, value in best_config.items():
        if key in search_space:
            print(f"  {key}: {value}")
    
    print(f"Best mAP@50: {best_metrics.get('map50', 0.0)}")
    
    try:
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
            
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "YOLO-RayTune")
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"best-{experiment_name}", log_system_metrics=True):
            for key, value in best_config.items():
                if key in search_space:
                    mlflow.log_param(key, value)
            
            for key, value in best_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
            mlflow.log_dict(best_config, "best_config.yaml")
    except Exception as e:
        print(f"Error logging to MLflow: {e}")
    
    return results, best_config