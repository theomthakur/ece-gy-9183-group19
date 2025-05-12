import os
import yaml
import mlflow
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def load_config_with_defaults(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    config['model_registry'] = {
        'enabled': False,
        'auto_transition_to_staging': False,
        'tags': {}
    }
    
    config['output_dir'] = './test_runs'
    return config

def setup_mlflow():
    mlflow.config.disable_system_metrics_logging()
    
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", f"YOLO12-Testing")
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")
    except Exception as e:
        print(f"Warning: Could not set up MLflow experiment: {e}")
        print(f"Make sure MLflow server is running and accessible at {tracking_uri}")
        print("Check your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        experiment_id = None
    
    return experiment_id

def download_yolo_weights(config_path):
    with open(config_path, "r") as file:
        CONFIG = yaml.safe_load(file)
    
    model_name = CONFIG['model_name']
    print(f"Checking for {model_name} weights...")
    
    possible_paths = [
        f"{model_name}.pt",
        os.path.join(CONFIG['weights_dir'], f"{model_name}.pt"),
        os.path.join(os.path.expanduser("~"), ".ultralytics", "assets", f"{model_name}.pt")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found existing weights at {path}")
            return path
    
    weights_dir = CONFIG['weights_dir']
    os.makedirs(weights_dir, exist_ok=True)
    print(f"Weights not found locally. The ultralytics package will attempt to download {model_name} automatically during model initialization.")
    
    return model_name

def log_basic_metrics_to_mlflow(config, training_duration, metrics_dict, experiment_id, run_name):
    try:
        if experiment_id:
            mlflow.set_experiment(experiment_id=experiment_id)
            
            with mlflow.start_run(run_name=run_name, log_system_metrics=True):
                print(f"Started MLflow run: {run_name}")
                
                mlflow.log_param("model_name", config['model_name'])
                mlflow.log_param("epochs", config['epochs'])
                mlflow.log_param("batch_size", config['batch_size'])
                mlflow.log_param("img_size", config['img_size'])
                
                for key, value in metrics_dict.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                        if 'time' in key:
                            print(f"Logged {key}: {value:.2f} minutes")
                
                run_id = mlflow.active_run().info.run_id
                print(f"Test run logged to MLflow with run_id: {run_id}")
                return run_id
    except Exception as e:
        print(f"Warning: Could not log to MLflow: {e}")
        return None

def main(config_path, yaml_path):
    print("Starting YOLO12 test process with preprocessed data...")
    CONFIG = load_config_with_defaults(config_path)
    
    os.makedirs(CONFIG['weights_dir'], exist_ok=True)
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"Test runs will be saved to: {CONFIG['output_dir']}")
    
    if not os.path.exists(yaml_path):
        print(f"Error: data.yaml not found at {yaml_path}")
        print("Please ensure your YOLO dataset is properly formatted.")
        return
    
    print(f"Using existing data.yaml at {yaml_path}")
    experiment_id = setup_mlflow()
    
    classes_path = os.path.join(CONFIG['yolo_data_dir'], 'classes.txt')
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Found {len(class_names)} classes in classes.txt")
    else:
        with open(yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
            if 'names' in data_yaml:
                class_names = data_yaml['names']
                print(f"Found {len(class_names)} classes in data.yaml")
            else:
                print("Warning: No class names found. Please check your dataset format.")
                class_names = []
    
    model_load_start = time.time()
    model_path = download_yolo_weights(config_path)
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded model: {CONFIG['model_name']}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to directly use the model name...")
        model = YOLO(CONFIG['model_name'])
    
    model_load_time = (time.time() - model_load_start) / 60
    training_start_time = time.time()
    
    custom_run_name = os.environ.get("CUSTOM_RUN_NAME")
    if custom_run_name:
        run_name = custom_run_name
        print(f"Using custom run name from command line: {run_name}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_name = f"test-{CONFIG['model_name']}-{timestamp}"
    
    print(f"Starting testing for {CONFIG['epochs']} epochs with run name: {run_name}...")
    
    try:
        results = model.train(
            data=yaml_path,
            epochs=CONFIG['epochs'],
            imgsz=CONFIG['img_size'],
            batch=CONFIG['batch_size'],
            device=CONFIG['device'],
            workers=CONFIG['workers'],
            project=CONFIG['output_dir'],
            name=run_name,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            translate=0.0,
            scale=0.0,
            fliplr=0.0,
            mosaic=0.0,
            auto_augment=None,
            erasing=0.0,
            deterministic=False,
            pretrained=True,
            close_mosaic=CONFIG['epochs'],
            patience=10,
            save=False,
            verbose=True,
        )
        
        run_dir = os.path.join(CONFIG['output_dir'], run_name)
        training_duration = (time.time() - training_start_time) / 60
        
        print(f"Testing completed in {training_duration:.2f} minutes")
        print(f"Results saved to {run_dir}")
        
        print("Test run complete! Evaluating model on validation set...")
        
        val_start_time = time.time()
        metrics = model.val()
        val_time = (time.time() - val_start_time) / 60
        
        metrics_dict = {}
        
        if isinstance(metrics, dict):
            metrics_dict = metrics.copy()
        elif hasattr(metrics, 'to_dict'):
            metrics_dict = metrics.to_dict()
        else:
            for attr in ['map', 'map50', 'map75', 'maps', 'fitness']:
                if hasattr(metrics, attr):
                    value = getattr(metrics, attr)
                    if isinstance(value, (int, float)):
                        metrics_dict[attr] = value
        
        metrics_dict['model_load_time_minutes'] = model_load_time
        metrics_dict['training_time_minutes'] = training_duration
        metrics_dict['validation_time_minutes'] = val_time
        metrics_dict['total_time_minutes'] = model_load_time + training_duration + val_time
        
        print(f"Validation metrics: {metrics_dict}")
        
        run_id = log_basic_metrics_to_mlflow(
            CONFIG,
            training_duration,
            metrics_dict,
            experiment_id,
            run_name
        )
        
        if run_id:
            mlflow_ui = os.environ.get('MLFLOW_TRACKING_URI')
            print(f"Test run logged successfully:")
            print(f" - Run name: {run_name}")
            print(f" - Run ID: {run_id}")
            print(f" - Training duration: {training_duration:.2f} minutes")
            print(f"You can access the test run at: {mlflow_ui}/#/experiments/{experiment_id}/runs/{run_id}")
        
        print(f"Testing process complete. Results saved to {run_dir}")
        print(f"You can view the experiment in MLflow at {os.environ.get('MLFLOW_TRACKING_URI')}")
    
    except Exception as training_error:
        print(f"Error during testing: {training_error}")
        print("Please check the model compatibility and data format.")