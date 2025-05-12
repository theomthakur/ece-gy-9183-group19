import os
import yaml
import mlflow
import time
from pathlib import Path
from datetime import datetime
from mlflow.tracking import MlflowClient

from ultralytics import YOLO

def load_config_with_defaults(config_path):
   with open(config_path, "r") as file:
       config = yaml.safe_load(file)
   
   if 'model_registry' not in config:
       config['model_registry'] = {
           'enabled': True,
           'auto_transition_to_staging': False,
           'tags': {}
       }
   
   return config

def setup_mlflow():
   tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
   mlflow.set_tracking_uri(tracking_uri)
   
   experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "YOLO12-Training")
   
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

def log_model_to_mlflow(model, config, metrics, experiment_id, config_path, yaml_path, run_name, run_dir):
   try:
       if experiment_id:
           mlflow.set_experiment(experiment_id=experiment_id)
       
       with mlflow.start_run(run_name=run_name, log_system_metrics=True):
           print(f"Started MLflow run: {run_name}")
           
           mlflow.log_param("model_name", config['model_name'])
           mlflow.log_param("epochs", config['epochs'])
           mlflow.log_param("batch_size", config['batch_size'])
           mlflow.log_param("img_size", config['img_size'])
           mlflow.log_param("device", config['device'])
           print(f"Logged parameters to MLflow")
           
           if metrics is not None:
               if isinstance(metrics, dict):
                   metrics_dict = metrics
               elif hasattr(metrics, 'to_dict'):
                   metrics_dict = metrics.to_dict()
               else:
                   metrics_dict = {}
                   for attr in ['map', 'map50', 'map75', 'maps', 'fitness']:
                       if hasattr(metrics, attr):
                           value = getattr(metrics, attr)
                           if isinstance(value, (int, float)):
                               metrics_dict[attr] = value
               
               for key, value in metrics_dict.items():
                   if isinstance(value, (int, float)):
                       mlflow.log_metric(key, value)
               print(f"Logged metrics to MLflow")
           
           run_id = mlflow.active_run().info.run_id
           
           best_pt_path = os.path.join(run_dir, 'weights', 'best.pt')
           try:
               if os.path.exists(best_pt_path):
                   mlflow.log_artifact(best_pt_path, "model")
                   print(f"Logged best PyTorch model to MLflow from {best_pt_path}")
               else:
                   print(f"Warning: Could not find best.pt at {best_pt_path}")
           except Exception as e:
               print(f"Error logging PyTorch model to MLflow: {e}")
           
           best_onnx_path = os.path.join(run_dir, 'weights', 'best.onnx')
           try:
               if os.path.exists(best_onnx_path):
                   mlflow.log_artifact(best_onnx_path, "model")
                   print(f"Logged ONNX model to MLflow from {best_onnx_path}")
               else:
                   print(f"Warning: Could not find best.onnx at {best_onnx_path}")
           except Exception as e:
               print(f"Error logging ONNX model to MLflow: {e}")
           
           try:
               mlflow.log_artifact(config_path, "config")
               mlflow.log_artifact(yaml_path, "config")
               print(f"Logged config files to MLflow")
           except Exception as e:
               print(f"Error logging config files to MLflow: {e}")
           
           model_version = None
           if config.get('model_registry', {}).get('enabled', True):
               try:
                   model_to_register = None
                   if os.path.exists(best_pt_path):
                       model_to_register = best_pt_path
                   elif os.path.exists(best_onnx_path):
                       model_to_register = best_onnx_path
                   
                   if model_to_register:
                       registered_model_name = f"{config['model_name']}"
                       model_type = "PyTorch" if model_to_register.endswith(".pt") else "ONNX"
                       artifact_path = "model"
                       model_uri = f"runs:/{run_id}/{artifact_path}/{os.path.basename(model_to_register)}"
                       
                       model_version = mlflow.register_model(
                           model_uri=model_uri,
                           name=registered_model_name
                       )
                       print(f"Registered {model_type} model as '{registered_model_name}' (version: {model_version.version})")
                       
                       client = MlflowClient()
                       
                       description = f"{config['model_name']} trained on custom dataset for {config['epochs']} epochs. "
                       description += f"Map50: {metrics_dict.get('map50', 'N/A')}, Map: {metrics_dict.get('map', 'N/A')}"
                       client.update_model_version(
                           name=registered_model_name,
                           version=model_version.version,
                           description=description
                       )
                       
                       registry_tags = config.get('model_registry', {}).get('tags', {})
                       for key, value in registry_tags.items():
                           client.set_model_version_tag(
                               name=registered_model_name,
                               version=model_version.version,
                               key=key,
                               value=str(value)
                           )
                       
                       client.set_model_version_tag(
                           name=registered_model_name,
                           version=model_version.version,
                           key="training_date",
                           value=datetime.now().strftime('%Y-%m-%d')
                       )
                       
                       if config.get('model_registry', {}).get('auto_transition_to_staging', False):
                           client.transition_model_version_stage(
                               name=registered_model_name,
                               version=model_version.version,
                               stage="Staging"
                           )
                           print(f"Model version {model_version.version} transitioned to 'Staging'")
                       
                       print(f"Added metadata to registered model version")
               except Exception as e:
                   print(f"Warning: Could not register model: {e}")
           
           print(f"Model and metrics successfully logged to MLflow")
           
           if model_version:
               return run_id, model_version.version
           return run_id, None
   except Exception as e:
       print(f"Warning: Could not log to MLflow: {e}")
       print("Check your network connection and credentials.")
       return None, None

def main(config_path, yaml_path):
   print("Starting YOLO12 fine-tuning process with preprocessed data...")

   CONFIG = load_config_with_defaults(config_path)
   
   os.makedirs(CONFIG['weights_dir'], exist_ok=True)
   
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
   
   model_path = download_yolo_weights(config_path)
   
   print(f"Loading {CONFIG['model_name']} model...")
   try:
       model = YOLO(model_path)
       print(f"Successfully loaded model: {CONFIG['model_name']}")
   except Exception as e:
       print(f"Error loading model: {e}")
       print("Attempting to directly use the model name...")
       model = YOLO(CONFIG['model_name'])
   
   start_time = time.time()
   
   custom_run_name = os.environ.get("CUSTOM_RUN_NAME")
   if custom_run_name:
       run_name = custom_run_name
       print(f"Using custom run name from command line: {run_name}")
   else:
       timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
       run_name = f"{CONFIG['model_name']}-{timestamp}"
   
   print(f"Starting training for {CONFIG['epochs']} epochs with run name: {run_name}...")
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
           save=True,
           verbose=True,
       )
       
       run_dir = os.path.join(CONFIG['output_dir'], run_name)
       training_duration = (time.time() - start_time) / 60
       print(f"Training completed in {training_duration:.2f} minutes")
       print(f"Results saved to {run_dir}")
       
       print("Training complete! Evaluating model on validation set...")
       
       metrics = model.val()
       
       if hasattr(metrics, 'to_dict'):
           metrics_dict = metrics.to_dict()
       else:
           metrics_dict = metrics
           
       print(f"Validation metrics: {metrics_dict}")
       
       print("Running inference on validation images...")
       val_images = []
       val_txt_path = os.path.join(CONFIG['yolo_data_dir'], 'validation.txt')
       if os.path.exists(val_txt_path):
           with open(val_txt_path, 'r') as f:
               val_images = [line.strip() for line in f.readlines()][:5]
       else:
           val_images_dir = os.path.join(CONFIG['yolo_data_dir'], 'val', 'images')
           if os.path.exists(val_images_dir):
               val_images = list(Path(val_images_dir).glob('*'))[:5]
           else:
               print("Warning: No validation images found for inference test.")
       
       if val_images:
           results = model(val_images)
           print(f"Inference complete on {len(val_images)} images")
       
       print("Exporting model to ONNX format...")
       try:
           model.export(format="onnx", imgsz=CONFIG['img_size'])
           print("Model successfully exported to ONNX format")
       except Exception as e:
           print(f"Error exporting model to ONNX: {e}")
       
       run_id, model_version = log_model_to_mlflow(
           model, 
           CONFIG, 
           metrics_dict, 
           experiment_id, 
           config_path, 
           yaml_path, 
           run_name, 
           run_dir
       )
       
       if run_id and model_version:
           mlflow_ui = os.environ.get('MLFLOW_TRACKING_URI')
           print(f"Model registered successfully:")
           print(f"  - Model name: {CONFIG['model_name']}")
           print(f"  - Version: {model_version}")
           print(f"  - Run ID: {run_id}")
           print(f"You can access the registered model at: {mlflow_ui}/#/models/{CONFIG['model_name']}/versions/{model_version}")
       
       print(f"Fine-tuning process complete. Model saved to {run_dir}")
       print(f"You can view the experiment in MLflow at {os.environ.get('MLFLOW_TRACKING_URI')}")
       
   except Exception as training_error:
       print(f"Error during training: {training_error}")
       print("Please check the model compatibility and data format.")
