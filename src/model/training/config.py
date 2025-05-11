import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import json

class Config:
    
    DEFAULT_CONFIG = {
        # Data configuration
        'data_csv': 'train.csv',
        'images_dir': 'data/train',
        'output_dir': 'runs/finetune',
        'yolo_data_dir': 'yolo_dataset',
        'weights_dir': 'weights',
        
        # Model configuration
        'model_name': 'yolo12n', 
        'pretrained': True,       
        
        # Training parameters
        'epochs': 50,
        'batch_size': 0.8,         
        'img_size': 640,
        'device': 'cuda',
        'workers': 8,
        'val_split': 0.2,
        'seed': 42,
        'patience': 10,           
        
        # Augmentation parameters
        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.0,
        'translate': 0.0,
        'scale': 0.0,
        'fliplr': 0.0,
        'mosaic': 0.0,
        'auto_augment': None,
        'erasing': 0.0,
        
        # Ray configuration
        'ray_address': 'auto',
        'num_workers': 2,
        'use_gpu': True,          
        'cpus_per_worker': 4,     
        'gpus_per_worker': 1,     
        
        # MLflow configuration
        'mlflow_tracking_uri': 'http://localhost:5000',
        'experiment_name': 'yolo-training',
        
        # Checkpoint configuration
        'checkpoint_dir': 'checkpoints',
        'save_freq': 5,           
        
        # Hyperparameter tuning
        'tune_trials': 10,
        'tune_resources': {       
            'cpu': 4,
            'gpu': 1
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path:
            self.load_from_file(config_path)
            
        self._ensure_directories()
    
    def _ensure_directories(self):
        dirs = ['weights_dir', 'output_dir', 'checkpoint_dir']
        for dir_key in dirs:
            os.makedirs(self.config[dir_key], exist_ok=True)
    
    def load_from_file(self, config_path: str):

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            if file_ext == '.yaml' or file_ext == '.yml':
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            elif file_ext == '.json':
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_ext}")
            
            self.config.update(loaded_config)
            print(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            raise
    
    def save_to_file(self, config_path: str):

        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            if file_ext == '.yaml' or file_ext == '.yml':
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, sort_keys=False)
            elif file_ext == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
            else:
                raise ValueError(f"Unsupported config file format: {file_ext}")
            
            print(f"Saved configuration to {config_path}")
            
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")
            raise
    
    def update(self, new_config: Dict[str, Any]):
        self.config.update(new_config)        
        self._ensure_directories()
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        self.config[key] = value        
        if key in ['weights_dir', 'output_dir', 'checkpoint_dir']:
            os.makedirs(value, exist_ok=True)
    
    def as_dict(self) -> Dict[str, Any]:
        return self.config.copy()

    def get_mlflow_params(self) -> Dict[str, Any]:
        return {k: v for k, v in self.config.items() 
                if isinstance(v, (str, int, float, bool)) or v is None}