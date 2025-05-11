import os
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any, Optional
import shutil

def download_yolo_weights(model_name: str, weights_dir: str) -> str:
    """
    Check for existing weights or prepare for download.
    
    Args:
        model_name: YOLO model name
        weights_dir: Directory to store weights
        
    Returns:
        Path to weights or model name for automatic download
    """
    print(f"Checking for {model_name} weights...")
    
    # Define common paths where weights might be stored
    possible_paths = [
        f"{model_name}.pt",
        os.path.join(weights_dir, f"{model_name}.pt"),
        os.path.join(os.path.expanduser("~"), ".ultralytics", "assets", f"{model_name}.pt")
    ]
    
    # Check if weights already exist
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found existing weights at {path}")
            return path
    
    # If not found, create weights directory
    os.makedirs(weights_dir, exist_ok=True)
    output_path = os.path.join(weights_dir, f"{model_name}.pt")
    
    print(f"Weights not found locally. The ultralytics package will attempt to download {model_name} automatically during model initialization.")
    print(f"Expected download path: {output_path}")
    
    return model_name  # Return the model name, let ultralytics handle the download

def prepare_yolo_dataset(df: pd.DataFrame, images_dir: str, output_dir: str, 
                         val_split: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Prepare the dataset in YOLO format.
    
    Args:
        df: Pandas DataFrame with annotations
        images_dir: Directory with images
        output_dir: Output directory for YOLO dataset
        val_split: Validation split ratio
        seed: Random seed for split
        
    Returns:
        Tuple of (train_ids, val_ids)
    """
    print(f"Preparing dataset in {output_dir}...")
    
    # Create directory structure
    for split in ['train', 'val']:
        (Path(output_dir) / split / 'images').mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get unique image IDs and split into train/val
    unique_image_ids = df['image_id'].unique()
    train_ids, val_ids = train_test_split(
        unique_image_ids, 
        test_size=val_split, 
        random_state=seed
    )
    
    print(f"Train images: {len(train_ids)}, Validation images: {len(val_ids)}")
    
    # Process each image
    processed_count = 0
    
    # Replace NaN values with [0, 0, 1, 1] for "No finding" class
    df_processed = df.copy()
    no_finding_mask = df_processed['class_name'] == 'No finding'
    df_processed.loc[no_finding_mask, 'x_min'] = 0
    df_processed.loc[no_finding_mask, 'y_min'] = 0
    df_processed.loc[no_finding_mask, 'x_max'] = 1
    df_processed.loc[no_finding_mask, 'y_max'] = 1
    
    for split, image_ids in [('train', train_ids), ('val', val_ids)]:
        for img_id in image_ids:
            image_path = os.path.join(images_dir, f"{img_id}")
            
            # Check if image file exists (try both .jpg and .png extensions)
            if os.path.exists(f"{image_path}.jpg"):
                image_path = f"{image_path}.jpg"
                extension = ".jpg"
            elif os.path.exists(f"{image_path}.png"):
                image_path = f"{image_path}.png"
                extension = ".png"
            else:
                print(f"Warning: Image file {img_id} not found. Skipping.")
                continue
            
            # Get all bounding boxes for this image
            img_annotations = df_processed[df_processed['image_id'] == img_id]
            
            # Process all images, including those with only "No finding"
            dest_img_path = os.path.join(output_dir, split, 'images', f"{img_id}{extension}")
            label_path = os.path.join(output_dir, split, 'labels', f"{img_id}.txt")
            
            # Copy image - using shutil instead of os.system for better cross-platform compatibility
            shutil.copy(image_path, dest_img_path)
            
            # Create label file in YOLO format
            with open(label_path, 'w') as f:
                for _, row in img_annotations.iterrows():
                    # Get class ID
                    class_id = int(row['class_id'])
                    
                    # YOLO format requires normalized coordinates:
                    # <class_id> <x_center> <y_center> <width> <height>
                    x_min, y_min = float(row['x_min']), float(row['y_min'])
                    x_max, y_max = float(row['x_max']), float(row['y_max'])
                    
                    # Calculate center and dimensions
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Ensure values are in the correct range
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            
            processed_count += 1
            if processed_count % 500 == 0:
                print(f"Processed {processed_count} images")
    
    print(f"Dataset preparation complete. Processed {processed_count} images.")
    return train_ids, val_ids

def create_data_yaml(output_dir: str, class_names: List[str]) -> str:
    """
    Create the data.yaml file required by YOLO.
    
    Args:
        output_dir: Dataset directory
        class_names: List of class names
        
    Returns:
        Path to data.yaml file
    """
    data = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f"Created data.yaml at {yaml_path}")
    return yaml_path

def load_dataset(data_csv: str, images_dir: str, yolo_data_dir: str, 
                val_split: float = 0.2, seed: int = 42) -> Tuple[str, List[str]]:
    """
    Load and prepare dataset for training.
    
    Args:
        data_csv: Path to CSV file with annotations
        images_dir: Directory with images
        yolo_data_dir: Output directory for YOLO dataset
        val_split: Validation split ratio
        seed: Random seed for split
        
    Returns:
        Tuple of (yaml_path, class_names)
    """
    # Load the CSV file
    df = pd.read_csv(data_csv)
    print(f"Loaded {len(df)} annotations from {data_csv}")
    
    # Get all class names and create a mapping
    class_names = sorted(df['class_name'].unique())
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Prepare the dataset in YOLO format
    prepare_yolo_dataset(df, images_dir, yolo_data_dir, val_split, seed)
    
    # Create data.yaml
    yaml_path = create_data_yaml(yolo_data_dir, class_names)
    
    return yaml_path, class_names