import os
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Configuration
CONFIG = {
    'data_csv': 'vinbigdata/scaled_bounding_boxes.csv',
    'images_dir': 'vinbigdata/train',
    'output_dir': 'runs/finetune',
    'yolo_data_dir': 'yolo_dataset',
    'weights_dir': 'weights',
    'model_name': 'yolo12n',
    'epochs': 50,
    'batch_size': 0.8,
    'img_size': 640,
    'device': 'mps',
    'workers': 0,
    'val_split': 0.2,
    'seed': 42,

    'hsv_h': 0.0,
    'hsv_s': 0.0,
    'hsv_v': 0.0,
    'translate': 0.0,
    'scale': 0.0,
    'fliplr': 0.0,
    'mosaic': 0.0,
    'auto_augment': None,
    'erasing': 0.0,
}

def download_yolo_weights(model_name=CONFIG['model_name']):

    print(f"Checking for {model_name} weights...")
    
    # Define common paths where weights might be stored
    possible_paths = [
        f"{model_name}.pt",
        os.path.join(CONFIG['weights_dir'], f"{model_name}.pt"),
        os.path.join(os.path.expanduser("~"), ".ultralytics", "assets", f"{model_name}.pt")
    ]
    
    # Check if weights already exist
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found existing weights at {path}")
            return path
    
    # If not found, create weights directory
    weights_dir = CONFIG['weights_dir']
    os.makedirs(weights_dir, exist_ok=True)
    output_path = os.path.join(weights_dir, f"{model_name}.pt")
    
    print(f"Weights not found locally. The ultralytics package will attempt to download {model_name} automatically during model initialization.")
    print(f"Expected download path: {output_path}")
    
    return model_name  # Return the model name, let ultralytics handle the download

def prepare_yolo_dataset(df, images_dir, output_dir):
    """
    Prepare the dataset in YOLO format
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
        test_size=CONFIG['val_split'], 
        random_state=CONFIG['seed']
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
            
            # Copy image
            os.system(f"cp '{image_path}' '{dest_img_path}'")
            
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

def create_data_yaml(output_dir, class_names):
    """
    Create the data.yaml file required by YOLO
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

def main():
    print("Starting YOLOv11x fine-tuning process...")
    
    # Check if the weights directory exists, create if not
    os.makedirs(CONFIG['weights_dir'], exist_ok=True)
    
    # Load the CSV file
    df = pd.read_csv(CONFIG['data_csv'])
    print(f"Loaded {len(df)} annotations from {CONFIG['data_csv']}")
    
    # Get all class names and create a mapping
    class_names = sorted(df['class_name'].unique())
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Prepare the dataset in YOLO format
    train_ids, val_ids = prepare_yolo_dataset(df, CONFIG['images_dir'], CONFIG['yolo_data_dir'])
    
    # Create data.yaml
    yaml_path = create_data_yaml(CONFIG['yolo_data_dir'], class_names)
    
    # Check for YOLO weights or prepare for automatic download
    model_path = download_yolo_weights(CONFIG['model_name'])
    
    # Load YOLO12x model - ultralytics will automatically download if needed
    print(f"Loading {CONFIG['model_name']} model...")
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded model: {CONFIG['model_name']}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to directly use the model name...")
        model = YOLO(CONFIG['model_name'])
    
    # Start training
    print(f"Starting training for {CONFIG['epochs']} epochs...")
    try:
        model.train(
            data=yaml_path,
            epochs=CONFIG['epochs'],
            imgsz=CONFIG['img_size'],
            batch=CONFIG['batch_size'],
            device=CONFIG['device'],
            workers=CONFIG['workers'],
            project=CONFIG['output_dir'],
            name='finetune',
            deterministic=False,
            pretrained=True,
            close_mosaic=0,
            patience=10,  # Early stopping patience
            save=True,    # Save best checkpoint
            verbose=True  # Detailed output
        )
        
        print("Training complete! Evaluating model on validation set...")
        
        # Evaluate the model
        metrics = model.val()
        print(f"Validation metrics: {metrics}")
        
        # Run inference on a few test images for verification
        print("Running inference on validation images...")
        val_images_dir = os.path.join(CONFIG['yolo_data_dir'], 'val', 'images')
        test_images = list(Path(val_images_dir).glob('*'))[:5]  # First 5 val images
        
        if test_images:
            results = model(test_images)
            print(f"Inference complete on {len(test_images)} images")
        
        print(f"Fine-tuning process complete. Model saved to {CONFIG['output_dir']}/finetune")
        
    except Exception as training_error:
        print(f"Error during training: {training_error}")
        print("Please check the model compatibility and data format.")

if __name__ == "__main__":
    main()