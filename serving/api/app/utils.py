import base64
import io
import time
from typing import List, Dict, Any
from PIL import Image
import numpy as np
from app.models import PredictionResponse, Detection

def load_model(model_path: str):
    """
    Load YOLOv11 model from path
    """
    # Load the YOLO model using ultralytics
    try:
        # Import here to avoid torch import issues if package isn't installed yet
        from ultralytics import YOLO
        
        # Try loading from local path
        model = YOLO(model_path)
        
        # Check if CUDA is available and set device accordingly
        # This works on both Mac (uses MPS or CPU) and NVIDIA systems (uses CUDA)
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon GPU support
        else:
            device = 'cpu'
            
        print(f"Using device: {device}")
    except Exception as e:
        # Fallback to loading pre-trained YOLOv11
        print(f"Error loading from path: {e}")
        print("Loading pre-trained YOLOv8 model as placeholder...")
        from ultralytics import YOLO
        model = YOLO('yolov11n.pt')
    
    return model

def decode_base64_image(base64_str: str) -> Image.Image:
    """
    Decode base64 string to PIL Image
    """
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Error decoding image: {e}")

def process_image(model, base64_image: str, confidence_threshold: float = 0.25, iou_threshold: float = 0.45) -> PredictionResponse:
    """
    Process image with YOLO model and return detections
    
    Parameters:
    - model: The loaded YOLO model
    - base64_image: Base64 encoded image string
    - confidence_threshold: Confidence threshold for detections (0.0-1.0)
    - iou_threshold: IoU threshold for NMS (0.0-1.0)
    """
    # Decode base64 image
    image = decode_base64_image(base64_image)
    
    # Measure inference time
    start_time = time.time()
    
    # Run inference using ultralytics YOLO model with the provided thresholds
    results = model.predict(
        source=np.array(image),
        conf=confidence_threshold,  # Use the provided confidence threshold
        iou=iou_threshold,          # Use the provided IoU threshold
        verbose=False
    )
    
    # Calculate inference time
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Process results - ultralytics YOLO format
    detections_list = []
    
    # Get the first result (assuming single image input)
    result = results[0]
    
    img_width, img_height = image.size
