import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import argparse

# For PyTorch model
try:
    import torch
except ImportError:
    print("PyTorch not installed. ONNX option will still work.")

# For ONNX model
try:
    import onnxruntime as ort
except ImportError:
    print("ONNX Runtime not installed. PyTorch option will still work.")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions using YOLO model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to YOLO model (.pt or .onnx)')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--sample_csv', type=str, default='sample_submission.csv', help='Path to sample submission CSV')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for input to model')
    parser.add_argument('--output_csv', type=str, default='submission.csv', help='Output CSV file path')
    return parser.parse_args()

def load_model(model_path):
    """Load either PyTorch or ONNX model based on file extension"""
    if model_path.endswith('.pt'):
        # Load PyTorch model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model.conf = args.conf_threshold  # Set confidence threshold
        return model, 'pt'
    elif model_path.endswith('.onnx'):
        # Load ONNX model
        session = ort.InferenceSession(model_path)
        return session, 'onnx'
    else:
        raise ValueError(f"Unsupported model format: {model_path}. Use .pt or .onnx")

def preprocess_image(img_path, img_size):
    """Preprocess image for model input"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image while maintaining aspect ratio
    h, w = img.shape[:2]
    scale = min(img_size / h, img_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Create img_size x img_size image with padding
    img_padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img_padded[:new_h, :new_w, :] = img_resized
    
    # Normalize image
    img_normalized = img_padded.astype(np.float32) / 255.0
    
    return img_normalized, (h, w)

def process_pt_predictions(results, orig_shape):
    """Process predictions from PyTorch model"""
    h, w = orig_shape
    
    # Find the top prediction (highest confidence)
    top_confidence = -1
    top_pred_str = ""
    
    # Extract predictions from PyTorch model output
    for det in results.xyxy[0]:  # First image's detections
        if len(det) >= 6:  # x1, y1, x2, y2, conf, cls
            x1, y1, x2, y2, conf, cls = det[:6]
            
            if float(conf) > top_confidence:
                top_confidence = float(conf)
                # Create prediction string: class_id confidence x1 y1 x2 y2
                top_pred_str = f"{int(cls)} {conf:.3f} {float(x1):.1f} {float(y1):.1f} {float(x2):.1f} {float(y2):.1f}"
    
    return top_pred_str

def process_onnx_predictions(session, img, orig_shape):
    """Process predictions from ONNX model"""
    h, w = orig_shape
    
    # Prepare input for ONNX model
    img_input = img.transpose(2, 0, 1)  # HWC to CHW
    img_input = np.expand_dims(img_input, axis=0)
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # Run inference
    outputs = session.run(output_names, {input_name: img_input})
    
    # Process output based on actual ONNX model output shape
    detections = outputs[0]  # First output contains detections
    
    # Debug printing
    print(f"Output shape: {detections.shape}")
    if len(detections) > 0:
        print(f"First detection: {detections[0]}")
    
    predictions = []
    
    # Based on the output shape (1, 300, 6), we need to handle it differently
    # Format appears to be [batch, detection_idx, [x1, y1, x2, y2, confidence, class_id]]
    for i in range(detections.shape[1]):  # Loop through all detections
        # Get the detection data
        detection = detections[0, i]  # First batch, i-th detection
        
        # Skip empty detections (all zeros)
        if np.sum(detection) == 0:
            continue
        
        # Extract coordinates, confidence, and class
        x1, y1, x2, y2, conf, cls_id = detection
        
        # Skip low confidence detections
        if float(conf) < args.conf_threshold:
            continue
        
        # Convert to COCO format [x, y, width, height]
        x = float(x1)
        y = float(y1)
        width = float(x2 - x1)
        height = float(y2 - y1)
        
        # Create prediction string: class_id confidence x1 y1 x2 y2
        pred_str = f"{int(cls_id)} {conf:.3f} {x:.1f} {y:.1f} {x+width:.1f} {y+height:.1f}"
        predictions.append(pred_str)
    
    return ' '.join(predictions)

def main():
    global args
    args = parse_args()
    
    # Load the sample submission file
    sample_df = pd.read_csv(args.sample_csv)
    print(f"Loaded sample submission with {len(sample_df)} entries")
    
    # Load model
    model, model_type = load_model(args.model_path)
    print(f"Loaded {model_type} model from {args.model_path}")
    
    # Create results dataframe
    results_df = pd.DataFrame(columns=['image_id', 'PredictionString'])
    
    # Process each image
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        image_id = row['image_id']
        img_path = os.path.join(args.img_dir, image_id + ".png")
        
        # Ensure file exists
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            results_df = pd.concat([results_df, pd.DataFrame({
                'image_id': [image_id],
                'PredictionString': ['']
            })], ignore_index=True)
            continue
        
        # Preprocess image
        img, orig_shape = preprocess_image(img_path, args.img_size)
        
        # Get predictions based on model type
        try:
            if model_type == 'pt':
                results = model(img)
                pred_string = process_pt_predictions(results, orig_shape)
            elif model_type == 'onnx':
                pred_string = process_onnx_predictions(model, img, orig_shape)
        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")
            pred_string = ""
        
        # Add to results dataframe
        results_df = pd.concat([results_df, pd.DataFrame({
            'image_id': [image_id],
            'PredictionString': [pred_string]
        })], ignore_index=True)
    
    # Save results to CSV
    results_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")

if __name__ == "__main__":
    main()