from fastapi import FastAPI
from pydantic import BaseModel, Field
import base64
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI(
    title="Chest X-Ray Detection API (ONNX)",
    description="API for detecting anomalies in chest X-rays using YOLO v11 with ONNX Runtime",
    version="1.0.0"
)

# Define the request and response models
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0, le=1)
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class PredictionResponse(BaseModel):
    detections: list[Detection]
    processing_time: float

# Load ONNX model
MODEL_PATH = "model.onnx"

# Set device (Use GPU if available, otherwise CPU)
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] 
session = ort.InferenceSession(MODEL_PATH, providers=providers)

# Define class labels for chest X-ray detection
# This is a placeholder list - you should replace these with your actual classes
# For YOLOv11 with 79 classes (based on tensor shape [1, 84, 8400])
# The first few classes are shown here - add all of your classes
classes = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis", "Lung Cancer", "Pleural Effusion", 
           "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule", "Atelectasis", 
           "Cardiomegaly", "Consolidation", "Edema", "Effusion"]

# Note: Update this list with all classes your model can detect

# Define the image preprocessing function for YOLO
def preprocess_image(img, input_height=640, input_width=640):
    # Convert PIL Image to OpenCV format
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Get original dimensions
    original_height, original_width = img.shape[:2]
    
    # Resize the image
    resized = cv2.resize(img, (input_width, input_height))
    
    # Normalize pixel values
    resized = resized / 255.0
    
    # Transpose to match YOLO expected format: (N, C, H, W)
    input_tensor = resized.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
    
    return input_tensor, original_width, original_height

def process_output(output, original_width, original_height, input_width=640, input_height=640, conf_threshold=0.25, iou_threshold=0.45):
    # Process YOLO v11 model output with shape [1, 84, 8400]
    # - 84 = 4 (box coordinates) + 1 (objectness) + 79 (class scores)
    # - 8400 = grid cells or anchors
    detections = []
    
    # Extract output tensor
    output = output[0]  # First element of the output list
    
    # Transpose from [1, 84, 8400] to [1, 8400, 84] for easier processing
    # This puts each detection in a separate row
    output = np.transpose(output[0], (1, 0))  # Shape becomes [8400, 84]
    
    # Prepare arrays for NMS
    boxes = []
    scores = []
    class_ids = []
    
    # Number of classes (84 - 5 = 79 classes)
    num_classes = output.shape[1] - 5
    
    # Process each detection
    for i in range(output.shape[0]):
        # Get box coordinates and confidence
        x, y, w, h = output[i, 0:4]  # First 4 values are box coordinates (center_x, center_y, width, height)
        confidence = output[i, 4]  # 5th value is objectness score
        
        # Filter by confidence threshold
        if confidence >= conf_threshold:
            # Get class scores
            class_scores = output[i, 5:5+num_classes]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            
            # Filter by class score
            if class_score >= conf_threshold:
                # Convert center_x, center_y, width, height to corner coordinates
                # Denormalize to get pixel coordinates
                x_min = (x - w/2) * original_width / input_width
                y_min = (y - h/2) * original_height / input_height
                x_max = (x + w/2) * original_width / input_width
                y_max = (y + h/2) * original_height / input_height
                
                boxes.append([x_min, y_min, x_max, y_max])
                scores.append(confidence * class_score)  # Combine objectness with class confidence
                class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    
    for i in indices:
        if isinstance(i, tuple) or isinstance(i, list):  # For older OpenCV versions
            i = i[0]
            
        box = boxes[i]
        class_id = class_ids[i]
        confidence = float(scores[i])
        class_name = classes[class_id] if class_id < len(classes) else f"Unknown-{class_id}"
        
        detections.append(Detection(
            class_id=int(class_id),
            class_name=class_name,
            confidence=confidence,
            x_min=float(box[0]),
            y_min=float(box[1]),
            x_max=float(box[2]),
            y_max=float(box[3])
        ))
    
    return detections

@app.post("/predict")
def predict_image(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Preprocess the image
        input_tensor, original_width, original_height = preprocess_image(image)

        # Use the exact input/output names from your model
        input_name = "images"  # Input name from your YOLO model
        output_name = "output0"  # Output name from your YOLO model
        
        # Run inference with ONNX
        import time
        start_time = time.time()
        outputs = session.run([output_name], {input_name: input_tensor})
        processing_time = time.time() - start_time
        
        # Process detections
        detections = process_output(outputs, original_width, original_height)

        return PredictionResponse(detections=detections, processing_time=processing_time)

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "Chest X-Ray Detection API is running. Use /predict endpoint with a base64 encoded image."}