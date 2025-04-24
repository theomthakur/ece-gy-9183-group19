from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import time
import cv2
import io
import onnxruntime as ort
import mlflow
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from app.models import DetectionResult, DetectionResponse
from app.utils import preprocess_image, postprocess_predictions

# Initialize FastAPI app
app = FastAPI(
    title="Chest X-ray Abnormality Detection API",
    description="API for detecting abnormalities in chest X-rays using YOLOv11-L",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path
MODEL_PATH = os.getenv("MODEL_PATH", "/models/yolov11_cxr.onnx")

# Initialize metrics
PREDICTION_COUNT = Counter("prediction_count", "Number of predictions made")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Time spent processing prediction")
ABNORMALITY_DETECTED = Counter("abnormality_detected", "Number of abnormalities detected")

# Class labels for the VinDr-CXR dataset
CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis"
]

# Initialize ONNX Runtime session
try:
    # Check if GPU is available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Using providers: {session.get_providers()}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to CPU if GPU is not available
    providers = ['CPUExecutionProvider']
    try:
        session = ort.InferenceSession(MODEL_PATH, providers=providers)
        print(f"Fallback to CPU. Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Fatal error loading model: {e}")
        raise

# Input and output names
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

@app.get("/")
async def root():
    return {"message": "Welcome to Chest X-ray Abnormality Detection API"}

@app.post("/predict", response_model=DetectionResponse)
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert grayscale to 3-channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Preprocess image
        input_data = preprocess_image(image)
        
        # Run inference
        outputs = session.run(output_names, {input_name: input_data})
        
        # Postprocess predictions
        boxes, scores, class_ids = postprocess_predictions(outputs, image.shape[:2])
        
        # Create response
        detections = []
        for i in range(len(boxes)):
            if scores[i] > 0.25:  # Confidence threshold
                x1, y1, x2, y2 = boxes[i]
                class_id = int(class_ids[i])
                class_name = CLASS_NAMES[class_id]
                
                detection = DetectionResult(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(scores[i]),
                    bbox=[float(x1), float(y1), float(x2), float(y2)]
                )
                detections.append(detection)
                ABNORMALITY_DETECTED.inc()  # Increment abnormality counter
        
        # Record metrics
        PREDICTION_COUNT.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return DetectionResponse(
            filename=file.filename,
            predictions=detections,
            processing_time=time.time() - start_time
        )
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/model-info")
async def model_info():
    input_details = [
        {
            "name": input.name,
            "shape": input.shape,
            "type": input.type
        }
        for input in session.get_inputs()
    ]
    
    output_details = [
        {
            "name": output.name,
            "shape": output.shape,
            "type": output.type
        }
        for output in session.get_outputs()
    ]
    
    return {
        "model_path": MODEL_PATH,
        "providers": session.get_providers(),
        "inputs": input_details,
        "outputs": output_details,
        "classes": CLASS_NAMES
    }