from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

from app.models import ImageRequest, PredictionResponse
from app.utils import load_model, process_image

# Initialize FastAPI app
app = FastAPI(
    title="Chest X-ray Abnormality Detection API",
    description="Detecting Chest X-ray Abnormalities using YOLOv11",
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

# Environment variables for configuration
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
MODEL_PATH = os.getenv("MODEL_PATH", "yolov11n.pt")

# Load the model at startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = load_model(os.path.join(MODEL_DIR, MODEL_PATH))
    except Exception as e:
        print(f"Error loading model: {e}")
        # Allow server to start even if model fails to load
        # The API endpoints will handle the None model case

@app.get("/")
def root():
    return {"message": "YOLOv11 Object Detection API"}

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Add device information to health check
    import torch
    device_info = "unknown"
    try:
        if torch.cuda.is_available():
            device_info = f"CUDA: {torch.cuda.get_device_name(0)}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info = "MPS (Apple Silicon)"
        else:
            device_info = "CPU"
    except:
        pass
        
    return {
        "status": "healthy", 
        "model": MODEL_PATH,
        "device": device_info
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_image(request: ImageRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process the image and run inference with user-provided thresholds
        detections = process_image(
            model, 
            request.image,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold
        )
        return detections
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))