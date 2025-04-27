from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import numpy as np
import tritonclient.http as httpclient
import base64

from app.models import ImageRequest, PredictionResponse, Detection

# Initialize FastAPI app
app = FastAPI(
    title="Chest X-ray Abnormality Detection API",
    description="Detecting Chest X-ray Abnormalities using YOLOv11 with Triton Inference Server",
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

# Configure Triton client
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "triton_server:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "chest_xray_detector")

@app.get("/")
def root():
    return {"message": "YOLOv11 Object Detection API with Triton Inference Server"}

@app.get("/health")
def health_check():
    try:
        # Connect to Triton server
        triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)
        
        # Check if server is alive
        if not triton_client.is_server_live():
            raise HTTPException(status_code=503, detail="Triton server is not live")
        
        # Check if server is ready
        if not triton_client.is_server_ready():
            raise HTTPException(status_code=503, detail="Triton server is not ready")
        
        # Check if our model is ready
        if not triton_client.is_model_ready(MODEL_NAME):
            raise HTTPException(status_code=503, detail=f"Model {MODEL_NAME} is not ready")
        
        # Get model configuration
        model_metadata = triton_client.get_model_metadata(MODEL_NAME)
        model_config = triton_client.get_model_config(MODEL_NAME)
        
        return {
            "status": "healthy",
            "model": MODEL_NAME,
            "triton_server": TRITON_SERVER_URL,
            "model_version": model_metadata.get("versions", [""])[0]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
def predict_image(request: ImageRequest):
    try:
        # Connect to Triton server
        triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)
        
        # Prepare inputs
        inputs = []
        
        # Image input (base64 encoded)
        inputs.append(httpclient.InferInput("INPUT_IMAGE", [1, 1], "BYTES"))
        input_data = np.array([[request.image]], dtype=object)
        inputs[0].set_data_from_numpy(input_data)
        
        # Confidence threshold input
        inputs.append(httpclient.InferInput("CONFIDENCE_THRESHOLD", [1, 1], "FP32"))
        conf_data = np.array([[request.confidence_threshold]], dtype=np.float32)
        inputs[1].set_data_from_numpy(conf_data)
        
        # IoU threshold input
        inputs.append(httpclient.InferInput("IOU_THRESHOLD", [1, 1], "FP32"))
        iou_data = np.array([[request.iou_threshold]], dtype=np.float32)
        inputs[2].set_data_from_numpy(iou_data)
        
        # Prepare outputs
        outputs = []
        outputs.append(httpclient.InferRequestedOutput("DETECTIONS", binary_data=False))
        outputs.append(httpclient.InferRequestedOutput("INFERENCE_TIME", binary_data=False))
        
        # Run inference
        results = triton_client.infer(MODEL_NAME, inputs, outputs=outputs)
        
        # Parse results
        detections_json = results.as_numpy("DETECTIONS")[0][0]
        inference_time = float(results.as_numpy("INFERENCE_TIME")[0][0])
        
        # Parse detections from JSON
        detections_list = json.loads(detections_json)
        
        # Convert to Detection objects
        detections = []
        for det in detections_list:
            detections.append(Detection(
                class_id=det["class_id"],
                class_name=det["class_name"],
                confidence=det["confidence"],
                bbox=det["bbox"]
            ))
        
        # Create response
        return PredictionResponse(
            detections=detections,
            inference_time=inference_time,
            model_name="yolov11n"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))