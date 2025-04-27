from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class ImageRequest(BaseModel):
    """Request model for image prediction"""
    image: str = Field(..., description="Base64 encoded image")
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Minimum confidence threshold for detections")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold for non-maximum suppression")

class Detection(BaseModel):
    """Individual detection result"""
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2] normalized to [0,1]")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    detections: List[Detection]
    inference_time: float = Field(..., description="Inference time in milliseconds")
    model_name: str