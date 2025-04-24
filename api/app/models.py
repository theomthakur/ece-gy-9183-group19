from pydantic import BaseModel
from typing import List, Optional, Union

class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class DetectionResponse(BaseModel):
    filename: str
    predictions: List[DetectionResult]
    processing_time: float