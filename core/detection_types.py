# core/detection_types.py
from pydantic import BaseModel
from typing import List

class DetectionBox(BaseModel):
    label: str
    conf: float
    bbox: List[int]

class DetectionResult(BaseModel):
    model_name: str
    boxes: List[DetectionBox]