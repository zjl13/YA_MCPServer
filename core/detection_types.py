"""
Detection data models.

- DetectionBox: single detection item.
- DetectionResult: model name and list of detections.
"""

from typing import List

from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    """A single detection box."""

    label: str = Field(..., description="Class label")
    conf: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: List[int] = Field(..., description="Bounding box as [x1, y1, x2, y2]")


class DetectionResult(BaseModel):
    """Detection output for one model run."""

    model_name: str = Field(..., description="Model identifier")
    image_width: int = Field(..., ge=1, description="Input image width")
    image_height: int = Field(..., ge=1, description="Input image height")
    boxes: List[DetectionBox] = Field(default_factory=list, description="Detections")