"""
Post-processing utilities for detection outputs.

- filter_by_conf: drop boxes below threshold.
"""

from typing import List

from .detection_types import DetectionBox


def filter_by_conf(boxes: List[DetectionBox], conf: float) -> List[DetectionBox]:
    """Filter detections by confidence.

    Args:
        boxes (List[DetectionBox]): Raw detection boxes.
        conf (float): Confidence threshold (0-1).

    Returns:
        List[DetectionBox]: Filtered detections.
    """
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1")

    return [box for box in boxes if box.conf >= conf]
