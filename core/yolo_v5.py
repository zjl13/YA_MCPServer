"""
YOLOv5 inference helpers.

- detect_with_yolo_v5: run detection with YOLOv5.
"""

from io import BytesIO
from typing import Any, List

from .detection_types import DetectionBox, DetectionResult
from .postprocess import filter_by_conf

_YOLO_V5_MODEL: Any = None


def _validate_thresholds(conf: float, iou: float) -> None:
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1")
    if not 0.0 <= iou <= 1.0:
        raise ValueError("iou must be between 0 and 1")


def _load_image(image_bytes: bytes) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install pillow") from exc

    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to decode image bytes: {exc}") from exc


def _get_config_value(key: str, default: Any) -> Any:
    try:
        from modules.YA_Common.utils.config import get_config
    except ImportError:
        return default
    return get_config(key, default)


def _get_model() -> Any:
    global _YOLO_V5_MODEL
    if _YOLO_V5_MODEL is not None:
        return _YOLO_V5_MODEL

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required: pip install torch") from exc

    weights_path = _get_config_value("yolo_v5.weights_path", None)
    model_variant = _get_config_value("yolo_v5.model", "yolov5s")
    device = _get_config_value("yolo_v5.device", "cpu")

    try:
        if weights_path:
            model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, force_reload=False)
        else:
            model = torch.hub.load("ultralytics/yolov5", model_variant, pretrained=True)
        model.to(device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLOv5 model: {exc}") from exc

    _YOLO_V5_MODEL = model
    return model


def _parse_results(model: Any, results: Any) -> List[DetectionBox]:
    try:
        names = getattr(model, "names", {})
        preds = results.xyxy[0].tolist()
    except Exception as exc:
        raise RuntimeError(f"Failed to parse YOLOv5 outputs: {exc}") from exc

    boxes: List[DetectionBox] = []
    for x1, y1, x2, y2, score, cls_id in preds:
        cls_index = int(cls_id)
        label = names.get(cls_index, str(cls_index)) if isinstance(names, dict) else str(cls_index)
        boxes.append(
            DetectionBox(
                label=label,
                conf=float(score),
                bbox=[int(x1), int(y1), int(x2), int(y2)],
            )
        )
    return boxes


async def detect_with_yolo_v5(image: bytes, conf: float, iou: float) -> DetectionResult:
    """Run YOLOv5 detection.

    Args:
        image (bytes): Input image bytes.
        conf (float): Confidence threshold (0-1).
        iou (float): IoU threshold (0-1).

    Returns:
        DetectionResult: Parsed detection results.

    Raises:
        ValueError: Invalid thresholds or empty image bytes.
        RuntimeError: Dependency missing or inference failed.
    """
    if not image:
        raise ValueError("image bytes are empty")

    _validate_thresholds(conf, iou)

    model = _get_model()
    model.conf = conf
    model.iou = iou

    img = _load_image(image)

    try:
        results = model(img)
    except Exception as exc:
        raise RuntimeError(f"YOLOv5 inference failed: {exc}") from exc

    boxes = _parse_results(model, results)
    filtered = filter_by_conf(boxes, conf)
    width, height = img.size
    return DetectionResult(
        model_name="yolo_v5",
        image_width=width,
        image_height=height,
        boxes=filtered,
    )