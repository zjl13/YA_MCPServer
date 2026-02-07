# core/yolo_v8.py
from .detection_types import DetectionResult, DetectionBox

async def detect_with_yolo_v8(image: bytes, conf: float, iou: float) -> DetectionResult:
    # 模拟 v8 的推理结果
    print(f"[Core] YOLOv8 推理中... conf={conf}, iou={iou}")
    return DetectionResult(
        model_name="yolo_v8_mock",  # <--- 这里改了名字，方便区分
        boxes=[
            DetectionBox(label="cat", conf=0.99, bbox=[50, 50, 150, 150]),
            DetectionBox(label="dog", conf=0.92, bbox=[200, 200, 400, 400])
        ]
    )