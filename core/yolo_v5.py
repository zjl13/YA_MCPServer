# core/yolo_v5.py
from .detection_types import DetectionResult, DetectionBox

# 模拟一个异步函数
async def detect_with_yolo_v5(image: bytes, conf: float, iou: float) -> DetectionResult:
    # 这里我们不真跑模型，直接返回假数据
    print(f"模拟推理中... conf={conf}, iou={iou}")
    return DetectionResult(
        model_name="yolo_v5_mock",
        boxes=[
            DetectionBox(label="person", conf=0.95, bbox=[100, 100, 200, 300]),
            DetectionBox(label="car", conf=0.88, bbox=[400, 300, 500, 400])
        ]
    )