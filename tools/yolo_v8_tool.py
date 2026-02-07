# tools/yolo_v8_tool.py
import base64

async def yolo_v8_tool_logic(
    image_base64: str, 
    conf_threshold: float = 0.25, 
    iou_threshold: float = 0.45
) -> str:
    """
    使用 YOLOv8 模型检测图像中的物体。
    """
    # 1. 导入 v8 的 Core
    try:
        from core.yolo_v8 import detect_with_yolo_v8
    except ImportError:
        return "系统错误: 核心检测模块(core/v8)尚未就绪。"

    # 2. Base64 解码
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        image_bytes = base64.b64decode(image_base64)
    except Exception as e:
        return f"参数错误: Base64 解码失败 - {str(e)}"

    # 3. 调用 v8 核心逻辑
    try:
        result = await detect_with_yolo_v8(image_bytes, conf_threshold, iou_threshold)
        return result.model_dump_json(indent=2)
    except Exception as e:
        return f"推理错误: {str(e)}"