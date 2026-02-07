# tools/yolo_v5_tool.py
import base64
from mcp.server.fastmcp import Context

# 注意：这里我们只定义逻辑函数，具体的注册在 server.py 里做
# 或者如果你的 server.py 支持自动扫描，可以使用装饰器。
# 这里假设我们需要显式定义函数供 server.py 调用。

async def yolo_v5_tool_logic(
    image_base64: str, 
    conf_threshold: float = 0.25, 
    iou_threshold: float = 0.45
) -> str:
    """
    使用 YOLOv5 模型检测图像中的物体。
    Args:
        image_base64: 图像的 Base64 编码字符串
        conf_threshold: 置信度阈值 (0-1)
        iou_threshold: IOU 阈值 (0-1)
    """
    # 1. 延迟导入 Core，防止循环依赖，并捕获错误
    try:
        from core.yolo_v5 import detect_with_yolo_v5
    except ImportError:
        return "系统错误: 核心检测模块(core)尚未就绪。"

    # 2. 处理 Base64 图片
    try:
        # 如果包含 data:image/jpeg;base64, 前缀，需要去掉
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        image_bytes = base64.b64decode(image_base64)
    except Exception as e:
        return f"参数错误: Base64 解码失败 - {str(e)}"

    # 3. 调用核心逻辑
    try:
        result = await detect_with_yolo_v5(image_bytes, conf_threshold, iou_threshold)
        
        # 4. 返回 JSON 字符串
        return result.model_dump_json(indent=2)
    except Exception as e:
        return f"推理错误: {str(e)}"