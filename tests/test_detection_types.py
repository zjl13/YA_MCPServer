"""Minimal serialization test for detection models."""

from core.detection_types import DetectionBox, DetectionResult


def _build_sample_result() -> DetectionResult:
    return DetectionResult(
        model_name="sample",
        image_width=640,
        image_height=480,
        boxes=[DetectionBox(label="person", conf=0.9, bbox=[10, 20, 100, 200])],
    )


def test_detection_result_serialization() -> None:
    result = _build_sample_result()
    payload = result.model_dump_json()
    assert "model_name" in payload
    assert "image_width" in payload
    assert "image_height" in payload
    assert "boxes" in payload


if __name__ == "__main__":
    test_detection_result_serialization()
    print("OK")
