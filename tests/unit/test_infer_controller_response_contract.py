import pytest

from yolo_inference_api.adapters.inbound.infer_controller import _validate_image_bytes, serialize_detection
from yolo_inference_api.domain.inference_detection import InferenceDetection


def test_validate_image_bytes_raises_when_empty():
    with pytest.raises(ValueError, match="Missing image bytes"):
        _validate_image_bytes(b"")


def test_serialize_detection_returns_label_and_bbox_edges_only():
    detection = InferenceDetection(label="bus", x1=10.0, y1=20.0, x2=30.0, y2=40.0)

    payload = serialize_detection(detection)

    assert payload == {
        "label": "bus",
        "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
    }
    assert set(payload.keys()) == {"label", "bbox"}
    assert set(payload["bbox"].keys()) == {"x1", "y1", "x2", "y2"}
