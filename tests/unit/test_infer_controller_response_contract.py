from yolo_inference_api.adapters.inbound.infer_controller import serialize_detection
from yolo_inference_api.domain.inference_detection import InferenceDetection


def test_serialize_detection_returns_label_and_bbox_edges_only():
    detection = InferenceDetection(label="bus", x1=10.0, y1=20.0, x2=30.0, y2=40.0)

    payload = serialize_detection(detection)

    assert payload == {
        "label": "bus",
        "bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
    }
    assert set(payload.keys()) == {"label", "bbox"}
    assert set(payload["bbox"].keys()) == {"x1", "y1", "x2", "y2"}
