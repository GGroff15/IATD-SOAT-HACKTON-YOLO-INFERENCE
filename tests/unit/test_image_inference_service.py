from unittest.mock import Mock

from yolo_inference_api.application.image_inference_use_case import ImageInferenceService
from yolo_inference_api.domain.inference_detection import InferenceDetection


def test_service_delegates_inference_call_to_outbound_port():
    image_bytes = b"image-bytes"
    inference_port = Mock()
    inference_port.infer = Mock(return_value=[])
    service = ImageInferenceService(inference_port=inference_port)

    service.infer(image_bytes)

    inference_port.infer.assert_called_once_with(image_bytes)


def test_service_returns_detections_from_outbound_port_without_changes():
    expected_detections = [
        InferenceDetection(label="person", x1=10.0, y1=20.0, x2=30.0, y2=40.0),
    ]
    inference_port = Mock()
    inference_port.infer = Mock(return_value=expected_detections)
    service = ImageInferenceService(inference_port=inference_port)

    detections = service.infer(b"image-bytes")

    assert detections == expected_detections