from unittest.mock import Mock

from yolo_inference_api.application.image_inference_use_case import ImageInferenceUseCase
from yolo_inference_api.domain.inference_detection import InferenceDetection


class FakeInferenceUseCase:
    def infer(self, image_bytes: bytes) -> list[InferenceDetection]:
        return [InferenceDetection(label="cat", x1=1.0, y1=2.0, x2=3.0, y2=4.0)]


def test_protocol_accepts_concrete_fake_implementation():
    fake_use_case = FakeInferenceUseCase()

    assert isinstance(fake_use_case, ImageInferenceUseCase)
    assert fake_use_case.infer(b"image-bytes")[0].label == "cat"


def test_protocol_accepts_mock_object_with_infer_method():
    mock_use_case = Mock()
    mock_use_case.infer = Mock(return_value=[])

    assert isinstance(mock_use_case, ImageInferenceUseCase)
    assert mock_use_case.infer(b"image-bytes") == []
