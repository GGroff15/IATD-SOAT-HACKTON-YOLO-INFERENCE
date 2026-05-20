from typing import Protocol, runtime_checkable

from yolo_inference_api.domain.image_inference_port import ImageInferencePort
from yolo_inference_api.domain.inference_detection import InferenceDetection

@runtime_checkable
class ImageInferenceUseCase(Protocol):
    def infer(self, image_bytes: bytes) -> list[InferenceDetection]:
        ...


class ImageInferenceService(ImageInferenceUseCase):
    def __init__(self, inference_port: ImageInferencePort):
        self._inference_port = inference_port

    def infer(self, image_bytes: bytes) -> list[InferenceDetection]:
        return self._inference_port.infer(image_bytes)
