from typing import Protocol, runtime_checkable

from yolo_inference_api.domain.inference_detection import InferenceDetection


@runtime_checkable
class ImageInferencePort(Protocol):
    def infer(self, image_bytes: bytes) -> list[InferenceDetection]:
        ...