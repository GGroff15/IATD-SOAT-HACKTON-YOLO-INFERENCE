from collections.abc import Iterable
from io import BytesIO
from typing import Any

from PIL import Image
from ultralytics import YOLO

from yolo_inference_api.domain.image_inference_port import ImageInferencePort
from yolo_inference_api.domain.inference_detection import InferenceDetection
from yolo_inference_api.infrastructure.settings import YoloInferenceSettings


class YoloImageInferenceAdapter(ImageInferencePort):
    def __init__(self, settings: YoloInferenceSettings):
        self._settings = settings
        self._model = YOLO(settings.model)

    def infer(self, image_bytes: bytes) -> list[InferenceDetection]:
        with Image.open(BytesIO(image_bytes)) as image:
            source_image = image.convert("RGB")

        results = self._model.predict(
            source=source_image,
            conf=self._settings.confidence,
            iou=self._settings.iou,
            device=self._settings.device,
            verbose=False,
        )
        return self._map_results(results)

    @staticmethod
    def _map_results(results: Iterable[object]) -> list[InferenceDetection]:
        detections: list[InferenceDetection] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            names = getattr(result, "names", {})
            box_rows = _as_box_rows(getattr(boxes, "xyxy", []))
            class_ids = _as_class_ids(getattr(boxes, "cls", []))

            for box, class_id in zip(box_rows, class_ids):
                label = _resolve_label(names, class_id)
                detections.append(
                    InferenceDetection(
                        label=label,
                        x1=box[0],
                        y1=box[1],
                        x2=box[2],
                        y2=box[3],
                    )
                )

        return detections


def _resolve_label(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)


def _as_box_rows(raw_boxes: object) -> list[list[float]]:
    values = _to_list(raw_boxes)
    if not values:
        return []

    if isinstance(values[0], (float, int)):
        if len(values) < 4:
            return []
        return [[float(values[0]), float(values[1]), float(values[2]), float(values[3])]]

    rows: list[list[float]] = []
    for box in values:
        coordinates = _to_list(box)
        if len(coordinates) < 4:
            continue
        rows.append([
            float(coordinates[0]),
            float(coordinates[1]),
            float(coordinates[2]),
            float(coordinates[3]),
        ])
    return rows


def _as_class_ids(raw_classes: object) -> list[int]:
    values = _to_list(raw_classes)
    class_ids: list[int] = []

    for value in values:
        candidate = _to_list(value)
        scalar = candidate[0] if candidate else value
        try:
            class_ids.append(int(float(scalar)))
        except (TypeError, ValueError):
            continue

    return class_ids


def _to_list(value: object) -> list[Any]:
    if value is None:
        return []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]
