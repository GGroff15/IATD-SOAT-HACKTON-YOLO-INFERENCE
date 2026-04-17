from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class YoloInferenceSettings:
    model: str
    device: str
    confidence: float
    iou: float

    @classmethod
    def from_env(cls) -> "YoloInferenceSettings":
        return cls(
            model=_read_non_empty_env("YOLO_MODEL", "yolov8n.pt"),
            device=_read_non_empty_env("YOLO_DEVICE", "cpu"),
            confidence=_read_probability_env("YOLO_CONFIDENCE", 0.25),
            iou=_read_probability_env("YOLO_IOU", 0.70),
        )


def _read_non_empty_env(variable_name: str, default_value: str) -> str:
    raw_value = os.getenv(variable_name, default_value)
    value = raw_value.strip()
    if not value:
        raise ValueError(f"{variable_name} cannot be empty")
    return value


def _read_probability_env(variable_name: str, default_value: float) -> float:
    raw_value = os.getenv(variable_name)
    if raw_value is None:
        return default_value

    try:
        parsed = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{variable_name} must be a float between 0 and 1") from error

    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{variable_name} must be between 0 and 1")

    return parsed
