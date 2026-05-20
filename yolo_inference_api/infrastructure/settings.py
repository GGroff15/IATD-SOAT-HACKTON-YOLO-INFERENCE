from dataclasses import dataclass
import logging
import os
import re


LOGGER = logging.getLogger(__name__)
_CUDA_DEVICE_PATTERN = re.compile(r"^cuda(?::(?P<indices>\d+(?:,\d+)*))?$")


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
            device=_read_device_env("YOLO_DEVICE", "cpu"),
            confidence=_read_probability_env("YOLO_CONFIDENCE", 0.25),
            iou=_read_probability_env("YOLO_IOU", 0.70),
        )


def _read_device_env(variable_name: str, default_value: str) -> str:
    raw_value = _read_non_empty_env(variable_name, default_value)
    return _normalize_device(raw_value, variable_name)


def _read_non_empty_env(variable_name: str, default_value: str) -> str:
    raw_value = os.getenv(variable_name, default_value)
    value = raw_value.strip()
    if not value:
        raise ValueError(f"{variable_name} cannot be empty")
    return value


def _normalize_device(raw_device_value: str, variable_name: str) -> str:
    normalized_value = raw_device_value.strip().lower()

    if normalized_value == "gpu":
        if _is_cuda_available():
            return "0"
        LOGGER.warning(
            "%s=%r requires CUDA but no CUDA devices were detected. Falling back to 'cpu'.",
            variable_name,
            raw_device_value,
        )
        return "cpu"

    cuda_match = _CUDA_DEVICE_PATTERN.fullmatch(normalized_value)
    if cuda_match:
        if _is_cuda_available():
            indices = cuda_match.group("indices")
            return indices or "0"
        LOGGER.warning(
            "%s=%r requires CUDA but no CUDA devices were detected. Falling back to 'cpu'.",
            variable_name,
            raw_device_value,
        )
        return "cpu"

    return normalized_value


def _is_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False

    return bool(torch.cuda.is_available())


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
