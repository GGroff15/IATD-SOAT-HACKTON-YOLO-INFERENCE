from io import BytesIO

import pytest
from PIL import Image

from yolo_inference_api.adapters.inbound.infer_controller import get_inference_use_case
from yolo_inference_api.domain.inference_detection import InferenceDetection


class FakeInferenceUseCase:
    def __init__(self, detections: list[InferenceDetection]):
        self._detections = detections

    def infer(self, image_bytes: bytes) -> list[InferenceDetection]:
        return self._detections


def _make_png_bytes() -> bytes:
    image = Image.new("RGB", (16, 16), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_infer_returns_422_when_file_is_missing(async_client, app_instance):
    app_instance.dependency_overrides[get_inference_use_case] = lambda: FakeInferenceUseCase([])

    response = await async_client.post("/infer")

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_infer_returns_400_for_invalid_image_bytes(async_client, app_instance):
    app_instance.dependency_overrides[get_inference_use_case] = lambda: FakeInferenceUseCase([])

    response = await async_client.post(
        "/infer",
        files={"file": ("not-an-image.bin", b"definitely-not-image-bytes", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid image file"}


@pytest.mark.asyncio
async def test_infer_returns_200_with_populated_detections(async_client, app_instance):
    detections = [
        InferenceDetection(label="person", x1=11.0, y1=22.0, x2=111.0, y2=222.0),
        InferenceDetection(label="dog", x1=1.0, y1=2.0, x2=3.0, y2=4.0),
    ]
    app_instance.dependency_overrides[get_inference_use_case] = lambda: FakeInferenceUseCase(detections)

    response = await async_client.post(
        "/infer",
        files={"file": ("image.png", _make_png_bytes(), "image/png")},
    )

    assert response.status_code == 200

    payload = response.json()
    assert "detections" in payload
    assert len(payload["detections"]) == 2

    first = payload["detections"][0]
    assert first["label"] == "person"
    assert set(first.keys()) == {"label", "bbox"}
    assert set(first["bbox"].keys()) == {"x1", "y1", "x2", "y2"}


@pytest.mark.asyncio
async def test_infer_returns_200_with_empty_detections(async_client, app_instance):
    app_instance.dependency_overrides[get_inference_use_case] = lambda: FakeInferenceUseCase([])

    response = await async_client.post(
        "/infer",
        files={"file": ("image.png", _make_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {"detections": []}
