from io import BytesIO

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from yolo_inference_api.application.image_inference_use_case import ImageInferenceUseCase
from yolo_inference_api.domain.inference_detection import InferenceDetection


router = APIRouter()


class BoundingBoxResponse(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResponse(BaseModel):
    label: str
    bbox: BoundingBoxResponse


class InferResponse(BaseModel):
    detections: list[DetectionResponse]


def get_inference_use_case() -> ImageInferenceUseCase:
    raise HTTPException(status_code=503, detail="Inference use case not configured")


def serialize_detection(detection: InferenceDetection) -> dict[str, object]:
    return {
        "label": detection.label,
        "bbox": {
            "x1": detection.x1,
            "y1": detection.y1,
            "x2": detection.x2,
            "y2": detection.y2,
        },
    }


def _validate_image_bytes(image_bytes: bytes) -> None:
    if not image_bytes:
        raise ValueError("Missing image bytes")

    with Image.open(BytesIO(image_bytes)) as image:
        image.verify()


@router.post("/infer", response_model=InferResponse)
async def infer(
    file: UploadFile = File(...),
    use_case: ImageInferenceUseCase = Depends(get_inference_use_case),
) -> InferResponse:
    image_bytes = await file.read()

    try:
        _validate_image_bytes(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Invalid image file") from error

    detections = use_case.infer(image_bytes)
    serialized_detections = [
        DetectionResponse.model_validate(serialize_detection(detection))
        for detection in detections
    ]
    return InferResponse(detections=serialized_detections)
