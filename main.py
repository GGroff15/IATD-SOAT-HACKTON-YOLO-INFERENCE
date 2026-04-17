from fastapi import FastAPI, HTTPException

from yolo_inference_api.adapters.inbound.infer_controller import (
	get_inference_use_case,
	router as infer_router,
)
from yolo_inference_api.adapters.outbound.yolo_adapter import YoloImageInferenceAdapter
from yolo_inference_api.application.image_inference_use_case import (
	ImageInferenceService,
	ImageInferenceUseCase,
)
from yolo_inference_api.infrastructure.settings import YoloInferenceSettings


def create_app() -> FastAPI:
	application = FastAPI(title="YOLO Inference API")
	application.include_router(infer_router)
	configured_use_case: ImageInferenceUseCase | None = None

	def get_inference_use_case_dependency() -> ImageInferenceUseCase:
		if configured_use_case is None:
			raise HTTPException(status_code=503, detail="Inference use case not configured")
		return configured_use_case

	@application.on_event("startup")
	def configure_inference_use_case() -> None:
		nonlocal configured_use_case
		active_override = application.dependency_overrides.get(get_inference_use_case)
		if active_override is not None and active_override is not get_inference_use_case_dependency:
			return

		settings = YoloInferenceSettings.from_env()
		inference_port = YoloImageInferenceAdapter(settings=settings)
		configured_use_case = ImageInferenceService(inference_port=inference_port)

	application.dependency_overrides[get_inference_use_case] = get_inference_use_case_dependency
	return application


app = create_app()
