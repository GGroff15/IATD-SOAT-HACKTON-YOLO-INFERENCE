from yolo_inference_api.infrastructure import settings as settings_module


def test_from_env_maps_gpu_to_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setenv("YOLO_DEVICE", "gpu")
    monkeypatch.setattr(settings_module, "_is_cuda_available", lambda: False)

    settings = settings_module.YoloInferenceSettings.from_env()

    assert settings.device == "cpu"


def test_from_env_maps_gpu_to_first_cuda_device_when_cuda_is_available(monkeypatch):
    monkeypatch.setenv("YOLO_DEVICE", "gpu")
    monkeypatch.setattr(settings_module, "_is_cuda_available", lambda: True)

    settings = settings_module.YoloInferenceSettings.from_env()

    assert settings.device == "0"


def test_from_env_maps_cuda_prefix_to_index_list_when_cuda_is_available(monkeypatch):
    monkeypatch.setenv("YOLO_DEVICE", "cuda:0,1")
    monkeypatch.setattr(settings_module, "_is_cuda_available", lambda: True)

    settings = settings_module.YoloInferenceSettings.from_env()

    assert settings.device == "0,1"


def test_from_env_maps_cuda_prefix_to_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setenv("YOLO_DEVICE", "cuda:0")
    monkeypatch.setattr(settings_module, "_is_cuda_available", lambda: False)

    settings = settings_module.YoloInferenceSettings.from_env()

    assert settings.device == "cpu"
