from io import BytesIO

from PIL import Image

from yolo_inference_api.adapters.outbound import yolo_adapter
from yolo_inference_api.adapters.outbound.yolo_adapter import YoloImageInferenceAdapter
from yolo_inference_api.infrastructure.settings import YoloInferenceSettings


class FakeTensor:
    def __init__(self, value):
        self._value = value

    def tolist(self):
        return self._value


class FakeBoxes:
    def __init__(self, xyxy, cls):
        self.xyxy = FakeTensor(xyxy)
        self.cls = FakeTensor(cls)


class FakeResult:
    def __init__(self, boxes=None, names=None):
        self.boxes = boxes
        self.names = names or {}


class FakeModel:
    def __init__(self, results):
        self._results = results
        self.predict_calls = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return self._results


def _make_png_bytes() -> bytes:
    image = Image.new("RGB", (16, 16), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_map_results_returns_detection_objects():
    results = [
        FakeResult(
            boxes=FakeBoxes(
                xyxy=[[10.0, 20.0, 110.0, 220.0], [1.0, 2.0, 3.0, 4.0]],
                cls=[0.0, 1.0],
            ),
            names={0: "person", 1: "dog"},
        )
    ]

    detections = YoloImageInferenceAdapter._map_results(results)

    assert len(detections) == 2
    assert detections[0].label == "person"
    assert detections[0].x1 == 10.0
    assert detections[0].y1 == 20.0
    assert detections[0].x2 == 110.0
    assert detections[0].y2 == 220.0
    assert detections[1].label == "dog"


def test_map_results_returns_empty_list_without_boxes():
    detections = YoloImageInferenceAdapter._map_results([FakeResult(boxes=None)])

    assert detections == []


def test_infer_calls_predict_with_configured_thresholds(monkeypatch):
    fake_results = [
        FakeResult(
            boxes=FakeBoxes(xyxy=[[5.0, 6.0, 7.0, 8.0]], cls=[0.0]),
            names={0: "cat"},
        )
    ]
    fake_model = FakeModel(fake_results)

    def fake_yolo_factory(model_name: str):
        assert model_name == "yolov8n.pt"
        return fake_model

    monkeypatch.setattr(yolo_adapter, "YOLO", fake_yolo_factory)

    settings = YoloInferenceSettings(
        model="yolov8n.pt",
        device="cpu",
        confidence=0.35,
        iou=0.55,
    )
    adapter = YoloImageInferenceAdapter(settings=settings)

    detections = adapter.infer(_make_png_bytes())

    assert len(fake_model.predict_calls) == 1
    predict_kwargs = fake_model.predict_calls[0]
    assert predict_kwargs["conf"] == 0.35
    assert predict_kwargs["iou"] == 0.55
    assert predict_kwargs["device"] == "cpu"
    assert predict_kwargs["verbose"] is False
    assert predict_kwargs["source"].size == (16, 16)

    assert len(detections) == 1
    assert detections[0].label == "cat"
    assert detections[0].x1 == 5.0
    assert detections[0].y1 == 6.0
    assert detections[0].x2 == 7.0
    assert detections[0].y2 == 8.0
