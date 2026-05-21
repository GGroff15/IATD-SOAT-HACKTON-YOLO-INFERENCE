# YOLO Inference API

FastAPI service for running YOLOv8 image inference and returning detected labels with bounding boxes.

The project uses a small Clean Architecture-style layout:

- `main.py`: FastAPI application factory, dependency wiring, and observability setup.
- `yolo_inference_api/adapters/inbound`: HTTP controller for the inference endpoint.
- `yolo_inference_api/adapters/outbound`: Ultralytics YOLO adapter.
- `yolo_inference_api/application`: inference use case protocol and service.
- `yolo_inference_api/domain`: inference contracts and detection entity.
- `yolo_inference_api/infrastructure`: environment settings and OpenTelemetry configuration.
- `models`: local model files used by the API and Docker Compose volume.
- `tests`: unit and integration tests for settings, controller contract, use case, and YOLO adapter behavior.

## Requirements

For local Python execution:

- Python 3.12
- `uv`

For container execution:

- Docker or Docker Compose
- A YOLO model file available locally or a downloadable model URL in `YOLO_MODEL_S3_OBJECT_URL`

Install `uv` if it is not already available:

```bash
pip install uv
```

## Configuration

The application reads runtime configuration from environment variables.

| Variable | Default | Description |
| --- | --- | --- |
| `YOLO_MODEL` | `yolov8n.pt` in app code, `models/yolov8_component_arrow.pt` in the provided env files | Path or model name passed to Ultralytics YOLO. |
| `YOLO_MODEL_S3_OBJECT_URL` | Empty | URL used by the Docker entrypoint to download the model when `YOLO_MODEL` is missing. |
| `YOLO_DEVICE` | `cpu` | Inference device. Supported examples: `cpu`, `gpu`, `cuda`, `cuda:0`, `cuda:0,1`, `0`, `0,1`. |
| `YOLO_CONFIDENCE` | `0.25` | Detection confidence threshold from `0` to `1`. |
| `YOLO_IOU` | `0.70` | Intersection over Union threshold from `0` to `1`. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Empty | Enables OpenTelemetry traces, metrics, logs, and request logging when set. |

If `YOLO_DEVICE` is set to `gpu` or `cuda*` on a machine without CUDA, the API falls back to `cpu`.

The included example files use:

```env
YOLO_MODEL=models/yolov8_component_arrow.pt
YOLO_DEVICE=cpu
YOLO_CONFIDENCE=0.80
YOLO_IOU=0.70
```

## Model Files

The repository includes model weights in the root and under `models/`. The default path used by the provided local and Docker environment examples is:

```text
models/yolov8_component_arrow.pt
```

When running with Docker or Docker Compose, `/app/models` is expected to contain the configured model. The Docker entrypoint checks `YOLO_MODEL` at startup:

- If the model exists, it starts the API with the existing file.
- If the model is missing and `YOLO_MODEL_S3_OBJECT_URL` is set, it downloads the file before starting.
- If the model is missing and `YOLO_MODEL_S3_OBJECT_URL` is empty, startup fails.

## Run Locally with uv

Create the local environment file:

```bash
cp .env.example .env
```

PowerShell:

```powershell
Copy-Item .env.example .env
```

Install dependencies:

```bash
uv sync
```

Start the API using the environment file:

```bash
uv run --env-file .env uvicorn main:app --reload
```

The API will be available at:

- API: `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

If you already exported the required environment variables in your shell, this also works:

```bash
uv run uvicorn main:app --reload
```

## Run with Docker

Create the Docker environment file:

```bash
cp .docker.env.example .docker.env
```

PowerShell:

```powershell
Copy-Item .docker.env.example .docker.env
```

Edit `.docker.env` if you need a different model path, device, threshold, or model download URL.

Build the image:

```bash
docker build -t yolo-inference-api:local .
```

Run the container:

```bash
docker run --rm -p 8000:8000 --env-file .docker.env yolo-inference-api:local
```

The API will be available at:

- API: `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`

## Run with Docker Compose

The Compose configuration:

- Builds the local `Dockerfile`.
- Loads `.docker.env`.
- Mounts `./models` to `/app/models`.
- Exposes container port `8000` on host port `8081`.
- Uses the external Docker network `soat-net`.
- Configures OTLP export to `http://otel-collector:4318`.

Create the external network once:

```bash
docker network create soat-net
```

Start the service:

```bash
docker compose up --build
```

Stop the service:

```bash
docker compose down
```

With Docker Compose, the API will be available at:

- API: `http://127.0.0.1:8081`
- Interactive docs: `http://127.0.0.1:8081/docs`
- Healthcheck target: `http://127.0.0.1:8081/openapi.json`

## API

### `POST /infer`

Runs inference on an uploaded image.

Request:

- Content type: `multipart/form-data`
- File field name: `file`
- File content: valid image bytes readable by Pillow, such as PNG or JPEG

Example:

```bash
curl -X POST "http://127.0.0.1:8000/infer" \
  -F "file=@path/to/image.png"
```

Docker Compose example:

```bash
curl -X POST "http://127.0.0.1:8081/infer" \
  -F "file=@path/to/image.png"
```

Success response:

```json
{
  "detections": [
    {
      "label": "component",
      "bbox": {
        "x1": 10.0,
        "y1": 20.0,
        "x2": 110.0,
        "y2": 220.0
      }
    }
  ]
}
```

If no objects are detected:

```json
{
  "detections": []
}
```

Error responses:

| Status | Cause |
| --- | --- |
| `400` | Invalid image bytes. Response: `{"detail":"Invalid image file"}`. |
| `422` | Missing required multipart field `file`. |
| `503` | Inference use case is not configured. |

## Observability

OpenTelemetry is optional and is enabled only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.

When enabled, the application configures:

- FastAPI instrumentation.
- OTLP traces.
- OTLP metrics.
- OTLP logs.
- JSON request logs containing method, path, status code, duration, trace ID, and span ID when available.

Docker Compose sets OTLP environment variables for a collector reachable as `otel-collector` on the `soat-net` network.

## Running Tests

Install development dependencies:

```bash
uv sync --dev
```

Copy the environment file before running tests that read settings:

```bash
cp .env.example .env
```

### Unit tests

Run the full unit test suite:

```bash
uv run pytest tests/unit/
```

Run with coverage report (minimum required: 80%):

```bash
uv run pytest tests/unit/ --cov=yolo_inference_api --cov-report=term-missing --cov-fail-under=80
```

The unit tests cover: settings validation, YOLO adapter mapping, inference service logic, controller contract, and use case protocol. They do not load any model file.

### Integration tests

Integration tests load the actual YOLO model and require the model file to be present at the path configured in `YOLO_MODEL`:

```bash
uv run pytest tests/integration/
```

Integration tests are not executed in the CI pipeline because they require the model file which is not committed to the repository.

### All tests

```bash
uv run pytest
```

### Coverage only

```bash
uv run pytest --cov=yolo_inference_api --cov-report=html
```

The HTML report is written to `htmlcov/index.html`.

## CI/CD

The repository uses GitHub Actions with a workflow at `.github/workflows/ci.yml` that runs automatically on every pull request targeting `main`.

The pipeline has two sequential jobs:

| Job | What it does |
| --- | --- |
| `run_tests` | Installs dependencies with `uv sync --dev`, copies `.env.example` to `.env`, runs `pytest tests/unit/` with 80% coverage enforcement. Fails the build if coverage drops below 80%. |
| `build_and_push` | Builds the Docker image and pushes `{DOCKER_USERNAME}/soat-yolo-inference:latest` to Docker Hub. Runs only after `run_tests` succeeds. |

Required GitHub secrets: `DOCKER_USERNAME` and `DOCKER_PASSWORD`.

> Integration tests are excluded from the pipeline because they require the YOLO model file which is not bundled in the CI environment.

## API Testing

There is no Postman collection in this repository. Use the auto-generated interactive docs instead:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Raw schema: `http://127.0.0.1:8000/openapi.json`

Quick curl test after the service is running:

```bash
curl -s -X POST http://127.0.0.1:8000/infer \
  -F "file=@path/to/image.png" | python3 -m json.tool
```

## Development

Add dependencies:

```bash
uv add <package>
```

Add development dependencies:

```bash
uv add --dev <package>
```

## Equipe

### Integrantes IADT

| Nome | RM |
|---|---|
| Angelo Rossi | RM365902 |
| Carlos Eduardo | RM365213 |
| Felipe Goiabeira | RM365753 |
| Guilherme Groff | RM365281 |
| Rafael Lua | RM366254 |

### Integrantes SOAT

| Nome | RM |
|---|---|
| Felipe Alves de Oliveira | RM365154 |
| Nicolas Henrique Correa Martins | RM365746 |
| William Francisco Leite | RM365973 |
