# YOLO Inference API

FastAPI-based YOLOv8 inference API project.

## Prerequisites

- Python 3.12+
- uv
- Docker or Podman

Install uv if needed:

```bash
pip install uv
```

## Run as a Python Project with uv

1. Create an environment file if needed:

```bash
cp .env.example .env
```

For PowerShell:

```powershell
Copy-Item .env.example .env
```

2. Sync dependencies:

```bash
uv sync
```

2. Create your local environment file from the example and adjust values if needed:

```bash
cp .env.example .env
```

YOLO runtime keys:

- `YOLO_MODEL` (default: `yolov8n.pt`)
- `YOLO_DEVICE` (default: `cpu`; examples: `cpu`, `gpu`, `0`, `0,1`, `cuda`, `cuda:0`)
- `YOLO_CONFIDENCE` (default: `0.25`)
- `YOLO_IOU` (default: `0.70`)

If `YOLO_DEVICE` is set to `gpu`/`cuda*` on a machine without CUDA, the API automatically falls back to `cpu`.

3. Run the current scaffold entry point:

```bash
uv run python main.py
```

4. Run the API server:
4. Run the API server:

```bash
uv run uvicorn main:app --reload
```

When the API is running, it is available at `http://127.0.0.1:8000` and interactive docs are at `http://127.0.0.1:8000/docs`.

## Run with Docker

This workflow builds a single image from `Dockerfile`.  
At container startup, the entrypoint downloads the YOLO model from S3 **only if** the file configured in `YOLO_MODEL` does not already exist.

Runtime environment keys for model download:

- `YOLO_MODEL` (optional, default: `models/yolov8_component_arrow.pt`)
- `YOLO_MODEL_S3_OBJECT_URL` (required when `YOLO_MODEL` file is not already present in the container)

1. Build the image (bash):

```bash
docker build -t yolo-inference-api:local .
```

2. Build the image (PowerShell):

```powershell
docker build -t yolo-inference-api:local .
```

3. Run the container:

```bash
docker run --rm -p 8000:8000 --env-file .env \
	-e YOLO_MODEL_S3_OBJECT_URL=<https://your-bucket-or-s3-url/model.pt> \
	yolo-inference-api:local
```

## Run with Podman

1. Build the image:

```bash
podman build -t yolo-inference-api:local .
```

2. Run the container:

```bash
podman run --rm -p 8000:8000 --env-file .env \
	-e YOLO_MODEL_S3_OBJECT_URL=<https://your-bucket-or-s3-url/model.pt> \
	yolo-inference-api:local
```

## Development Commands

- Add runtime dependency: `uv add <package>`
- Add development dependency: `uv add --dev <package>`
- Run tests: `uv run pytest`
- Build package artifacts: `uv build`
