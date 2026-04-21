# YOLO Inference API

FastAPI-based YOLOv8 inference API project.

## Prerequisites

- **Docker workflow (required to run container):**
  - Docker Desktop or Docker Engine
  - Network access to `YOLO_MODEL_S3_OBJECT_URL` when the model file does not already exist in the container
- **Local Python workflow (optional, for running outside Docker):**
  - Python 3.12+
  - uv

Install uv if needed:

```bash
pip install uv
```

## Run as a Python Project with uv

1. Create your local environment file:

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

```bash
uv run uvicorn main:app --reload
```

When the API is running, it is available at `http://127.0.0.1:8000` and interactive docs are at `http://127.0.0.1:8000/docs`.

## Run with Docker

The Docker image installs project dependencies from `pyproject.toml` during build (`uv sync --frozen --no-dev --no-install-project`), so you do not need local Python packages inside the container.
At startup, the entrypoint downloads the YOLO model **only if** the file configured in `YOLO_MODEL` does not already exist.

1. Create a Docker environment file from the example:

```bash
cp .docker.env.example .docker.env
```

For PowerShell:

```powershell
Copy-Item .docker.env.example .docker.env
```

2. Configure `.docker.env`:

- `YOLO_MODEL` (optional, default: `models/yolov8_component_arrow.pt`)
- `YOLO_MODEL_S3_OBJECT_URL` (**required** when the `YOLO_MODEL` file is not present in the container)
- `YOLO_DEVICE`, `YOLO_CONFIDENCE`, `YOLO_IOU` (optional inference settings)

3. Build the image:

```bash
docker build -t yolo-inference-api:local .
```

4. Run the container:

```bash
docker run --rm -p 8000:8000 --env-file .docker.env yolo-inference-api:local
```

When the API is running, it is available at `http://127.0.0.1:8000` and interactive docs are at `http://127.0.0.1:8000/docs`.

## Run with Podman

1. Build the image (same build args as Docker):

```bash
podman build \
	--build-arg YOLO_MODEL_S3_BUCKET=<your-bucket> \
	--build-arg YOLO_MODEL_S3_KEY=<path/in/bucket/model.pt> \
	--build-arg AWS_DEFAULT_REGION=<aws-region> \
	--build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
	--build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
	--build-arg AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
	-t yolo-inference-api:local .
```

2. Run the container:

```bash
podman run --rm -p 8000:8000 --env-file .docker.env \
	-e YOLO_MODEL_S3_OBJECT_URL=<https://your-bucket-or-s3-url/model.pt> \
	yolo-inference-api:local
```

## Development Commands

- Add runtime dependency: `uv add <package>`
- Add development dependency: `uv add --dev <package>`
- Run tests: `uv run pytest`
- Build package artifacts: `uv build`
