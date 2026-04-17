# Project Guidelines

## Stack
- Service goal: FastAPI-based YOLOv8 inference API.
- Use `uv` for dependency management and command execution.
- Python version: 3.12+ (see `.python-version`).
- Project metadata and dependencies live in `pyproject.toml`.
- The service must run both locally and as a Docker container.

## Build And Run
- Sync dependencies: `uv sync`
- Add runtime dependencies: `uv add <package>`
- Add development dependencies: `uv add --dev <package>`
- Run local API server: `uv run uvicorn main:app --reload`
- Run current scaffold entry point: `uv run python main.py`
- Build Docker image: `docker build -t yolo-inference-api:local .`
- Run Docker container: `docker run --rm -p 8000:8000 --env-file .env yolo-inference-api:local`
- Build package artifacts: `uv build`
- Run tests: `uv run pytest`

## Architecture
- Follow hexagonal architecture (ports and adapters) for all new code.
- Keep domain logic framework-agnostic: no FastAPI, YOLO SDK, or infrastructure imports in domain modules.
- Keep application/use-case services dependent on domain ports, not concrete adapters.
- Keep inbound adapters responsible for transport concerns only (HTTP parsing, validation, serialization).
- Keep outbound adapters responsible for YOLO model and other external integrations.
- Compose dependencies at startup and inject adapters into use cases.
- Keep request/response payloads explicit and stable.

## Containerization
- Keep `Dockerfile` and `.dockerignore` present and updated with dependency and runtime changes.
- Ensure the container starts the FastAPI app via `uvicorn` and respects environment-based configuration.
- Keep container builds reproducible and avoid embedding secrets in image layers.

## Configuration
- Keep all YOLOv8 configuration keys documented in `.env.example`.
- When adding or renaming environment variables, update `.env.example` in the same change.
- Keep model path/name, device selection, confidence, IoU, and related inference knobs configurable via environment variables.
- Never commit secrets; `.env.example` should contain only safe defaults and examples.

## Documentation
- Keep quick-start and API usage docs in `README.md`.
- Link to detailed docs from there instead of duplicating long guidance in customization files.
