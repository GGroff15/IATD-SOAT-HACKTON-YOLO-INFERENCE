## Plan: TDD Controller-Only Image Inference Endpoint

Implement a POST /infer endpoint using multipart image upload, returning only label plus bbox edges (x1, y1, x2, y2). Follow strict TDD by writing integration and unit tests first, then implement only inbound controller code plus the application use-case interface. Domain and outbound layers remain untouched.

**Steps**
1. Phase 1 - Baseline setup (blocks all other steps): update pyproject.toml with runtime dependencies (FastAPI, uvicorn, python-multipart, Pillow, pydantic) and test dependencies (pytest, pytest-asyncio, httpx), and convert main.py from scaffold print script into an importable FastAPI app entrypoint.
2. Phase 2 - Integration tests first (depends on 1): add a shared test fixture module under tests and create endpoint tests under tests/integration that assert 422 for missing file, 400 for invalid image bytes, 200 with populated detections, and 200 with empty detections.
3. Phase 3 - Unit tests first for in-scope code (parallel with 2 after 1): add unit tests under tests/unit for the use-case interface contract (protocol compatibility with fake/mock) and for controller response contract shape (label and bbox edge keys only).
4. Phase 4 - Implement use-case interface only (depends on 2 and 3): add the application interface module in yolo_inference_api/application with a protocol method that accepts image bytes and returns detection DTOs needed by the controller.
5. Phase 5 - Implement inbound controller only (depends on 2 and 3): add the HTTP controller module in yolo_inference_api/adapters/inbound with POST /infer, UploadFile parsing, image validation, dependency-injected interface call, and response serialization containing label and bbox edges only.
6. Phase 6 - App wiring (depends on 4 and 5): wire router registration and dependency hook in main.py without adding concrete domain or outbound implementations.
7. Phase 7 - Verify and stabilize (depends on 6): run tests in TDD order (unit, integration, full suite) and confirm Swagger docs render expected multipart request and response schema.

**Relevant files**
- main.py
- pyproject.toml
- tests
- tests/integration
- tests/unit
- yolo_inference_api/adapters/inbound
- yolo_inference_api/application
- AGENTS.md

**Verification**
1. Run uv sync.
2. Run uv run pytest tests/unit -q.
3. Run uv run pytest tests/integration -q.
4. Run uv run pytest -q.
5. Run uv run uvicorn main:app --reload and check /docs for POST /infer multipart input and detections response contract.

**Decisions**
- Endpoint path: POST /infer.
- Input format: multipart/form-data with file field named file.
- Detection fields: label plus bbox edges x1, y1, x2, y2.
- Scope boundary: only inbound controller layer and application use-case interface.
- Testing strategy: tests first, production code second (strict TDD).