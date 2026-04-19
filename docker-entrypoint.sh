#!/bin/sh
set -eu

MODEL_PATH="${YOLO_MODEL:-models/yolov8_component_arrow.pt}"
MODEL_URL="${YOLO_MODEL_S3_OBJECT_URL:-}"

if [ ! -f "${MODEL_PATH}" ]; then
	if [ -z "${MODEL_URL}" ]; then
		echo "ERROR: YOLO model not found at '${MODEL_PATH}' and YOLO_MODEL_S3_OBJECT_URL is not set." >&2
		exit 1
	fi

	mkdir -p "$(dirname "${MODEL_PATH}")"
	echo "Downloading YOLO model to ${MODEL_PATH}..."
	curl --fail --location --silent --show-error "${MODEL_URL}" --output "${MODEL_PATH}"

	if [ ! -s "${MODEL_PATH}" ]; then
		echo "ERROR: Downloaded model file is empty: '${MODEL_PATH}'." >&2
		exit 1
	fi
else
	echo "Using existing YOLO model at ${MODEL_PATH}."
fi

exec "$@"
