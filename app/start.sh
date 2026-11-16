#!/usr/bin/env bash
# Starts the API.
set -euo pipefail
PYTHON="${PYTHON:-python}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PYTHON" -m uvicorn server:app --app-dir "$APP_DIR" --host 0.0.0.0 --port "${PORT:-8000}"
