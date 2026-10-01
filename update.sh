#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

BASE_IMAGE="ghcr.io/open-webui/open-webui:main"

echo "==> Pulling updated base image..."
podman pull "$BASE_IMAGE"

echo "==> Rebuilding open-webui:vision..."
podman build -t open-webui:vision pdf-vision

echo "==> Verifying patch survived the update..."
podman run --rm open-webui:vision sh -c '
  grep -q TRANSCRIBE_PROMPT /app/backend/open_webui/retrieval/loaders/pdf.py || { echo "FAIL: patched loader not in image"; exit 1; }
  python -c "
import sys
sys.path.insert(0, \"/app/backend\")
import fitz
from open_webui.retrieval.loaders.pdf import PDFLoader
loader = PDFLoader(\"/dev/null\")
import inspect
sig = inspect.signature(PDFLoader.__init__)
for expected in (\"extract_images\", \"mode\"):
    if expected not in sig.parameters:
        raise SystemExit(f\"FAIL: upstream PDFLoader signature changed: {sig}\")
print(\"patch ok: fitz\", fitz.version[0], \"| loader interface intact\")
"
'

echo "==> Replacing running container..."
podman stop open-webui 2>/dev/null || true
podman rm open-webui 2>/dev/null || true

./start.sh