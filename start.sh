#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -f .env ] || ! grep -q '^WEBUI_SECRET=' .env; then
  echo "Generating WEBUI_SECRET..."
  echo "WEBUI_SECRET=$(openssl rand -hex 32)" > .env
fi

. ./.env

PDF_VISION_MODEL="${PDF_VISION_MODEL:-Qwen3.8 Flash Next}"
PDF_VISION_API_BASE_URL="${PDF_VISION_API_BASE_URL:-http://127.0.0.1:8080/v1}"

podman build -t open-webui:vision pdf-vision

podman run -d \
  --name open-webui \
  --network host \
  -e WEBUI_SECRET="$WEBUI_SECRET" \
  -e WEBUI_HOST="0.0.0.0" \
  -e PORT="3000" \
  -e OPENAI_API_BASE_URLS='["http://127.0.0.1:8080/v1","http://127.0.0.1:8081/v1"]' \
  -e OLLAMA_BASE_URLS='[]' \
  -e ENABLE_SIGNUP="true" \
  -e PDF_VISION_MODEL="$PDF_VISION_MODEL" \
  -e PDF_VISION_API_BASE_URL="$PDF_VISION_API_BASE_URL" \
  -v open-webui:/app/backend/data \
  --restart unless-stopped \
  open-webui:vision

echo "Waiting for Open WebUI to start..."
for i in $(seq 1 90); do
  if curl -s -o /dev/null http://127.0.0.1:3000/ 2>/dev/null; then
    echo "Open WebUI is running!"
    echo "Local:    http://halo-server.local:3000"
    echo "LAN IP:   http://192.168.2.70:3000"
    break
  fi
  sleep 1
done
