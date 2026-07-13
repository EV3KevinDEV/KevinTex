#!/usr/bin/env bash
# Start KevinTex and open it in the browser.
cd "$(dirname "$0")"
./.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8321 &
SERVER_PID=$!
sleep 2
xdg-open http://127.0.0.1:8321 >/dev/null 2>&1 || true
wait $SERVER_PID
