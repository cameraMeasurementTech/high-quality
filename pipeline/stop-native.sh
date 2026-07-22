#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/runs/pipeline.pid"
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  kill "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "Stopped pipeline pid $pid"
else
  echo "No pipeline pid file"
fi
