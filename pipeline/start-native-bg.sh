#!/usr/bin/env bash
# Start shiny-guide native pipeline in background.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$TRAINING_ROOT/run/env.sh"

PID_FILE="$PIPELINE_DIR/runs/pipeline.pid"
LOG_FILE="$PIPELINE_DIR/runs/pipeline-server.log"

if [[ -f "$PID_FILE" ]]; then
  old_pid=$(cat "$PID_FILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "==> Pipeline already running (pid $old_pid)"
    exit 0
  fi
fi

mkdir -p "$PIPELINE_DIR/runs"
require_shiny_guide
require_openrouter_if_needed

echo "==> Starting pipeline in background"
echo "    log: $LOG_FILE"
nohup "$PIPELINE_DIR/run-native.sh" >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "    pid $(cat "$PID_FILE")"
