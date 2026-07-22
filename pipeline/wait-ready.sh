#!/usr/bin/env bash
# Wait until shiny-guide pipeline responds on PIPELINE_URL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$TRAINING_ROOT/run/env.sh"

URL="${PIPELINE_URL:-http://127.0.0.1:10006}"
TIMEOUT="${PIPELINE_WAIT_TIMEOUT:-7200}"
INTERVAL="${PIPELINE_WAIT_INTERVAL:-15}"

echo "==> Waiting for pipeline at $URL (timeout ${TIMEOUT}s)"
deadline=$((SECONDS + TIMEOUT))
while (( SECONDS < deadline )); do
  if curl -sf "$URL/health" >/dev/null 2>&1; then
    echo "==> Pipeline ready"
    curl -s "$URL/status" 2>/dev/null | head -20 || true
    exit 0
  fi
  echo "    ... not ready yet (${SECONDS}s elapsed)"
  sleep "$INTERVAL"
done

echo "ERROR: pipeline not ready after ${TIMEOUT}s" >&2
echo "Check: $PIPELINE_DIR/runs/pipeline-server.log" >&2
exit 1
