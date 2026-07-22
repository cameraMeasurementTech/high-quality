#!/usr/bin/env bash
# Start shiny-guide natively on GPU — standalone (no Docker, no full monorepo).
#
# Prereq: ./setup-native.sh
# Env:    OPENROUTER_API_KEY (+ HF_TOKEN recommended)
#         MODEL_PATH=/path/to/local/AstroWolf  (optional)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$TRAINING_ROOT/run/env.sh"

if [[ -f "$TRAINING_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$TRAINING_ROOT/.env"
  set +a
fi

BASE_CONFIG="${CONFIG_FILE:-$PIPELINE_DIR/configuration.gpu-native.yaml}"
export NODE_CWD="${NODE_CWD:-$SHINY_GUIDE_ROOT/docker}"
export PYTHONPATH="$SHINY_GUIDE_ROOT/pipeline_service:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$TRAINING_ROOT/data/hf_cache}"

VLLM_VENV="$PIPELINE_DIR/.vllm-env"
VLLM_BIN="${VLLM_BIN:-$VLLM_VENV/bin/vllm}"
PIPE_VENV="$PIPELINE_DIR/.venv"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: Set OPENROUTER_API_KEY in $TRAINING_ROOT/.env"
  exit 1
fi

for req in "$PIPE_VENV/bin/python" "$VLLM_BIN" "$BASE_CONFIG"; do
  if [[ ! -e "$req" ]]; then
    echo "ERROR: Missing $req — run $PIPELINE_DIR/setup-native.sh"
    exit 1
  fi
done

if [[ ! -d "$NODE_CWD/node_modules" ]]; then
  echo "ERROR: Missing $NODE_CWD/node_modules — run setup-native.sh"
  exit 1
fi

resolve_node_ge_20() {
  local candidate ver major
  for candidate in ${NODE_BIN:-} node; do
    [[ -n "$candidate" ]] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    ver=$("$candidate" -v 2>/dev/null | sed 's/^v//')
    major=${ver%%.*}
    [[ "$major" =~ ^[0-9]+$ && "$major" -ge 20 ]] && { echo "$candidate"; return 0; }
  done
  return 1
}
NODE20=$(resolve_node_ge_20) || {
  echo "ERROR: Node.js >= 20 required"
  exit 1
}
export PATH="$(dirname "$NODE20"):$PATH"

mkdir -p "$PIPELINE_DIR/runs"
RUNTIME_CONFIG="$PIPELINE_DIR/runs/configuration.runtime.yaml"

# vllm_bin path
sed "s|vllm_bin: \"/opt/vllm-env/bin/vllm\"|vllm_bin: \"$VLLM_BIN\"|" "$BASE_CONFIG" > "$RUNTIME_CONFIG"

# Optional local model path
if [[ -n "${MODEL_PATH:-}${CODER_MODEL_PATH:-}" ]]; then
  "$PIPE_VENV/bin/python" "$TRAINING_ROOT/scripts/patch_pipeline_config.py" \
    --in "$RUNTIME_CONFIG" \
    --out "$RUNTIME_CONFIG" \
    --model "${MODEL_PATH:-$CODER_MODEL_PATH}"
fi

export CONFIG_FILE="$RUNTIME_CONFIG"
export CONFIG_PATH="$RUNTIME_CONFIG"

cd "$SHINY_GUIDE_ROOT/pipeline_service"
mkdir -p logs

echo "CONFIG_FILE=$CONFIG_FILE"
echo "SHINY_GUIDE=$SHINY_GUIDE_ROOT"
echo "VLLM_BIN=$VLLM_BIN"
echo "Wait for: curl -s http://127.0.0.1:10006/health"
echo "Logs: $PIPELINE_DIR/runs/pipeline-server.log"

exec bash run.sh 2>&1 | tee -a "$PIPELINE_DIR/runs/pipeline-server.log"
