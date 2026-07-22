#!/usr/bin/env bash
# Shared environment for all training/run/*.sh scripts.
# Standalone: only the training/ directory is required after bootstrap.
set -euo pipefail

TRAINING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TRAINING_ROOT

if [[ -f "$TRAINING_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$TRAINING_ROOT/.env"
  set +a
fi

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$TRAINING_ROOT}"
export SHINY_GUIDE_ROOT="${SHINY_GUIDE_ROOT:-$TRAINING_ROOT/vendor/shiny-guide}"
export PIPELINE_DIR="$TRAINING_ROOT/pipeline"
export PROMPTS_ROOT="${PROMPTS_ROOT:-shiny-guide}"
export PYTHONPATH="$TRAINING_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$TRAINING_ROOT/data/hf_cache}"
export MODEL_PATH="${MODEL_PATH:-${CODER_MODEL_PATH:-$TRAINING_ROOT/data/models/Qwen-3.6-27B-AstroWolf}}"

if [[ -z "${PROMPTS_POOL:-}" ]]; then
  if [[ -f "$TRAINING_ROOT/data/prompts.txt" ]]; then
    export PROMPTS_POOL="$TRAINING_ROOT/data/prompts.txt"
  elif [[ -f "$WORKSPACE_ROOT/prompts.txt" ]]; then
    export PROMPTS_POOL="$WORKSPACE_ROOT/prompts.txt"
  fi
fi

resolve_prompts_pool() {
  if [[ -n "${PROMPTS_POOL:-}" ]]; then
    echo "$PROMPTS_POOL"
    return
  fi
  echo "ERROR: prompts.txt not found. Run ./run/00_bootstrap_assets.sh" >&2
  echo "  expected: $TRAINING_ROOT/data/prompts.txt" >&2
  exit 1
}

export PROMPTS_POOL="$(resolve_prompts_pool)"

# Legacy aliases (some docs still mention REPO)
export REPO="$WORKSPACE_ROOT"
export ROOT="$TRAINING_ROOT"

require_shiny_guide() {
  if [[ ! -d "$SHINY_GUIDE_ROOT/pipeline_service" ]]; then
    echo "ERROR: shiny-guide missing at $SHINY_GUIDE_ROOT" >&2
    echo "Run: ./run/00_bootstrap_assets.sh" >&2
    exit 1
  fi
}

pipeline_config_file() {
  echo "${CONFIG_FILE:-$PIPELINE_DIR/configuration.gpu-native.yaml}"
}

require_openrouter_if_needed() {
  local cfg
  cfg="$(pipeline_config_file)"
  if [[ ! -f "$cfg" ]]; then
    return 0
  fi
  if [[ -x "$TRAINING_ROOT/.venv/bin/python" ]]; then
    PY="$TRAINING_ROOT/.venv/bin/python"
  else
    PY=python3
  fi
  if "$PY" "$TRAINING_ROOT/scripts/pipeline_config.py" --config "$cfg" --needs-openrouter; then
    : "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in $TRAINING_ROOT/.env (required by $cfg)}"
  fi
}
