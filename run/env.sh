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

# Resolve a path relative to TRAINING_ROOT (does not require the file to exist).
_abs_under_training() {
  local p="${1:-}"
  [[ -z "$p" ]] && return 1
  if [[ "$p" != /* ]]; then
    p="$TRAINING_ROOT/${p#./}"
  fi
  # Normalize .. components without requiring the file.
  local dir base
  dir="$(dirname "$p")"
  base="$(basename "$p")"
  if [[ -d "$dir" ]]; then
    echo "$(cd "$dir" && pwd)/$base"
  else
    echo "$p"
  fi
}

# Find prompts.txt for standalone layout. Soft: may leave PROMPTS_POOL empty.
resolve_prompts_pool() {
  local candidates=()
  local c

  if [[ -n "${PROMPTS_POOL:-}" ]]; then
    candidates+=("$(_abs_under_training "$PROMPTS_POOL")")
  fi
  candidates+=(
    "$TRAINING_ROOT/data/prompts.txt"
    "$WORKSPACE_ROOT/prompts.txt"
    "$TRAINING_ROOT/prompts.txt"
  )

  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  echo ""
  return 0
}

export PROMPTS_POOL="$(resolve_prompts_pool)"

# Hard check for dataset-prep scripts (after bootstrap).
require_prompts_pool() {
  if [[ -z "${PROMPTS_POOL:-}" ]] || [[ ! -f "$PROMPTS_POOL" ]]; then
    echo "ERROR: prompts.txt not found." >&2
    echo "  Run: ./run/00_bootstrap_assets.sh" >&2
    echo "  expected: $TRAINING_ROOT/data/prompts.txt (~99k image URLs)" >&2
    exit 1
  fi
  local lines
  lines="$(wc -l < "$PROMPTS_POOL" | tr -d ' ')"
  if [[ "$lines" -lt 1000 ]]; then
    echo "ERROR: prompts pool too small ($lines lines) at $PROMPTS_POOL" >&2
    echo "  Re-download: FORCE_PROMPTS_DOWNLOAD=1 ./run/00_bootstrap_assets.sh" >&2
    exit 1
  fi
  if [[ "$lines" -lt 50000 ]]; then
    echo "WARNING: prompts pool has $lines lines (expected ~99k). Continuing." >&2
  fi
  export PROMPTS_POOL
  echo "==> prompts pool: $PROMPTS_POOL ($lines lines)"
}

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
