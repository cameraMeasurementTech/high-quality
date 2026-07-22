#!/usr/bin/env bash
# Shared environment for all training/run/*.sh scripts.
# Works in standalone (workspace/shiny-guide + workspace/training) and monorepo layouts.
set -euo pipefail

TRAINING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$TRAINING_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$TRAINING_ROOT/.env"
  set +a
fi

detect_workspace() {
  if [[ -n "${WORKSPACE_ROOT:-}" && -d "${WORKSPACE_ROOT}/shiny-guide" ]]; then
    echo "$WORKSPACE_ROOT"
    return
  fi
  local parent grand
  parent="$(dirname "$TRAINING_ROOT")"
  if [[ -d "$parent/shiny-guide" ]]; then
    echo "$parent"
    return
  fi
  grand="$(dirname "$parent")"
  if [[ -d "$grand/shiny-guide" ]]; then
    echo "$grand"
    return
  fi
  echo "$parent"
}

export TRAINING_ROOT
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(detect_workspace)}"
export SHINY_GUIDE_ROOT="${SHINY_GUIDE_ROOT:-$WORKSPACE_ROOT/shiny-guide}"
export PIPELINE_DIR="$TRAINING_ROOT/pipeline"
export PROMPTS_ROOT="${PROMPTS_ROOT:-shiny-guide}"
export PYTHONPATH="$TRAINING_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$TRAINING_ROOT/data/hf_cache}"

if [[ -z "${PROMPTS_POOL:-}" ]]; then
  if [[ -f "$WORKSPACE_ROOT/prompts.txt" ]]; then
    export PROMPTS_POOL="$WORKSPACE_ROOT/prompts.txt"
  elif [[ -f "$TRAINING_ROOT/data/prompts.txt" ]]; then
    export PROMPTS_POOL="$TRAINING_ROOT/data/prompts.txt"
  fi
fi

resolve_prompts_pool() {
  if [[ -n "${PROMPTS_POOL:-}" ]]; then
    echo "$PROMPTS_POOL"
    return
  fi
  echo "ERROR: prompts.txt not found. Copy to $WORKSPACE_ROOT/prompts.txt or" >&2
  echo "  $TRAINING_ROOT/data/prompts.txt and set PROMPTS_POOL in .env" >&2
  exit 1
}

export PROMPTS_POOL="$(resolve_prompts_pool)"

# Legacy alias used by some docs
export REPO="$WORKSPACE_ROOT"
export ROOT="$TRAINING_ROOT"
