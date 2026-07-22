#!/usr/bin/env bash
# Fetch / clone everything needed before training:
#   - shiny-guide (top agent pipeline)
#   - prompts.txt (validator image pool)
#   - Tooony133/Qwen-3.6-27B-AstroWolf (local coder model)
#
# Safe to re-run — skips steps when assets already exist.
#
# Env (see .env.template):
#   SHINY_GUIDE_REPO, PROMPTS_URL, CODER_MODEL_ID, MODEL_DIR, WORKSPACE_ROOT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$TRAINING_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$TRAINING_ROOT/.env"
  set +a
fi

export TRAINING_ROOT
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
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(detect_workspace)}"
export SHINY_GUIDE_ROOT="${SHINY_GUIDE_ROOT:-$WORKSPACE_ROOT/shiny-guide}"
export HF_HOME="${HF_HOME:-$TRAINING_ROOT/data/hf_cache}"
export MODEL_DIR="${MODEL_DIR:-$WORKSPACE_ROOT/models/Qwen-3.6-27B-AstroWolf}"

SHINY_GUIDE_REPO="${SHINY_GUIDE_REPO:-https://github.com/mokabetrade/shiny-guide.git}"
SHINY_GUIDE_REF="${SHINY_GUIDE_REF:-main}"
PROMPTS_URL="${PROMPTS_URL:-https://raw.githubusercontent.com/cameraMeasurementTech/image-three.js-localeval/main/prompts.txt}"
CODER_MODEL_ID="${CODER_MODEL_ID:-Tooony133/Qwen-3.6-27B-AstroWolf}"
PROMPTS_DEST="${PROMPTS_POOL:-$WORKSPACE_ROOT/prompts.txt}"

mkdir -p "$WORKSPACE_ROOT" "$WORKSPACE_ROOT/models" "$HF_HOME" "$TRAINING_ROOT/data"

echo "==> Bootstrap assets"
echo "    WORKSPACE_ROOT=$WORKSPACE_ROOT"
echo "    SHINY_GUIDE_ROOT=$SHINY_GUIDE_ROOT"
echo "    MODEL_DIR=$MODEL_DIR"
echo "    PROMPTS_DEST=$PROMPTS_DEST"

# --- 1. shiny-guide pipeline ---
if [[ -d "$SHINY_GUIDE_ROOT/pipeline_service" ]]; then
  echo "==> [skip] shiny-guide already at $SHINY_GUIDE_ROOT"
elif [[ -d "$WORKSPACE_ROOT/404-gen-subnet/shiny-guide/pipeline_service" ]]; then
  export SHINY_GUIDE_ROOT="$WORKSPACE_ROOT/404-gen-subnet/shiny-guide"
  echo "==> [link] using monorepo shiny-guide at $SHINY_GUIDE_ROOT"
else
  echo "==> [clone] shiny-guide from $SHINY_GUIDE_REPO (ref=$SHINY_GUIDE_REF)"
  if [[ -d "$SHINY_GUIDE_ROOT/.git" ]]; then
    git -C "$SHINY_GUIDE_ROOT" fetch --depth 1 origin "$SHINY_GUIDE_REF" || true
    git -C "$SHINY_GUIDE_ROOT" checkout "$SHINY_GUIDE_REF" || true
  else
    git clone --depth 1 --branch "$SHINY_GUIDE_REF" "$SHINY_GUIDE_REPO" "$SHINY_GUIDE_ROOT"
  fi
fi

if [[ ! -d "$SHINY_GUIDE_ROOT/pipeline_service" ]]; then
  echo "ERROR: shiny-guide pipeline_service missing at $SHINY_GUIDE_ROOT" >&2
  exit 1
fi

# --- 2. prompts.txt ---
if [[ -f "$PROMPTS_DEST" ]] && [[ $(wc -l < "$PROMPTS_DEST") -gt 1000 ]]; then
  echo "==> [skip] prompts pool $(wc -l < "$PROMPTS_DEST") lines at $PROMPTS_DEST"
elif [[ -f "$WORKSPACE_ROOT/prompts.txt" ]] && [[ $(wc -l < "$WORKSPACE_ROOT/prompts.txt") -gt 1000 ]]; then
  PROMPTS_DEST="$WORKSPACE_ROOT/prompts.txt"
  echo "==> [skip] using $PROMPTS_DEST"
elif [[ -f "$WORKSPACE_ROOT/404-gen-subnet/prompts.txt" ]]; then
  PROMPTS_DEST="$TRAINING_ROOT/data/prompts.txt"
  cp "$WORKSPACE_ROOT/404-gen-subnet/prompts.txt" "$PROMPTS_DEST"
  echo "==> [copy] prompts from monorepo -> $PROMPTS_DEST"
else
  echo "==> [download] prompts.txt"
  TMP="$TRAINING_ROOT/data/prompts.txt.tmp"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$PROMPTS_URL" -o "$TMP"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$PROMPTS_URL" -O "$TMP"
  else
    echo "ERROR: need curl or wget to fetch prompts.txt" >&2
    exit 1
  fi
  lines=$(wc -l < "$TMP")
  if [[ "$lines" -lt 1000 ]]; then
    echo "ERROR: downloaded prompts look too small ($lines lines). Check PROMPTS_URL or copy manually." >&2
    exit 1
  fi
  mv "$TMP" "$PROMPTS_DEST"
  echo "    saved $lines lines -> $PROMPTS_DEST"
fi
export PROMPTS_POOL="$PROMPTS_DEST"

# --- 3. coder model (AstroWolf) ---
if [[ -f "$MODEL_DIR/config.json" ]] || [[ -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "==> [skip] coder model already at $MODEL_DIR"
else
  echo "==> [download] $CODER_MODEL_ID -> $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    python3 -m pip install -U "huggingface_hub[cli]" --quiet
  fi
  HF_ARGS=(download "$CODER_MODEL_ID" --local-dir "$MODEL_DIR")
  if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_ARGS+=(--token "$HF_TOKEN")
  fi
  huggingface-cli "${HF_ARGS[@]}"
fi

export MODEL_PATH="$MODEL_DIR"
export CODER_MODEL_PATH="$MODEL_DIR"

# Persist paths into .env if missing
ENV_FILE="$TRAINING_ROOT/.env"
touch "$ENV_FILE"
upsert_env() {
  local key="$1" val="$2"
  if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
    return
  fi
  echo "${key}=${val}" >> "$ENV_FILE"
}
upsert_env WORKSPACE_ROOT "$WORKSPACE_ROOT"
upsert_env SHINY_GUIDE_ROOT "$SHINY_GUIDE_ROOT"
upsert_env PROMPTS_POOL "$PROMPTS_POOL"
upsert_env MODEL_PATH "$MODEL_PATH"
upsert_env HF_HOME "$HF_HOME"
upsert_env CODER_MODEL_ID "$CODER_MODEL_ID"

echo ""
echo "==> Bootstrap assets complete"
echo "    SHINY_GUIDE_ROOT=$SHINY_GUIDE_ROOT"
echo "    PROMPTS_POOL=$PROMPTS_POOL"
echo "    MODEL_PATH=$MODEL_PATH"
echo "    Next: ./run/00_install_all.sh  (or ./run/run_all.sh)"
