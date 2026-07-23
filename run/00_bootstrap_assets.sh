#!/usr/bin/env bash
# Fetch everything needed for standalone training (no monorepo required).
#
# Creates under training/:
#   vendor/shiny-guide/          king pipeline (git clone)
#   vendor/pipeline_prompts/       coder prompt snapshot for dataset packing
#   data/prompts.txt               validator image pool
#   data/models/Qwen-3.6-27B-AstroWolf/
#
# Safe to re-run — skips steps when assets already exist.
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
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$TRAINING_ROOT}"
export SHINY_GUIDE_ROOT="${SHINY_GUIDE_ROOT:-$TRAINING_ROOT/vendor/shiny-guide}"
export HF_HOME="${HF_HOME:-$TRAINING_ROOT/data/hf_cache}"
export MODEL_DIR="${MODEL_DIR:-$TRAINING_ROOT/data/models/Qwen-3.6-27B-AstroWolf}"

# Standalone default: always training/data/prompts.txt unless PROMPTS_POOL is an
# absolute path the user intentionally set.
DEFAULT_PROMPTS="$TRAINING_ROOT/data/prompts.txt"
if [[ -n "${PROMPTS_POOL:-}" ]]; then
  if [[ "${PROMPTS_POOL}" != /* ]]; then
    PROMPTS_POOL="$TRAINING_ROOT/${PROMPTS_POOL#./}"
  fi
  PROMPTS_DEST="$PROMPTS_POOL"
else
  PROMPTS_DEST="$DEFAULT_PROMPTS"
fi
export PROMPTS_DEST

SHINY_GUIDE_REPO="${SHINY_GUIDE_REPO:-https://github.com/mokabetrade/shiny-guide.git}"
SHINY_GUIDE_REF="${SHINY_GUIDE_REF:-main}"
PROMPTS_URL="${PROMPTS_URL:-https://raw.githubusercontent.com/cameraMeasurementTech/image-three.js-localeval/main/prompts.txt}"
CODER_MODEL_ID="${CODER_MODEL_ID:-Tooony133/Qwen-3.6-27B-AstroWolf}"

mkdir -p "$TRAINING_ROOT/vendor" "$TRAINING_ROOT/data/models" "$(dirname "$PROMPTS_DEST")" "$HF_HOME"

echo "==> Bootstrap assets (standalone training/)"
echo "    TRAINING_ROOT=$TRAINING_ROOT"
echo "    SHINY_GUIDE_ROOT=$SHINY_GUIDE_ROOT"
echo "    MODEL_DIR=$MODEL_DIR"
echo "    PROMPTS_DEST=$PROMPTS_DEST"
echo "    PROMPTS_URL=$PROMPTS_URL"

# --- 1. shiny-guide pipeline (vendored) ---
if [[ -d "$SHINY_GUIDE_ROOT/pipeline_service" ]]; then
  echo "==> [skip] shiny-guide at $SHINY_GUIDE_ROOT"
else
  echo "==> [clone] shiny-guide -> $SHINY_GUIDE_ROOT"
  mkdir -p "$(dirname "$SHINY_GUIDE_ROOT")"
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

echo "==> [patch] throughput overlays (skip_render + prepare concurrency)"
python3 "$TRAINING_ROOT/pipeline/apply_throughput_overlays.py" "$SHINY_GUIDE_ROOT" || true

# --- 1b. vendored coder prompts (dataset packing without monorepo paths) ---
VENDOR_PROMPTS="$TRAINING_ROOT/vendor/pipeline_prompts/scene_coder"
mkdir -p "$VENDOR_PROMPTS"
SCENE_CODER="$SHINY_GUIDE_ROOT/pipeline_service/modules/scene_coder"
for f in prompts.py few_shot_examples.py threejs_reference.py; do
  if [[ -f "$SCENE_CODER/$f" ]]; then
    cp "$SCENE_CODER/$f" "$VENDOR_PROMPTS/$f"
  fi
done
echo "==> [sync] coder prompts -> $VENDOR_PROMPTS"

# --- 2. prompts.txt (~99k image URLs, one URL per line) ---
# Used by prepare_splits.py → data/splits/{train,val,duel}.txt → download_images.py
FORCE_PROMPTS="${FORCE_PROMPTS_DOWNLOAD:-0}"
prompts_ok=0
if [[ -f "$PROMPTS_DEST" ]]; then
  prompts_lines="$(wc -l < "$PROMPTS_DEST" | tr -d ' ')"
  if [[ "$prompts_lines" -gt 1000 ]]; then
    prompts_ok=1
  fi
fi

if [[ "$FORCE_PROMPTS" == "1" ]] || [[ "$prompts_ok" != "1" ]]; then
  echo "==> [download] prompts.txt from $PROMPTS_URL"
  TMP="$PROMPTS_DEST.tmp"
  mkdir -p "$(dirname "$PROMPTS_DEST")"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 --retry-delay 2 "$PROMPTS_URL" -o "$TMP"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$PROMPTS_URL" -O "$TMP"
  else
    echo "ERROR: need curl or wget" >&2
    exit 1
  fi
  lines="$(wc -l < "$TMP" | tr -d ' ')"
  if [[ "$lines" -lt 1000 ]]; then
    echo "ERROR: downloaded prompts too small ($lines lines) from $PROMPTS_URL" >&2
    rm -f "$TMP"
    exit 1
  fi
  # Sanity: first non-empty line should be an http(s) URL
  first="$(grep -m1 -E 'https?://' "$TMP" || true)"
  if [[ -z "$first" ]]; then
    echo "ERROR: downloaded prompts.txt has no http URLs" >&2
    rm -f "$TMP"
    exit 1
  fi
  mv "$TMP" "$PROMPTS_DEST"
  echo "    saved $lines lines -> $PROMPTS_DEST"
  if [[ "$lines" -lt 50000 ]]; then
    echo "    WARNING: expected ~99k lines; got $lines — check PROMPTS_URL" >&2
  fi
else
  echo "==> [skip] prompts pool $prompts_lines lines at $PROMPTS_DEST"
fi
export PROMPTS_POOL="$PROMPTS_DEST"

# --- 3. coder model (AstroWolf) ---
if [[ -f "$MODEL_DIR/config.json" ]] || [[ -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "==> [skip] coder model at $MODEL_DIR"
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

ENV_FILE="$TRAINING_ROOT/.env"
touch "$ENV_FILE"
upsert_env() {
  local key="$1" val="$2"
  if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
    return
  fi
  echo "${key}=${val}" >> "$ENV_FILE"
}
# Always refresh paths that must stay absolute under this training/ tree.
upsert_env_force() {
  local key="$1" val="$2"
  if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
    # portable in-place replace
    local tmp
    tmp="$(mktemp)"
    awk -v k="$key" -v v="$val" 'BEGIN{FS=OFS="="} $1==k{$0=k"="v} {print}' "$ENV_FILE" > "$tmp"
    mv "$tmp" "$ENV_FILE"
  else
    echo "${key}=${val}" >> "$ENV_FILE"
  fi
}
upsert_env WORKSPACE_ROOT "$WORKSPACE_ROOT"
upsert_env TRAINING_ROOT "$TRAINING_ROOT"
upsert_env_force SHINY_GUIDE_ROOT "$SHINY_GUIDE_ROOT"
upsert_env_force PROMPTS_POOL "$PROMPTS_POOL"
upsert_env_force MODEL_PATH "$MODEL_PATH"
upsert_env HF_HOME "$HF_HOME"
upsert_env CODER_MODEL_ID "$CODER_MODEL_ID"

echo ""
echo "==> Bootstrap complete (standalone — no external repo required)"
echo "    vendor/shiny-guide  -> pipeline + vLLM"
echo "    data/prompts.txt    -> ~99k URLs for train/val/duel splits"
echo "    data/models/        -> AstroWolf weights"
echo "    Next: ./run/00_install_all.sh"
echo "    Then: prepare splits use PROMPTS_POOL=$PROMPTS_POOL"
