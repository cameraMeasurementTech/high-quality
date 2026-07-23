#!/usr/bin/env bash
# One-time setup: shiny-guide native GPU pipeline + bundled validator + training venv.
# Standalone — uses training/vendor/shiny-guide (from 00_bootstrap_assets.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$TRAINING_ROOT/run/env.sh"

echo "==> Standalone workspace"
echo "    WORKSPACE_ROOT=$WORKSPACE_ROOT"
echo "    SHINY_GUIDE_ROOT=$SHINY_GUIDE_ROOT"
echo "    TRAINING_ROOT=$TRAINING_ROOT"

if [[ ! -d "$SHINY_GUIDE_ROOT/pipeline_service" ]]; then
  echo "ERROR: shiny-guide not found at $SHINY_GUIDE_ROOT" >&2
  exit 1
fi

nvidia-smi || { echo "WARN: nvidia-smi failed — vLLM needs GPU" >&2; }

echo "==> Node >= 20"
node --version

echo "==> Bundled validator (DPO/GRPO rewards)"
VALIDATOR_NPM="$TRAINING_ROOT/third_party/miner-reference/validator"
(cd "$VALIDATOR_NPM" && npm install)

echo "==> shiny-guide docker/ npm deps (puppeteer, three, babel)"
cd "$SHINY_GUIDE_ROOT/docker"
if [[ -f package-lock.json ]]; then npm ci; else npm install; fi

RENDER_SVC="$SHINY_GUIDE_ROOT/pipeline_service/modules/renderer/render_service"
JS_CHECK="$SHINY_GUIDE_ROOT/pipeline_service/modules/js_checker"
ln -sfn "$SHINY_GUIDE_ROOT/docker/node_modules" "$RENDER_SVC/node_modules"
ln -sfn "$SHINY_GUIDE_ROOT/docker/node_modules" "$JS_CHECK/node_modules"

echo "==> Pipeline Python venv ($PIPELINE_DIR/.venv)"
PIPE_VENV="$PIPELINE_DIR/.venv"
python3 -m venv "$PIPE_VENV"
# shellcheck disable=SC1091
source "$PIPE_VENV/bin/activate"
pip install -U pip wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r "$SHINY_GUIDE_ROOT/docker/requirements.txt"
pip install -r "$PIPELINE_DIR/requirements.txt"
pip install transformers pillow accelerate httpx openai loguru
deactivate

echo "==> vLLM venv ($PIPELINE_DIR/.vllm-env)"
VLLM_VENV="$PIPELINE_DIR/.vllm-env"
if [[ ! -x "$VLLM_VENV/bin/vllm" ]]; then
  python3 -m venv "$VLLM_VENV"
  # shellcheck disable=SC1091
  source "$VLLM_VENV/bin/activate"
  pip install -U pip
  pip install vllm==0.17.1 --extra-index-url https://download.pytorch.org/whl/cu128 || \
    pip install "vllm>=0.6.3"
  deactivate
fi

echo "==> Training Python venv ($TRAINING_ROOT/.venv) if missing"
if [[ ! -x "$TRAINING_ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$TRAINING_ROOT/.venv"
  # shellcheck disable=SC1091
  source "$TRAINING_ROOT/.venv/bin/activate"
  pip install -U pip wheel
  echo "    Install CUDA torch, then: pip install -r requirements.txt"
  deactivate
fi

mkdir -p "$PIPELINE_DIR/runs" "$SHINY_GUIDE_ROOT/pipeline_service/logs"
mkdir -p "$TRAINING_ROOT/data"/{splits,images,candidates,hf,checkpoints,logs}

cat <<EOF

Setup complete.

1. cp $TRAINING_ROOT/.env.template $TRAINING_ROOT/.env
   # set OPENROUTER_API_KEY, HF_TOKEN, optional MODEL_PATH

2. Copy prompts.txt to:
     $WORKSPACE_ROOT/prompts.txt
   or $TRAINING_ROOT/data/prompts.txt

3. Start pipeline (Terminal A):
     source $TRAINING_ROOT/.env
     $PIPELINE_DIR/run-native.sh

4. Prepare + train (Terminal B):
     cd $TRAINING_ROOT && source .venv/bin/activate
     ALIGN=dpo ./run/01_prepare_shiny_align.sh
     CONFIG=configs/dpo_shiny_27b.yaml ./run/03_dpo.sh

See STANDALONE.md for the full beat-the-king loop.

EOF
