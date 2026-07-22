#!/usr/bin/env bash
# Install all packages and runtime assets for standalone training.
#
#   - system deps (optional, INSTALL_SYSTEM=1)
#   - bundled validate.js npm
#   - training Python venv + CUDA torch + requirements.txt
#   - shiny-guide npm sidecars + pipeline venv + vLLM
#
# Prereq: ./run/00_bootstrap_assets.sh (or monorepo with shiny-guide + prompts)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$TRAINING_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$TRAINING_ROOT/.env"
  set +a
fi

detect_ws() {
  if [[ -n "${WORKSPACE_ROOT:-}" && -d "${WORKSPACE_ROOT}/shiny-guide" ]]; then echo "$WORKSPACE_ROOT"; return; fi
  local p g; p="$(dirname "$TRAINING_ROOT")"; g="$(dirname "$p")"
  [[ -d "$p/shiny-guide" ]] && { echo "$p"; return; }
  [[ -d "$g/shiny-guide" ]] && { echo "$g"; return; }
  echo "$p"
}
WS="$(detect_ws)"
SG="${SHINY_GUIDE_ROOT:-$WS/shiny-guide}"

need_bootstrap=0
[[ -d "$SG/pipeline_service" ]] || need_bootstrap=1
[[ -f "${PROMPTS_POOL:-}" ]] || [[ -f "$WS/prompts.txt" ]] || [[ -f "$TRAINING_ROOT/data/prompts.txt" ]] || need_bootstrap=1

if [[ "$need_bootstrap" == "1" ]]; then
  "$SCRIPT_DIR/00_bootstrap_assets.sh"
fi

if [[ -f "$TRAINING_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$TRAINING_ROOT/.env"
  set +a
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
cd "$TRAINING_ROOT"

echo "==> Install all (workspace=$WORKSPACE_ROOT)"

if [[ "${INSTALL_SYSTEM:-0}" == "1" ]]; then
  echo "==> System packages (Chromium sidecars, build tools)"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y git curl wget build-essential python3 python3-venv python3-pip \
      nodejs npm \
      fonts-liberation libnss3 libatk-bridge2.0-0 libx11-xcb1 libxcomposite1 \
      libxdamage1 libxrandr2 libgbm1 libasound2 libpangocairo-1.0-0 libgtk-3-0 \
      libegl1 libgl1-mesa-dri libgles2 xvfb
  else
    echo "WARN: apt-get not found — install Node>=20 and Chromium libs manually"
  fi
fi

node_major=$(node -v 2>/dev/null | sed 's/^v//' | cut -d. -f1)
if [[ -z "$node_major" ]] || [[ "$node_major" -lt 20 ]]; then
  echo "ERROR: Node.js >= 20 required (got: $(node -v 2>/dev/null || echo missing))" >&2
  exit 1
fi

echo "==> Bundled validator"
VALIDATOR_NPM="$TRAINING_ROOT/third_party/miner-reference/validator"
(cd "$VALIDATOR_NPM" && npm install)

echo "==> Training venv"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel

CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "    torch+cuda already OK"
else
  echo "    Installing CUDA torch from $CUDA_INDEX"
  pip install torch torchvision --index-url "$CUDA_INDEX"
fi

pip install -r requirements.txt
pip install "huggingface_hub[cli]"

mkdir -p data/{splits,images,candidates,hf,checkpoints,logs}

echo "==> Pipeline + vLLM (shiny-guide native)"
"$PIPELINE_DIR/setup-native.sh"

echo ""
echo "==> Install complete"
echo "    source $TRAINING_ROOT/.venv/bin/activate"
echo "    source $TRAINING_ROOT/.env"
echo "    Full run: ./run/run_all.sh"
