#!/usr/bin/env bash
# Step 0 — training venv only (use 00_install_all.sh for full standalone setup).
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"
cd "$TRAINING_ROOT"

echo "==> Quick setup (for full install use ./run/00_install_all.sh or ./run/run_all.sh)"
echo "    Workspace: $WORKSPACE_ROOT"

VALIDATOR_NPM="$TRAINING_ROOT/third_party/miner-reference/validator"
if [[ ! -d "$VALIDATOR_NPM/node_modules" ]]; then
  (cd "$VALIDATOR_NPM" && npm install)
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel
pip install torch torchvision --index-url "${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}" || true
pip install -r requirements.txt || true
mkdir -p data/{splits,images,candidates,hf,checkpoints,logs}

echo "==> Done. Full bootstrap: ./run/run_all.sh"
