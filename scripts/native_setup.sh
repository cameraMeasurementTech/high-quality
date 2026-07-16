#!/usr/bin/env bash
# One-time bare-metal setup (no Docker). Builds coder + GLM vLLM envs and Node deps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PS="$ROOT/pipeline_service"
VENVS="$ROOT/.venvs"
CODER_VENV="$VENVS/vllm-env"
GLM_VENV="$VENVS/vllm-glm-env"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

export PATH="${HOME}/.local/bin:${PATH}"
command -v uv >/dev/null || { echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

echo "=== [1/4] system packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  ca-certificates curl wget build-essential \
  fonts-liberation fonts-noto-color-emoji \
  libnss3 libatk-bridge2.0-0 libx11-xcb1 libxcomposite1 \
  libxdamage1 libxrandr2 libgbm1 libasound2t64 \
  libpangocairo-1.0-0 libgtk-3-0 libegl1 libgl1 libgles2 \
  xvfb speedtest-cli 2>/dev/null \
  || apt-get install -y -qq \
  ca-certificates curl wget build-essential \
  fonts-liberation libnss3 libgbm1 libasound2t64 \
  libgtk-3-0 libegl1 libgl1 xvfb || true

if ! command -v npm >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y -qq nodejs
fi

echo "=== [2/4] coder vLLM env → $CODER_VENV ==="
mkdir -p "$VENVS"
uv python install "$PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION" "$CODER_VENV"
uv pip install --python "$CODER_VENV/bin/python" --upgrade pip
# PyTorch index also publishes a `vllm` stub; without unsafe-best-match uv
# stops at that index and never sees vllm==0.17.1 on PyPI.
uv pip install --python "$CODER_VENV/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r "$ROOT/docker/requirements.vllm.txt"
uv pip install --python "$CODER_VENV/bin/python" \
  -r "$ROOT/docker/requirements.txt" \
  requests numpy pillow transformers accelerate huggingface_hub PyYAML

echo "=== [3/4] GLM vLLM env → $GLM_VENV ==="
# GLM needs vllm 0.23+cu129 + matching torch 2.11+cu129 (NOT the coder cu128
# stack). Driver 570 runs cu129 via CUDA 12.x minor-version compatibility.
# Never overwrite torch with cu128 — that causes: undefined symbol CUDAStream::query
if ! VENV="$GLM_VENV" TORCH_BACKEND=cu129 \
    bash "$PS/scripts/setup_glm_vllm_env.sh"; then
  echo "[native_setup] GLM script failed (often pytorch.org 403) — fallback install"
  rm -rf "$GLM_VENV"
  python3.11 -m venv "$GLM_VENV"
  "$GLM_VENV/bin/pip" install --upgrade pip uv
  WHEEL_URL="${WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl}"
  "$GLM_VENV/bin/uv" pip install --python "$GLM_VENV/bin/python" \
    --index-strategy unsafe-best-match \
    --torch-backend cu129 \
    "vllm @ ${WHEEL_URL}" "transformers>=5.0.0rc0" "fastapi<0.137"
fi
test -x "$GLM_VENV/bin/vllm" || { echo "GLM vllm binary missing"; exit 1; }
"$GLM_VENV/bin/python" -c 'import vllm, torch; torch.zeros(1).cuda(); print(f"GLM ok — vllm {vllm.__version__} torch {torch.__version__}")'

echo "=== [4/4] Node deps + Chrome + config paths ==="
cp -f "$ROOT/docker/package.json" "$PS/package.json"
if [[ -f "$ROOT/docker/package-lock.json" ]]; then
  cp -f "$ROOT/docker/package-lock.json" "$PS/package-lock.json"
  (cd "$PS" && npm ci --no-audit --no-fund)
else
  (cd "$PS" && npm install --no-audit --no-fund)
fi

# Puppeteer Chrome (render sidecars) — required or serve.py dies at startup
export PUPPETEER_CACHE_DIR="${PUPPETEER_CACHE_DIR:-$ROOT/.cache/puppeteer}"
mkdir -p "$PUPPETEER_CACHE_DIR"
if [[ ! -x "$(find "$PUPPETEER_CACHE_DIR" -type f -name chrome -path '*/chrome-linux64/chrome' 2>/dev/null | head -1)" ]]; then
  echo "[native_setup] installing Puppeteer Chrome into $PUPPETEER_CACHE_DIR"
  (cd "$PS" && PUPPETEER_CACHE_DIR="$PUPPETEER_CACHE_DIR" npx --yes puppeteer browsers install chrome)
fi

# Point configuration.yaml at local venvs (absolute paths — /opt symlinks optional)
mkdir -p /opt 2>/dev/null || true
ln -sfn "$CODER_VENV" /opt/vllm-env 2>/dev/null || true
ln -sfn "$GLM_VENV" /opt/vllm-glm-env 2>/dev/null || true

python3 - <<PY
from pathlib import Path
import re
cfg = Path("$ROOT/configuration.yaml")
text = cfg.read_text()
coder = "$CODER_VENV/bin/vllm"
glm = "$GLM_VENV/bin/vllm"
text = text.replace("/opt/vllm-glm-env/bin/vllm", glm)
text = text.replace("/opt/vllm-env/bin/vllm", coder)
if "coder-instance:" in text and 'vllm_bin:' not in text.split("judge-critic-instance:")[0]:
    text = text.replace('gpu_ids: "0,1"', f'gpu_ids: "0,1"\n      vllm_bin: "{coder}"', 1)
else:
    # refresh coder path if present
    text = re.sub(
        r'(coder-instance:.*?vllm:.*?)(vllm_bin:\s*")[^"]*(")',
        rf'\1\2{coder}\3',
        text,
        count=1,
        flags=re.S,
    )
cfg.write_text(text)
print(f"coder -> {coder}")
print(f"glm   -> {glm}")
PY

echo ""
echo "Setup done. Start with:  bash $ROOT/scripts/native_run.sh"
