#!/usr/bin/env bash
# Start miner API + vLLM on the host GPUs (no Docker).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PS="$ROOT/pipeline_service"
CODER_VENV="${CODER_VENV:-$ROOT/.venvs/vllm-env}"
GLM_VENV="${GLM_VENV:-$ROOT/.venvs/vllm-glm-env}"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ ! -x "$CODER_VENV/bin/python" ]]; then
  echo "Missing $CODER_VENV — run: bash $ROOT/scripts/native_setup.sh" >&2
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN unset — gated model downloads will fail" >&2
fi

export PATH="${HOME}/.local/bin:${CODER_VENV}/bin:${PATH}"
export CONFIG_PATH="${CONFIG_PATH:-$ROOT/configuration.yaml}"
export CONFIG_FILE="$CONFIG_PATH"
export GLM_VLLM_BIN="${GLM_VLLM_BIN:-$GLM_VENV/bin/vllm}"
export NODE_CWD="${NODE_CWD:-$PS}"
export PUPPETEER_CACHE_DIR="${PUPPETEER_CACHE_DIR:-$ROOT/.cache/puppeteer}"
# Cap BLAS/OpenMP threads so 2× vLLM + Chromium don't hit pthread_create EAGAIN.
# Pod cgroup pids.max=16384; without these caps warmup burst hits ~15k threads and
# GLM's EngineCore dies with `libgomp: Thread creation failed`.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
# glibc default is 8×NCPU malloc arenas → thousands of helper threads per Python
# process on 128-core hosts. Cap to 2 to reclaim ~10k threads at burst.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
# HF tokenizer spawns NCPU Rayon threads per call; disable in the miner
# processes (vLLM's server already tokenizes on its own thread pool).
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Torch inductor/dynamo compile threads (defaults to NCPU).
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
# CUDA lazy loading — avoid CUDA runtime spawning helper threads for every
# kernel at init time.
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
# /tmp is noexec in this Sysbox pod — Triton/Inductor .so mmap fails with
# "failed to map segment from shared object" unless caches live on /home.
export TMPDIR="${TMPDIR:-$ROOT/.cache/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.cache/torchinductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$ROOT/.cache/vllm}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$ROOT/.cache/flashinfer}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"

# vLLM RPC default is 10s; a cold first request JIT-compiles Triton kernels and
# blows past that, killing EngineCore with "RPC call to sample_tokens timed out".
# 300s covers cold-start + fallback paths.
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-300000}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-600}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-300}"
# flashinfer GDN prefill kernel fails to build for sm_90a here; force the
# non-flashinfer sampler so the fallback path isn't hit at request time.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

cd "$PS"
# shellcheck disable=SC1091
source "$CODER_VENV/bin/activate"
exec bash "$PS/run.sh"
