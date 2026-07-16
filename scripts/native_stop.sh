#!/usr/bin/env bash
# Stop native miner stack (serve + vLLM + renderers + chrome leftovers)
set -euo pipefail
pkill -9 -f '/home/high-quality/pipeline_service' 2>/dev/null || true
pkill -9 -f 'python serve.py' 2>/dev/null || true
pkill -9 -f 'render_runner.mjs' 2>/dev/null || true
pkill -9 -f 'vllm serve' 2>/dev/null || true
pkill -9 -f 'EngineCore|Worker_TP' 2>/dev/null || true
# Orphan Chromium from failed parallel puppeteer launches
pkill -9 -f '/home/high-quality/.cache/puppeteer/chrome/.*/chrome' 2>/dev/null || true
pkill -9 -f 'puppeteer_dev_chrome_profile' 2>/dev/null || true
sleep 1
if pgrep -af 'serve.py|vllm serve|render_runner|puppeteer_dev_chrome' | grep -v grep >/dev/null; then
  echo "Still running:"
  pgrep -af 'serve.py|vllm serve|render_runner|puppeteer_dev_chrome' | grep -v grep || true
  exit 1
fi
echo "Stopped."
