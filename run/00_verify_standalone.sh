#!/usr/bin/env bash
# Verify training/ is self-contained enough to run after bootstrap.
# Usage: ./run/00_verify_standalone.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$TRAINING_ROOT"

fail=0
pass() { echo "  PASS  $*"; }
warn() { echo "  WARN  $*"; }
bad()  { echo "  FAIL  $*"; fail=1; }

echo "==> Standalone check: $TRAINING_ROOT"

# Bundled (must ship with the tree)
[[ -f third_party/miner-reference/tools/validate.js ]] && pass "bundled validate.js" || bad "missing third_party/.../validate.js"
[[ -f third_party/miner-reference/validator/package.json ]] && pass "bundled validator package.json" || bad "missing validator package.json"
[[ -f pipeline/configuration.h200x4-dpo-duel.yaml ]] && pass "h200x4-dpo-duel pipeline yaml" || bad "missing configuration.h200x4-dpo-duel.yaml"
[[ -f pipeline/configuration.duel-judge.yaml ]] && pass "duel-judge yaml" || bad "missing configuration.duel-judge.yaml"
[[ -f configs/dpo_shiny_27b_duel.yaml ]] && pass "dpo_shiny_27b_duel train config" || bad "missing train config"
[[ -x run/01_prepare_dpo_duel_scored.sh ]] || chmod +x run/01_prepare_dpo_duel_scored.sh
[[ -f run/01_prepare_sft_openrouter.sh ]] && pass "sft openrouter prep script" || bad "missing 01_prepare_sft_openrouter.sh"
[[ -f configs/sft_shiny_27b_gpt_teacher.yaml ]] && pass "sft gpt teacher config" || bad "missing sft_shiny_27b_gpt_teacher.yaml"
[[ -f docs/SFT_GPT_TEACHER.md ]] && pass "SFT_GPT_TEACHER.md" || warn "missing SFT_GPT_TEACHER.md"
[[ -f scripts/collect_candidates.py ]] && pass "collect_candidates.py" || bad "missing collect_candidates.py"
[[ -f scripts/duel_score_candidates.py ]] && pass "duel_score_candidates.py" || bad "missing duel_score_candidates.py"
[[ -f scripts/pack_dpo_dataset.py ]] && pass "pack_dpo_dataset.py" || bad "missing pack_dpo_dataset.py"
[[ -f scripts/paths.py ]] && pass "paths.py" || bad "missing paths.py"

# Must NOT require monorepo siblings for defaults (ignore comments / this script)
hits=$(grep -RIn --include='*.sh' 'local-eval/' run/ pipeline/ 2>/dev/null \
  | grep -v '00_verify_standalone.sh' \
  | grep -vE '^\S+:[0-9]+:\s*#' \
  | grep -v 'no local-eval' \
  | head -10 || true)
if [[ -n "$hits" ]]; then
  echo "$hits"
  bad "run/pipeline shells still hard-require local-eval"
else
  pass "no local-eval hard deps in run/ + pipeline/*.sh"
fi

# Bootstrap artifacts (created on machine)
if [[ -d vendor/shiny-guide/pipeline_service ]]; then
  pass "vendor/shiny-guide present"
else
  warn "vendor/shiny-guide missing — run ./run/00_bootstrap_assets.sh"
fi
# Bootstrap artifacts (created on machine)
if [[ -f data/prompts.txt ]]; then
  lines=$(wc -l < data/prompts.txt | tr -d ' ')
  if [[ "$lines" -gt 50000 ]]; then
    pass "data/prompts.txt present ($lines lines, ~99k pool)"
  elif [[ "$lines" -gt 1000 ]]; then
    warn "data/prompts.txt has only $lines lines (expected ~99k) — FORCE_PROMPTS_DOWNLOAD=1 ./run/00_bootstrap_assets.sh"
  else
    bad "data/prompts.txt too small ($lines lines)"
  fi
  # Format: one https URL per line (or stem\\turl)
  if head -5 data/prompts.txt | grep -qE 'https?://'; then
    pass "prompts.txt starts with http(s) URLs"
  else
    bad "prompts.txt does not look like URL list"
  fi
else
  warn "data/prompts.txt missing — run ./run/00_bootstrap_assets.sh"
fi
if [[ -f data/models/Qwen-3.6-27B-AstroWolf/config.json ]] || [[ -n "${MODEL_PATH:-}" && -f "${MODEL_PATH}/config.json" ]]; then
  pass "coder model weights present"
else
  warn "AstroWolf weights missing — run bootstrap (needs HF_TOKEN)"
fi

# .env hygiene
if [[ -f .env ]]; then
  if grep -qE '^CONFIG_FILE=/home/' .env 2>/dev/null; then
    bad ".env has absolute CONFIG_FILE — re-run ./run/00_configure_profile.sh"
  else
    pass ".env CONFIG_FILE looks portable (or unset)"
  fi
else
  warn ".env missing — cp .env.template .env && configure profile"
fi

# Python path helpers
export PYTHONPATH="$TRAINING_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<'PY' || bad "paths.py import failed"
from paths import training_root, validate_js_cli, vendor_shiny_guide_root, default_data_root
assert training_root().name == "training" or (training_root() / "run").is_dir()
assert validate_js_cli().is_file()
print("  PASS  paths.py resolves bundled validate.js")
print("  INFO  vendor expected at", vendor_shiny_guide_root())
print("  INFO  data root", default_data_root())
PY

if [[ -f scripts/check_prep_regressions.py ]]; then
  if python3 scripts/check_prep_regressions.py; then
    pass "prep regression checks"
  else
    bad "prep regression checks failed"
  fi
fi

echo ""
if [[ "$fail" -eq 0 ]]; then
  echo "==> Standalone gate: OK (bootstrap warnings above are expected on a fresh copy)"
  exit 0
fi
echo "==> Standalone gate: FAILED — fix items above before deploy"
exit 1
