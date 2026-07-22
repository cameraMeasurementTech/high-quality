#!/usr/bin/env bash
# Prepare shiny-guide datasets (legacy wrapper — includes SFT pack by default).
#
# For DPO/GRPO-only (skip SFT training), prefer:
#   ALIGN=dpo|grpo|both ./run/01_prepare_shiny_align.sh
#
# Env: same as 01_prepare_shiny_align.sh plus TEACHER_MODE for SFT pack.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec env ALIGN="${ALIGN:-both}" PREP_SFT="${PREP_SFT:-1}" TEACHER_MODE="${TEACHER_MODE:-pipeline}" \
  "$ROOT/run/01_prepare_shiny_align.sh" "$@"
