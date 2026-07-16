#!/usr/bin/env bash
# Local pipeline smoke: fetch round → generate .js → validate → (optional) R2 upload.
#
# Prereqs:
#   1. Miner API already running on :10006 (Docker or bare-metal ./pipeline_service/run.sh)
#   2. .venv created (uv venv .venv && uv pip install boto3 httpx)
#   3. For upload: R2_* + CDN_PUBLIC_BASE in env or .env
#
# Usage:
#   bash scripts/smoke_round.sh 24                 # full round (128 prompts)
#   bash scripts/smoke_round.sh 24 --smoke         # first 2 prompts only
#   bash scripts/smoke_round.sh 24 --smoke --upload
#   bash scripts/smoke_round.sh 24 --limit 5 --upload
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Load optional secrets
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

ROUND=""
SMOKE=false
UPLOAD=false
LIMIT=""
HOST="${HOST:-localhost}"
PORT="${PORT:-10006}"
SKIP_GENERATE=false
SKIP_VALIDATE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=true; shift ;;
    --upload) UPLOAD=true; shift ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --skip-generate) SKIP_GENERATE=true; shift ;;
    --skip-validate) SKIP_VALIDATE=true; shift ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      if [[ -z "$ROUND" && "$1" =~ ^[0-9]+$ ]]; then
        ROUND="$1"; shift
      else
        echo "Unknown arg: $1"; exit 2
      fi
      ;;
  esac
done

ROUND="${ROUND:-24}"
PY="${ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Missing $PY — run: uv venv .venv && uv pip install boto3 httpx"; exit 1; }

echo "============================================================"
echo " smoke_round  round=$ROUND  smoke=$SMOKE  upload=$UPLOAD"
echo "============================================================"

# --- 1. fetch inputs ---------------------------------------------------------
bash "$ROOT/scripts/fetch_round.sh" "$ROUND"
SEED="$("$PY" -c "import json; print(json.load(open('rounds/$ROUND/seed.json'))['seed'])")"
if [[ "$SMOKE" == true ]]; then
  PROMPTS="rounds/$ROUND/round.smoke.txt"
  NAME="round${ROUND}-smoke"
elif [[ -n "$LIMIT" ]]; then
  PROMPTS="rounds/$ROUND/round.txt"
  NAME="round${ROUND}-limit${LIMIT}"
else
  PROMPTS="rounds/$ROUND/round.txt"
  NAME="round${ROUND}"
fi
OUT_DIR="$ROOT/runs"
JS_DIR="$OUT_DIR/$NAME"

echo "seed=$SEED"
echo "prompts_file=$PROMPTS"
echo "out=$JS_DIR"

# --- 2. generate -------------------------------------------------------------
if [[ "$SKIP_GENERATE" != true ]]; then
  echo ""
  echo ">>> generate via http://${HOST}:${PORT}"
  GEN_ARGS=(
    "$PROMPTS"
    --host "$HOST" --port "$PORT"
    --seed "$SEED"
    --name "$NAME"
    --out-dir "$OUT_DIR"
    --timeout 14400
  )
  [[ -n "$LIMIT" && "$SMOKE" != true ]] && GEN_ARGS+=(--limit "$LIMIT")
  "$PY" "$ROOT/tests/test_pipeline.py" "${GEN_ARGS[@]}"
else
  echo ">>> skip generate (using existing $JS_DIR)"
fi

# Collect upload-only dir
UPLOAD_DIR="$JS_DIR/upload"
mkdir -p "$UPLOAD_DIR"
shopt -s nullglob
js_files=("$JS_DIR"/*.js)
if [[ ${#js_files[@]} -eq 0 ]]; then
  echo "ERROR: no .js produced in $JS_DIR"
  exit 1
fi
rm -f "$UPLOAD_DIR"/*.js
cp "$JS_DIR"/*.js "$UPLOAD_DIR"/
echo "upload set: ${#js_files[@]} files → $UPLOAD_DIR"

# --- 3. validate -------------------------------------------------------------
if [[ "$SKIP_VALIDATE" != true ]]; then
  echo ""
  echo ">>> validate"
  bash "$ROOT/scripts/validate_js.sh" "$UPLOAD_DIR"
fi

# --- 4. optional R2 upload ---------------------------------------------------
PREFIX="round-${ROUND}$([[ "$SMOKE" == true ]] && echo -smoke || true)"
if [[ "$UPLOAD" == true ]]; then
  echo ""
  echo ">>> upload R2 prefix=$PREFIX"
  "$PY" "$ROOT/scripts/upload_r2.py" "$UPLOAD_DIR" --prefix "$PREFIX"
else
  echo ""
  echo ">>> skip upload (pass --upload). Dry-run preview:"
  "$PY" "$ROOT/scripts/upload_r2.py" "$UPLOAD_DIR" --prefix "$PREFIX" --dry-run || true
fi

echo ""
echo "DONE"
echo "  js:      $UPLOAD_DIR"
echo "  cdn_url: \${CDN_PUBLIC_BASE}/${PREFIX}"
echo "============================================================"
