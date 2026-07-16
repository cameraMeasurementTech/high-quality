#!/usr/bin/env bash
# Validate every {stem}.js with the subnet miner-reference validator.
# Usage: bash scripts/validate_js.sh runs/round24-smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JS_DIR="${1:?usage: validate_js.sh <dir-with-js>}"
LOCALEVAL="${LOCALEVAL:-/home/image-three.js-localeval}"
VAL="$LOCALEVAL/miner-reference/tools/validate.js"

# Prefer a real node >= 20; fall back to Cursor's bundled node.
resolve_node() {
  local c ver major
  for c in ${NODE_BIN:-} node; do
    [[ -n "$c" ]] || continue
    if [[ "$c" == node ]] && ! command -v node >/dev/null 2>&1; then continue; fi
    ver=$("$c" -v 2>/dev/null | sed 's/^v//'); major=${ver%%.*}
    [[ "$major" =~ ^[0-9]+$ ]] && [[ "$major" -ge 20 ]] && { echo "$c"; return 0; }
  done
  for c in /root/.cursor-server/bin/linux-x64/*/node; do
    [[ -x "$c" ]] || continue
    ver=$("$c" -v 2>/dev/null | sed 's/^v//'); major=${ver%%.*}
    [[ "$major" -ge 20 ]] && { echo "$c"; return 0; }
  done
  return 1
}

NODE="$(resolve_node)" || { echo "Need Node.js >= 20"; exit 1; }
[[ -f "$VAL" ]] || { echo "Missing validator: $VAL"; exit 1; }

# tools/validate.js is ESM but tools/ has no package.json "type":"module"
NODE_ESM=("$NODE" --experimental-default-type=module)

# Ensure validator deps exist
if [[ ! -d "$LOCALEVAL/miner-reference/validator/node_modules/three" ]]; then
  echo "Installing miner-reference/validator deps…"
  (cd "$LOCALEVAL/miner-reference/validator" && npm ci --no-audit --no-fund)
fi

shopt -s nullglob
files=("$JS_DIR"/*.js)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No .js files in $JS_DIR"
  exit 1
fi

fail=0
ok=0
for f in "${files[@]}"; do
  if "${NODE_ESM[@]}" "$VAL" "$f" >/tmp/validate.out 2>&1; then
    echo "OK  $(basename "$f")"
    ok=$((ok + 1))
  else
    echo "FAIL $(basename "$f")"
    sed -n '1,20p' /tmp/validate.out
    fail=$((fail + 1))
  fi
done

echo "validated ok=$ok fail=$fail total=${#files[@]}"
[[ "$fail" -eq 0 ]]
