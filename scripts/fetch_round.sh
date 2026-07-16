#!/usr/bin/env bash
# Fetch seed.json + prompts.txt for a competition round and build round.txt.
# Usage: bash scripts/fetch_round.sh [ROUND]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMP="${COMP:-https://raw.githubusercontent.com/404-Repo/404-active-competition/main}"
ROUND="${1:-}"

if [[ -z "$ROUND" ]]; then
  ROUND="$(curl -fsSL "$COMP/state.json" | python3 -c 'import sys,json; print(json.load(sys.stdin)["current_round"])')"
fi

OUT="$ROOT/rounds/$ROUND"
mkdir -p "$OUT"

echo "Fetching round $ROUND → $OUT"
curl -fsSL "$COMP/rounds/$ROUND/seed.json" -o "$OUT/seed.json"
curl -fsSL "$COMP/rounds/$ROUND/prompts.txt" -o "$OUT/prompts.txt"

"$ROOT/.venv/bin/python" - <<PY
from pathlib import Path
from urllib.parse import urlparse
import json

out = Path("$OUT")
seed = json.loads((out / "seed.json").read_text())
urls = [u.strip() for u in (out / "prompts.txt").read_text().splitlines() if u.strip()]
lines = [f"{Path(urlparse(u).path).stem}\t{u}" for u in urls]
(out / "round.txt").write_text("\n".join(lines) + "\n")
# tiny smoke subset (first 2) for pipeline checks
(out / "round.smoke.txt").write_text("\n".join(lines[:2]) + "\n")
print(f"seed={seed['seed']}")
print(f"prompts={len(urls)}")
print(f"wrote {out/'round.txt'} and {out/'round.smoke.txt'}")
PY
