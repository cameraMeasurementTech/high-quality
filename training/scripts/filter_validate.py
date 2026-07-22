#!/usr/bin/env python3
"""Filter teacher JS with miner-reference validate.js; keep only passed files.

Usage:
  python filter_validate.py \\
    --js-dir ../data/raw_js/shiny-guide \\
    --out-dir ../data/filtered_js \\
    --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from paths import default_data_root, validate_js_cli


def validate_one(js_path: Path, node_bin: str, validate_cli: Path) -> dict:
    try:
        proc = subprocess.run(
            [node_bin, str(validate_cli), "--json", str(js_path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"stem": js_path.stem, "passed": False, "error": "timeout"}
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {
            "stem": js_path.stem,
            "passed": False,
            "error": (proc.stderr or proc.stdout or "no json")[:500],
        }
    return {
        "stem": js_path.stem,
        "passed": bool(payload.get("passed")),
        "failures": payload.get("failures"),
        "metrics": payload.get("metrics"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Keep only validator-passing JS")
    ap.add_argument("--js-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=default_data_root() / "filtered_js")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--node", type=str, default="node")
    ap.add_argument("--copy-fail", action="store_true", help="also copy fails into out-dir/_failed")
    args = ap.parse_args()

    cli = validate_js_cli()
    if not cli.is_file():
        raise SystemExit(f"validate.js not found: {cli}")

    files = sorted(args.js_dir.glob("*.js"))
    if not files:
        raise SystemExit(f"no .js in {args.js_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fail_dir = args.out_dir / "_failed"
    if args.copy_fail:
        fail_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    passed = failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_map = {ex.submit(validate_one, p, args.node, cli): p for p in files}
        for fut in tqdm(concurrent.futures.as_completed(fut_map), total=len(fut_map), desc="validate"):
            src = fut_map[fut]
            res = fut.result()
            results.append(res)
            if res.get("passed"):
                shutil.copy2(src, args.out_dir / src.name)
                meta = src.with_suffix(".meta.json")
                if meta.is_file():
                    shutil.copy2(meta, args.out_dir / meta.name)
                passed += 1
            else:
                failed += 1
                if args.copy_fail:
                    shutil.copy2(src, fail_dir / src.name)

    report_path = args.report or (args.out_dir / "validate_report.json")
    report = {
        "js_dir": str(args.js_dir),
        "out_dir": str(args.out_dir),
        "total": len(files),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / max(len(files), 1), 4),
        "results": sorted(results, key=lambda r: r["stem"]),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"passed={passed} failed={failed} rate={report['pass_rate']} -> {args.out_dir}")
    print(f"report -> {report_path}")
    if passed == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
