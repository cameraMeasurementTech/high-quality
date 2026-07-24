#!/usr/bin/env python3
"""Download prompt images for a stem<TAB>url list into data/images/{stem}.png.

Usage:
  python download_images.py --list ../data/splits/train.txt --out ../data/images --workers 16
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path

import httpx
from tqdm import tqdm

from paths import default_data_root

USER_AGENT = "404-gen-training/1.0 (+research; contact=miner)"


def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise SystemExit(f"bad line (want stem\\turl): {line!r}")
        pairs.append((parts[0], parts[1]))
    return pairs


def fetch_one(stem: str, url: str, out_dir: Path, timeout: float, retries: int) -> str:
    dest = out_dir / f"{stem}.png"
    if dest.is_file() and dest.stat().st_size > 0:
        return "skip"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
                r = client.get(url)
                r.raise_for_status()
                dest.write_bytes(r.content)
            return "ok"
        except Exception as exc:  # noqa: BLE001 — network resilience
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"{stem}: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache prompt PNGs locally")
    ap.add_argument("--list", type=Path, required=True, help="stem\\turl file")
    ap.add_argument("--out", type=Path, default=default_data_root() / "images")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument(
        "--fail-ok",
        action="store_true",
        help="do not exit non-zero when some downloads fail (recommended for ~99k pools)",
    )
    ap.add_argument(
        "--max-fail-ratio",
        type=float,
        default=0.05,
        help="with default strict mode, allow up to this fraction of failures before exit 1",
    )
    args = ap.parse_args()

    pairs = load_pairs(args.list)
    args.out.mkdir(parents=True, exist_ok=True)

    ok = skip = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(fetch_one, stem, url, args.out, args.timeout, args.retries): stem
            for stem, url in pairs
        }
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="download"):
            stem = futs[fut]
            try:
                status = fut.result()
                if status == "skip":
                    skip += 1
                else:
                    ok += 1
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"FAIL {stem}: {exc}", file=sys.stderr)

    total = ok + skip + fail
    ratio = (fail / total) if total else 1.0
    print(f"done ok={ok} skip={skip} fail={fail} fail_ratio={ratio:.3f} -> {args.out}")
    if fail and not args.fail_ok and ratio > float(args.max_fail_ratio):
        sys.exit(1)
    if fail and ok + skip == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
