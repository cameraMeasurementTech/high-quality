#!/usr/bin/env python3
"""Carve train / val / duel splits from prompts.txt with no stem overlap.

Usage:
  python prepare_splits.py \\
    --train 10000 --val 500 --duel 200 \\
    --seed 7 --out-dir ../data/splits
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from urllib.parse import urlparse

from paths import default_data_root, prompts_pool, repo_root


def url_to_stem(url: str) -> str:
    return Path(urlparse(url.strip()).path).stem


def load_urls(pool: Path) -> list[str]:
    urls: list[str] = []
    for line in pool.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("http"):
            urls.append(line)
        else:
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[1].startswith("http"):
                urls.append(parts[1])
    if not urls:
        raise SystemExit(f"no URLs in {pool}")
    return urls


def load_exclude_stems(paths: list[Path]) -> set[str]:
    stems: set[str] = set()
    for p in paths:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("http"):
                stems.add(url_to_stem(line))
            else:
                stems.add(line.split("\t", 1)[0].split()[0])
    return stems


def write_stem_url(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{stem}\t{url}" for stem, url in pairs) + ("\n" if pairs else "")
    path.write_text(body, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare non-overlapping train/val/duel splits")
    ap.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="prompts.txt (default: PROMPTS_POOL or training/data/prompts.txt)",
    )
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--duel", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[],
        help="stem/url lists to exclude (e.g. recent live round prompts)",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    pool = Path(args.pool).resolve() if args.pool else prompts_pool()
    out = Path(args.out_dir).resolve() if args.out_dir else (default_data_root() / "splits")

    if not pool.is_file():
        raise SystemExit(
            f"prompts pool not found: {pool}\n"
            "Run ./run/00_bootstrap_assets.sh (downloads ~99k URLs to data/prompts.txt)"
        )

    urls = load_urls(pool)
    exclude = load_exclude_stems(args.exclude)
    candidates = [(url_to_stem(u), u) for u in urls if url_to_stem(u) not in exclude]
    need = args.train + args.val + args.duel
    if need > len(candidates):
        raise SystemExit(f"need {need} stems but only {len(candidates)} available after exclude")

    rng = random.Random(args.seed)  # nosec B311
    rng.shuffle(candidates)

    duel = candidates[: args.duel]
    val = candidates[args.duel : args.duel + args.val]
    train = candidates[args.duel + args.val : args.duel + args.val + args.train]

    write_stem_url(out / "train.txt", train)
    write_stem_url(out / "val.txt", val)
    write_stem_url(out / "duel.txt", duel)

    meta = {
        "seed": args.seed,
        "pool": str(pool),
        "pool_urls": len(urls),
        "exclude_files": [str(p) for p in args.exclude],
        "exclude_stems": len(exclude),
        "counts": {"train": len(train), "val": len(val), "duel": len(duel)},
        "repo": str(repo_root()),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote splits -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
