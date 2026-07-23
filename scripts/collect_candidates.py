#!/usr/bin/env python3
"""Collect K JS candidates per prompt stem for offline DPO pair mining.

Writes:
  {out}/{stem}/sample_{i}.js
  {out}/{stem}/sample_{i}.meta.json

Diversity strategy (pipeline mode — recommended for duel-scored DPO):
  Same image + same coder system/user prompts + same temperature (from
  pipeline yaml, typically 0.6).  Only the RNG *seed* changes per sample:

    sample_i seed = base_seed + sample_i * seed_stride + batch_i

  This matches production multigen (ensemble_size>1 uses seed+k at fixed
  ensemble_temperature). Prefer seeds over temperature sweeps: high temp
  increases invalid JS and weakens pairwise preference quality.

Modes:
  --from-pipeline   POST /generate K times per stem (different seeds)
  --from-openai     Sample K completions (can vary temperature)
  --from-dir        Copy existing flat *.js dirs into per-stem folders

Usage:
  # Duel DPO: 2 candidates, large batches on 4× H200
  python collect_candidates.py --from-pipeline \\
    --list data/splits/train.txt --base-url http://127.0.0.1:10006 \\
    --samples 2 --batch-size 48 --out data/candidates/shiny_k2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from collect_teacher_js import collect_from_openai, load_pairs, strip_fences
from paths import default_data_root
from pipeline_client import submit_and_wait


def _stem_dir(out: Path, stem: str) -> Path:
    d = out / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def sample_seed(base_seed: int, sample_i: int, seed_stride: int, batch_i: int) -> int:
    return base_seed + sample_i * seed_stride + batch_i


def collect_pipeline_multi(
    pairs: list[tuple[str, str]],
    base_url: str,
    out_dir: Path,
    samples: int,
    timeout: float,
    skip_existing: bool,
    batch_size: int = 16,
    base_seed: int = 42,
    seed_stride: int = 1000,
) -> tuple[int, int]:
    """Run K batch generations with different seeds for diversity.

    Same prompt/image/temperature each time — only seed changes (production-style).
    """
    ok = fail = 0
    for sample_i in range(samples):
        pending: list[tuple[str, str]] = []
        for stem, url in pairs:
            dest = _stem_dir(out_dir, stem) / f"sample_{sample_i}.js"
            if skip_existing and dest.is_file() and dest.stat().st_size > 0:
                ok += 1
                continue
            pending.append((stem, url))

        for batch_i in tqdm(
            range(0, len(pending), batch_size),
            desc=f"pipeline-sample-{sample_i}",
        ):
            chunk = pending[batch_i : batch_i + batch_size]
            prompts = [{"stem": stem, "image_url": url} for stem, url in chunk]
            seed = sample_seed(base_seed, sample_i, seed_stride, batch_i)
            try:
                js_map = submit_and_wait(
                    base_url,
                    prompts,
                    seed=seed,
                    timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001
                fail += len(chunk)
                print(f"batch fail sample={sample_i}: {exc}", file=sys.stderr)
                continue

            for stem, url in chunk:
                source = js_map.get(stem)
                if not source:
                    fail += 1
                    continue
                js = strip_fences(str(source))
                dest = _stem_dir(out_dir, stem) / f"sample_{sample_i}.js"
                dest.write_text(js if js.endswith("\n") else js + "\n", encoding="utf-8")
                meta = {
                    "source": "pipeline",
                    "sample_index": sample_i,
                    "url": url,
                    "seed": seed,
                    "diversity": "seed",
                    "note": (
                        "Same prompt/image/temperature as other samples; "
                        "only RNG seed differs (production multigen style)."
                    ),
                }
                (dest.with_suffix(".meta.json")).write_text(
                    json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                )
                ok += 1
    return ok, fail


def collect_openai_multi(
    pairs: list[tuple[str, str]],
    images_dir: Path,
    base_url: str,
    model: str,
    api_key: str,
    out_dir: Path,
    samples: int,
    temperatures: list[float],
    max_tokens: int,
    skip_existing: bool,
) -> tuple[int, int]:
    ok = fail = 0
    temps = temperatures or [0.7]
    for i in range(samples):
        temp = temps[i % len(temps)]
        tmp = out_dir / f"_flat_sample_{i}"
        tmp.mkdir(parents=True, exist_ok=True)
        sub_ok, sub_fail = collect_from_openai(
            pairs,
            images_dir,
            base_url,
            model,
            api_key,
            tmp,
            max_tokens,
            temp,
            skip_existing=False,
        )
        ok += sub_ok
        fail += sub_fail
        for js_path in tmp.glob("*.js"):
            stem = js_path.stem
            dest_dir = _stem_dir(out_dir, stem)
            dest = dest_dir / f"sample_{i}.js"
            if skip_existing and dest.is_file() and dest.stat().st_size > 0:
                continue
            dest.write_text(js_path.read_text(encoding="utf-8"), encoding="utf-8")
            meta = {
                "source": "openai",
                "model": model,
                "sample_index": i,
                "temperature": temp,
                "diversity": "temperature",
            }
            (dest.with_suffix(".meta.json")).write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
        for p in tmp.glob("*"):
            p.unlink(missing_ok=True)
        tmp.rmdir()
    return ok, fail


def collect_from_flat_dir(src_dir: Path, out_dir: Path) -> int:
    """Copy flat {stem}.js into {out}/{stem}/sample_0.js (one candidate per stem)."""
    n = 0
    for js in sorted(src_dir.glob("*.js")):
        stem = js.stem
        if stem.endswith("_osd"):
            continue
        dest = _stem_dir(out_dir, stem) / "sample_0.js"
        dest.write_text(js.read_text(encoding="utf-8"), encoding="utf-8")
        meta = {"source": str(js), "sample_index": 0}
        (dest.with_suffix(".meta.json")).write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect K JS candidates per stem for DPO")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-pipeline", action="store_true")
    mode.add_argument("--from-openai", action="store_true")
    mode.add_argument("--from-dir", action="store_true")

    ap.add_argument("--list", type=Path, help="stem\\turl list")
    ap.add_argument("--src-dir", type=Path, help="flat js dir (--from-dir)")
    ap.add_argument("--images", type=Path, default=default_data_root() / "images")
    ap.add_argument("--base-url", type=str, default="http://127.0.0.1:10006")
    ap.add_argument("--model", type=str, default="google/gemini-2.5-pro-preview")
    ap.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=2, help="candidates per stem (2 for duel DPO)")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "16")),
        help="stems per /generate call (raise to 48–64 on 4× H200)",
    )
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument(
        "--seed-stride",
        type=int,
        default=1000,
        help="seed spacing between sample indices (sample_i * stride)",
    )
    ap.add_argument("--temperatures", type=str, default="0.5,0.7,0.9,1.0")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.from_dir:
        if not args.src_dir:
            raise SystemExit("--from-dir requires --src-dir")
        n = collect_from_flat_dir(args.src_dir, args.out)
        print(json.dumps({"mode": "from-dir", "stems": n, "out": str(args.out)}, indent=2))
        return

    if not args.list:
        raise SystemExit("--list required for pipeline/openai modes")
    pairs = load_pairs(args.list)

    if args.from_pipeline:
        ok, fail = collect_pipeline_multi(
            pairs,
            args.base_url,
            args.out,
            args.samples,
            args.timeout,
            args.skip_existing,
            batch_size=max(1, args.batch_size),
            base_seed=args.base_seed,
            seed_stride=args.seed_stride,
        )
        print(
            json.dumps(
                {
                    "mode": "pipeline",
                    "ok": ok,
                    "fail": fail,
                    "samples": args.samples,
                    "batch_size": args.batch_size,
                    "diversity": "seed",
                    "out": str(args.out),
                },
                indent=2,
            )
        )
        if fail and ok == 0:
            sys.exit(1)
        return

    api_key = os.environ.get(args.api_key_env, "") or os.environ.get("OPENAI_API_KEY", "local")
    temps = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]
    ok, fail = collect_openai_multi(
        pairs,
        args.images,
        args.base_url,
        args.model,
        api_key,
        args.out,
        args.samples,
        temps,
        args.max_tokens,
        args.skip_existing,
    )
    print(
        json.dumps(
            {
                "mode": "openai",
                "ok": ok,
                "fail": fail,
                "samples": args.samples,
                "out": str(args.out),
            },
            indent=2,
        )
    )
    if fail and ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
