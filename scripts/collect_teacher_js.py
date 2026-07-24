#!/usr/bin/env python3
"""Collect teacher JS for SFT from a running miner pipeline or existing run dirs.

Modes:
  --from-pipeline   POST each image URL to a miner /generate endpoint
  --from-runs       Copy *.js from one or more local-eval run directories
  --from-openai     Call an OpenAI-compatible VLM with the frozen coder prompts

Usage:
  # From shiny-guide / my-agent HTTP API (CPU OpenRouter or GPU Docker)
  python collect_teacher_js.py --from-pipeline \\
    --list ../data/splits/train.txt --base-url http://127.0.0.1:10006 \\
    --out ../data/raw_js/shiny-guide

  # Harvest already-generated JS from local-eval runs
  python collect_teacher_js.py --from-runs \\
    --run-dirs ../../local-eval/runs/duel/shiny-guide \\
               ../../local-eval/runs/pool \\
    --out ../data/raw_js/harvest

  # Direct teacher API (OpenRouter / local vLLM)
  python collect_teacher_js.py --from-openai \\
    --list ../data/splits/train.txt --images ../data/images \\
    --base-url https://openrouter.ai/api/v1 --model google/gemini-2.5-pro-preview \\
    --out ../data/raw_js/teacher
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from tqdm import tqdm

from coder_prompts import load_coder_prompts
from paths import default_data_root
from pipeline_client import submit_and_wait

FENCE_RE = re.compile(r"```(?:javascript|js)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise SystemExit(f"bad line: {line!r}")
        pairs.append((parts[0], parts[1]))
    return pairs


def strip_fences(text: str) -> str:
    text = text.strip()
    m = FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def write_js(out_dir: Path, stem: str, source: str, meta: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.js").write_text(source if source.endswith("\n") else source + "\n", encoding="utf-8")
    if meta is not None:
        (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def collect_from_runs(run_dirs: list[Path], out_dir: Path) -> int:
    n = 0
    for rd in run_dirs:
        if not rd.is_dir():
            print(f"skip missing run dir: {rd}", file=sys.stderr)
            continue
        for js in sorted(rd.glob("*.js")):
            stem = js.stem
            if stem.endswith("_osd"):
                continue
            write_js(out_dir, stem, js.read_text(encoding="utf-8"), {"source": str(js)})
            n += 1
    return n


def collect_from_pipeline(
    pairs: list[tuple[str, str]],
    base_url: str,
    out_dir: Path,
    timeout: float,
    skip_existing: bool,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[int, int]:
    """Collect JS via shiny-guide batch API: POST /generate → poll → GET /debug/tasks/{stem}."""
    ok = fail = 0
    pending: list[tuple[str, str]] = []
    for stem, url in pairs:
        dest = out_dir / f"{stem}.js"
        if skip_existing and dest.is_file() and dest.stat().st_size > 0:
            ok += 1
            continue
        pending.append((stem, url))

    for i in tqdm(range(0, len(pending), batch_size), desc="pipeline-batch"):
        chunk = pending[i : i + batch_size]
        prompts = [{"stem": stem, "image_url": url} for stem, url in chunk]
        try:
            js_map = submit_and_wait(
                base_url,
                prompts,
                seed=seed + i,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            fail += len(chunk)
            print(f"batch fail: {exc}", file=sys.stderr)
            continue

        for stem, url in chunk:
            source = js_map.get(stem)
            if not source:
                fail += 1
                print(f"FAIL {stem}: no js_code in task", file=sys.stderr)
                continue
            write_js(
                out_dir,
                stem,
                strip_fences(str(source)),
                {"source": "pipeline", "url": url, "base_url": base_url},
            )
            ok += 1
    return ok, fail


def collect_from_openai(
    pairs: list[tuple[str, str]],
    images_dir: Path,
    base_url: str,
    model: str,
    api_key: str,
    out_dir: Path,
    max_tokens: int,
    temperature: float,
    skip_existing: bool,
    workers: int = 4,
    save_request: bool = False,
) -> tuple[int, int]:
    """Call OpenRouter/OpenAI with the *same* messages the miner coder receives."""
    import concurrent.futures

    CODER_SYSTEM_PROMPT, CODER_USER_TEMPLATE_IMAGE_ONLY = load_coder_prompts()

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if "openrouter.ai" in base_url:
        headers["HTTP-Referer"] = "https://github.com/404-Repo/404-gen-subnet"
        headers["X-Title"] = "404-gen-training-sft-teacher"

    def one(stem: str, url: str) -> tuple[str, str | None, str | None]:
        dest = out_dir / f"{stem}.js"
        if skip_existing and dest.is_file() and dest.stat().st_size > 0:
            return stem, "skip", None
        img_path = images_dir / f"{stem}.png"
        if not img_path.is_file():
            return stem, None, f"missing image {img_path}"
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        # Miner parity: system string + user [image_url, text] (image first).
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": CODER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": CODER_USER_TEMPLATE_IMAGE_ONLY},
                    ],
                },
            ],
        }
        if save_request:
            slim = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stem": stem,
                "image_path": str(img_path.resolve()),
                "messages": [
                    {"role": "system", "content": CODER_SYSTEM_PROMPT[:200] + "…"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,<{len(b64)} chars>"}},
                            {"type": "text", "text": CODER_USER_TEMPLATE_IMAGE_ONLY[:200] + "…"},
                        ],
                    },
                ],
            }
            (out_dir / f"{stem}.request_meta.json").write_text(
                json.dumps(slim, indent=2) + "\n", encoding="utf-8"
            )
        try:
            with httpx.Client(timeout=600.0, headers=headers) as client:
                r = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                source = strip_fences(str(content))
                write_js(
                    out_dir,
                    stem,
                    source,
                    {
                        "source": "openai",
                        "model": model,
                        "url": url,
                        "temperature": temperature,
                        "miner_parity": True,
                        "base_url": base_url,
                    },
                )
                return stem, "ok", None
        except Exception as exc:  # noqa: BLE001
            return stem, None, str(exc)

    ok = fail = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, stem, url) for stem, url in pairs]
        for fut in tqdm(
            concurrent.futures.as_completed(futs), total=len(futs), desc="openai-teacher"
        ):
            stem, status, err = fut.result()
            if status in {"ok", "skip"}:
                ok += 1
            else:
                fail += 1
                print(f"FAIL {stem}: {err}", file=sys.stderr)
    return ok, fail


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect teacher JS for SFT")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-pipeline", action="store_true")
    mode.add_argument("--from-runs", action="store_true")
    mode.add_argument("--from-openai", action="store_true")

    ap.add_argument("--list", type=Path, help="stem\\turl list (pipeline/openai)")
    ap.add_argument("--run-dirs", type=Path, nargs="+", help="dirs with *.js (from-runs)")
    ap.add_argument("--base-url", type=str, default="http://127.0.0.1:10006")
    ap.add_argument("--model", type=str, default=os.environ.get("TEACHER_MODEL", "openai/gpt-5-chat"))
    ap.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY")
    ap.add_argument("--images", type=Path, default=default_data_root() / "images")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("TEACHER_MAX_TOKENS", "24576")))
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("TEACHER_TEMPERATURE", "0.4")))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("TEACHER_WORKERS", "4")),
        help="parallel OpenRouter requests",
    )
    ap.add_argument(
        "--save-request",
        action="store_true",
        help="write slim *.request_meta.json next to each JS (miner payload summary)",
    )
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    args = ap.parse_args()

    if args.from_runs:
        if not args.run_dirs:
            raise SystemExit("--from-runs requires --run-dirs")
        n = collect_from_runs(args.run_dirs, args.out)
        print(f"copied {n} js files -> {args.out}")
        return

    if not args.list:
        raise SystemExit("--list required for pipeline/openai modes")
    pairs = load_pairs(args.list)

    if args.from_pipeline:
        ok, fail = collect_from_pipeline(
            pairs,
            args.base_url,
            args.out,
            args.timeout,
            args.skip_existing,
            args.batch_size,
            args.seed,
        )
        print(f"pipeline ok={ok} fail={fail} -> {args.out}")
        if fail and ok == 0:
            sys.exit(1)
        return

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        # Also accept OPENAI_API_KEY for local vLLM
        api_key = os.environ.get("OPENAI_API_KEY", "local")
    ok, fail = collect_from_openai(
        pairs,
        args.images,
        args.base_url,
        args.model,
        api_key,
        args.out,
        args.max_tokens,
        args.temperature,
        args.skip_existing,
        workers=args.workers,
        save_request=args.save_request,
    )
    print(f"openai ok={ok} fail={fail} model={args.model} -> {args.out}")
    if fail and ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
