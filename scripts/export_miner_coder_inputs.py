#!/usr/bin/env python3
"""Export miner-identical coder inputs (system + multimodal user) for inspection / teachers.

Matches shiny-guide SceneCoderAgent.code() when use_planner=false:
  system: CODER_SYSTEM_PROMPT  (plain string)
  user:   [ image_url(data:...), text(CODER_USER_TEMPLATE_IMAGE_ONLY) ]

Writes per stem:
  {out}/{stem}.request.json   — OpenRouter/OpenAI chat.completions payload (no huge base64 by default)
  {out}/{stem}.messages.json  — SFT-style messages with image path (for packing parity)
  {out}/_prompt_meta.json     — prompt source + lengths

Usage:
  python export_miner_coder_inputs.py \\
    --list data/splits/train.txt --images data/images \\
    --out data/miner_inputs/sft \\
    --include-base64   # optional: embed data URLs (large)
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from tqdm import tqdm

from coder_prompts import load_coder_prompts
from paths import default_data_root


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


def build_openai_messages(
    system: str,
    user_tpl: str,
    image_path: Path,
    *,
    include_base64: bool,
) -> list[dict]:
    if include_base64:
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        image_part: dict = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }
    else:
        image_part = {
            "type": "image_url",
            "image_url": {"url": f"file://{image_path.resolve()}"},
            "_note": "placeholder; teacher collect embeds real data: URL at call time",
        }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                image_part,
                {"type": "text", "text": user_tpl},
            ],
        },
    ]


def build_sft_messages(system: str, user_tpl: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_tpl},
            ],
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export miner coder inputs")
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--images", type=Path, default=default_data_root() / "images")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=str, default="openai/gpt-5-chat")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--include-base64", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    system, user_tpl = load_coder_prompts()
    pairs = load_pairs(args.list)
    if args.limit:
        pairs = pairs[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    missing = 0
    written = 0
    for stem, url in tqdm(pairs, desc="export-miner-inputs"):
        img = args.images / f"{stem}.png"
        if not img.is_file():
            missing += 1
            continue
        messages = build_openai_messages(
            system, user_tpl, img, include_base64=args.include_base64
        )
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": messages,
            "stem": stem,
            "reference_url": url,
            "image_path": str(img.resolve()),
            "miner_parity": {
                "use_planner": False,
                "image_first_then_text": True,
                "system": "CODER_SYSTEM_PROMPT",
                "user_text": "CODER_USER_TEMPLATE_IMAGE_ONLY",
            },
        }
        (args.out / f"{stem}.request.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        sft = {
            "stem": stem,
            "image": str(img.resolve()),
            "messages": build_sft_messages(system, user_tpl),
            "note": "assistant JS filled later by teacher; same prompts as pack_sft_dataset.py",
        }
        (args.out / f"{stem}.messages.json").write_text(
            json.dumps(sft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        written += 1

    meta = {
        "n_written": written,
        "missing_images": missing,
        "system_prompt_chars": len(system),
        "user_template_chars": len(user_tpl),
        "user_template_preview": user_tpl[:240],
        "model_default": args.model,
        "out": str(args.out.resolve()),
    }
    (args.out / "_prompt_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
