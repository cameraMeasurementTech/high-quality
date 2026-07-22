#!/usr/bin/env python3
"""Pack prompt-only HF dataset for online GRPO (completions sampled on-policy).

Usage:
  python pack_grpo_prompts.py \\
    --list ../data/splits/train.txt \\
    --images ../data/images \\
    --out ../data/hf/grpo_train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, Image
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
        if len(parts) >= 1:
            stem = parts[0]
            url = parts[1] if len(parts) > 1 else ""
            pairs.append((stem, url))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack GRPO prompt dataset")
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--images", type=Path, default=default_data_root() / "images")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    CODER_SYSTEM_PROMPT, CODER_USER_TEMPLATE_IMAGE_ONLY = load_coder_prompts()

    rows: list[dict] = []
    missing = 0
    for stem, url in tqdm(load_pairs(args.list), desc="pack-grpo"):
        img = args.images / f"{stem}.png"
        if not img.is_file():
            missing += 1
            continue
        # Store prompt messages; train_grpo.py applies the processor chat template.
        prompt = [
            {"role": "system", "content": [{"type": "text", "text": CODER_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": CODER_USER_TEMPLATE_IMAGE_ONLY},
                ],
            },
        ]
        rows.append(
            {
                "stem": stem,
                "image": str(img.resolve()),
                "prompt": json.dumps(prompt),
                "image_url": url,
            }
        )

    if not rows:
        raise SystemExit("no GRPO rows — download images first")

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out / "grpo.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ds = Dataset.from_list(rows)
    ds = ds.cast_column("image", Image())
    ds.save_to_disk(str(args.out / "dataset"))

    meta = {"n": len(rows), "missing_images": missing, "dataset": str(args.out / "dataset")}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
