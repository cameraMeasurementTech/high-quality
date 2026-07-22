#!/usr/bin/env python3
"""Pack filtered (image, js) pairs into a HuggingFace datasets arrow/jsonl for SFT.

Uses the same coder system/user prompts as production my-agent (train=serve).

Usage:
  python pack_sft_dataset.py \\
    --js-dir ../data/filtered_js \\
    --images ../data/images \\
    --list ../data/splits/train.txt \\
    --out ../data/hf/sft_train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, Image
from tqdm import tqdm

from coder_prompts import load_coder_prompts
from paths import default_data_root


def load_stem_set(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    stems: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stems.add(line.split()[0])
    return stems


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack SFT HF dataset")
    ap.add_argument("--js-dir", type=Path, required=True)
    ap.add_argument("--images", type=Path, default=default_data_root() / "images")
    ap.add_argument("--list", type=Path, default=None, help="optional stem allow-list")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-tag", type=str, default="teacher")
    args = ap.parse_args()

    CODER_SYSTEM_PROMPT, CODER_USER_TEMPLATE_IMAGE_ONLY = load_coder_prompts()

    allow = load_stem_set(args.list)
    rows: list[dict] = []
    missing_img = 0

    for js_path in tqdm(sorted(args.js_dir.glob("*.js")), desc="pack-sft"):
        stem = js_path.stem
        if allow is not None and stem not in allow:
            continue
        img_path = args.images / f"{stem}.png"
        if not img_path.is_file():
            missing_img += 1
            continue
        js = js_path.read_text(encoding="utf-8").strip()
        if "export default function generate" not in js:
            continue

        # Chat messages in the shape TRL / Qwen-VL SFT expects.
        # Image is referenced as a path; the trainer loads PIL at train time.
        messages = [
            {"role": "system", "content": [{"type": "text", "text": CODER_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": CODER_USER_TEMPLATE_IMAGE_ONLY},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": js}]},
        ]
        rows.append(
            {
                "stem": stem,
                "images": [{"bytes": None, "path": str(img_path.resolve())}],
                "messages": messages,
                "source": args.source_tag,
            }
        )

    if not rows:
        raise SystemExit("no rows packed — check js-dir / images / list")

    # Save as jsonl + separate image paths for portability, and as HF Dataset.
    args.out.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out / "sft.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "stem": row["stem"],
                        "image": row["images"][0]["path"],
                        "messages": row["messages"],
                        "source": row["source"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # HF dataset with Image feature for trainer convenience
    ds_rows = [
        {
            "stem": r["stem"],
            "image": r["images"][0]["path"],
            "messages": json.dumps(r["messages"]),
            "source": r["source"],
        }
        for r in rows
    ]
    ds = Dataset.from_list(ds_rows)
    ds = ds.cast_column("image", Image())
    ds.save_to_disk(str(args.out / "dataset"))

    meta = {
        "n": len(rows),
        "missing_images": missing_img,
        "jsonl": str(jsonl_path),
        "dataset": str(args.out / "dataset"),
        "system_prompt_chars": len(CODER_SYSTEM_PROMPT),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
