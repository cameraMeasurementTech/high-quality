#!/usr/bin/env python3
"""Merge a LoRA adapter into the base VLM for Docker / vLLM serving.

Usage:
  python merge_lora.py \\
    --base handsometiger0202/Qwen-3.6-27B-AstroWolf \\
    --adapter ../data/checkpoints/sft_8b/final \\
    --out ../data/checkpoints/merged_coder
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge LoRA into base weights")
    ap.add_argument("--base", type=str, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"Loading base {args.base} …")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
        )

    print(f"Loading adapter {args.adapter} …")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    print("Merging …")
    model = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True)
    try:
        processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
        processor.save_pretrained(str(args.out))
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not save processor: {exc}")

    print(f"Merged weights -> {args.out}")
    print("Point my-agent configuration.yaml coder vllm.model at this path, rebuild Docker, duel.")


if __name__ == "__main__":
    main()
