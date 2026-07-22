#!/usr/bin/env python3
"""Patch pipeline YAML to use a local model path (merged weights or HF cache dir)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def patch_model(cfg: dict, model_path: str) -> None:
    model_path = str(Path(model_path).resolve())
    if "llm_clients" in cfg and "coder-instance" in cfg["llm_clients"]:
        cfg["llm_clients"]["coder-instance"].setdefault("vllm", {})["model"] = model_path
    for actor in ("planner", "coder"):
        if "actors" in cfg and actor in cfg["actors"]:
            cfg["actors"][actor]["model"] = model_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Inject local MODEL_PATH into pipeline config")
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--model",
        type=str,
        default="",
        help="Model path (default: MODEL_PATH or CODER_MODEL_PATH env)",
    )
    args = ap.parse_args()

    model = args.model.strip() or os.environ.get("MODEL_PATH", "").strip() or os.environ.get(
        "CODER_MODEL_PATH", ""
    ).strip()
    if not model:
        raise SystemExit("Set --model or MODEL_PATH / CODER_MODEL_PATH")

    with args.in_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    patch_model(cfg, model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Patched config -> {args.out}")
    print(f"  model = {Path(model).resolve()}")


if __name__ == "__main__":
    main()
