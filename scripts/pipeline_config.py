#!/usr/bin/env python3
"""Read pipeline YAML flags for standalone launch scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def needs_openrouter(cfg: dict) -> bool:
    llm = cfg.get("llm_clients") or {}
    openrouter = llm.get("openrouter") or {}
    if openrouter.get("enabled") is False:
        return False

    pipeline = cfg.get("pipeline") or {}
    if pipeline.get("refinement_enabled") is False and pipeline.get("use_planner") is False:
        actors = cfg.get("actors") or {}
        critic = actors.get("critic") or {}
        judge = actors.get("judge") or {}
        critic_client = str(critic.get("client", ""))
        judge_client = str(judge.get("client", ""))
        if critic_client != "openrouter" and judge_client != "openrouter":
            return False

    return bool(openrouter.get("enabled", True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--needs-openrouter", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.needs_openrouter:
        return 0 if needs_openrouter(cfg) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
