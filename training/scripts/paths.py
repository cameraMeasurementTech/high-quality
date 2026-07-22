"""Shared paths for the my-agent training pipeline."""
from __future__ import annotations

import sys
from pathlib import Path


def training_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def shiny_guide_root() -> Path:
    return repo_root() / "shiny-guide"


def my_agent_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pipeline_root_for_prompts() -> Path:
    """Which pipeline_service tree supplies coder prompts (train=serve).

    Set PROMPTS_ROOT=shiny-guide (default) or my-agent.
    """
    import os

    choice = os.environ.get("PROMPTS_ROOT", "shiny-guide").strip().lower()
    if choice in {"shiny-guide", "shiny", "sg"}:
        return shiny_guide_root()
    if choice in {"my-agent", "myagent", "ma"}:
        return my_agent_root()
    p = Path(choice)
    if p.is_dir():
        return p
    raise ValueError(f"unknown PROMPTS_ROOT={choice!r}")


def default_data_root() -> Path:
    return training_root() / "data"


def ensure_repo_on_path() -> Path:
    """Allow importing scene_coder prompts from my-agent pipeline_service."""
    ps = my_agent_root() / "pipeline_service"
    if str(ps) not in sys.path:
        sys.path.insert(0, str(ps))
    return ps


def validate_js_cli() -> Path:
    return repo_root() / "miner-reference" / "tools" / "validate.js"


def prompts_pool() -> Path:
    return repo_root() / "prompts.txt"
