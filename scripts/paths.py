"""Shared paths for standalone shiny-guide + training workspace.

Standalone layout (recommended):
  workspace/
    shiny-guide/
    training/          # this tree (may live at my-agent/training inside monorepo)
    prompts.txt        # optional; or training/data/prompts.txt

Monorepo layout is still supported via auto-detection.

Environment overrides (see training/.env.template):
  WORKSPACE_ROOT, SHINY_GUIDE_ROOT, TRAINING_ROOT, PROMPTS_POOL, VALIDATE_JS
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def training_root() -> Path:
    env = os.environ.get("TRAINING_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    env = os.environ.get("WORKSPACE_ROOT", "").strip()
    if env:
        return Path(env).resolve()

    tr = training_root()
    for candidate in (tr.parent, tr.parent.parent):
        if (candidate / "shiny-guide").is_dir():
            return candidate.resolve()
    return tr.parent.resolve()


def repo_root() -> Path:
    """Legacy alias — same as workspace_root for path resolution."""
    return workspace_root()


def shiny_guide_root() -> Path:
    env = os.environ.get("SHINY_GUIDE_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return workspace_root() / "shiny-guide"


def my_agent_root() -> Path:
    """Optional challenger pipeline (monorepo only). Not required for standalone."""
    env = os.environ.get("MY_AGENT_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    candidate = workspace_root() / "my-agent"
    if candidate.is_dir():
        return candidate.resolve()
    return training_root().parent


def pipeline_root() -> Path:
    return training_root() / "pipeline"


def pipeline_root_for_prompts() -> Path:
    """Which pipeline_service tree supplies coder prompts (train=serve).

    Standalone: always shiny-guide. Set PROMPTS_ROOT only for monorepo my-agent fork.
    """
    choice = os.environ.get("PROMPTS_ROOT", "shiny-guide").strip().lower()
    if choice in {"shiny-guide", "shiny", "sg"}:
        return shiny_guide_root()
    if choice in {"my-agent", "myagent", "ma"}:
        root = my_agent_root()
        if not (root / "pipeline_service").is_dir():
            raise FileNotFoundError(
                f"PROMPTS_ROOT=my-agent but {root}/pipeline_service not found "
                "(standalone workspace only needs shiny-guide)"
            )
        return root
    p = Path(choice)
    if p.is_dir():
        return p.resolve()
    raise ValueError(f"unknown PROMPTS_ROOT={choice!r}")


def default_data_root() -> Path:
    return training_root() / "data"


def bundled_validator_root() -> Path:
    return training_root() / "third_party" / "miner-reference"


def validate_js_cli() -> Path:
    env = os.environ.get("VALIDATE_JS", "").strip()
    if env:
        return Path(env).resolve()

    bundled = bundled_validator_root() / "tools" / "validate.js"
    if bundled.is_file():
        return bundled

    monorepo = workspace_root() / "miner-reference" / "tools" / "validate.js"
    if monorepo.is_file():
        return monorepo

    raise FileNotFoundError(
        "validate.js not found. Run ./run/00_setup.sh or set VALIDATE_JS= "
        f"(expected bundled path: {bundled})"
    )


def validator_npm_root() -> Path:
    env = os.environ.get("VALIDATOR_ROOT", "").strip()
    if env:
        return Path(env).resolve()

    bundled = bundled_validator_root() / "validator"
    if (bundled / "package.json").is_file():
        return bundled

    monorepo = workspace_root() / "miner-reference" / "validator"
    if (monorepo / "package.json").is_file():
        return monorepo

    return bundled


def prompts_pool() -> Path:
    env = os.environ.get("PROMPTS_POOL", "").strip()
    if env:
        p = Path(env).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"PROMPTS_POOL not found: {p}")
        return p

    for candidate in (
        workspace_root() / "prompts.txt",
        default_data_root() / "prompts.txt",
        training_root() / "prompts.txt",
    ):
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "prompts.txt not found. Copy the validator image pool to one of:\n"
        f"  {workspace_root() / 'prompts.txt'}\n"
        f"  {default_data_root() / 'prompts.txt'}\n"
        "Or set PROMPTS_POOL=/path/to/prompts.txt"
    )


def ensure_repo_on_path() -> Path:
    """Allow importing scene_coder prompts from the active pipeline tree."""
    ps = pipeline_root_for_prompts() / "pipeline_service"
    if str(ps) not in sys.path:
        sys.path.insert(0, str(ps))
    return ps


def resolve_model_path(cfg_value: str) -> str:
    """Resolve model path from config — env MODEL_PATH / CODER_MODEL_PATH override HF id."""
    override = (
        os.environ.get("MODEL_PATH", "").strip()
        or os.environ.get("CODER_MODEL_PATH", "").strip()
    )
    if override:
        return str(Path(override).resolve())
    return cfg_value
