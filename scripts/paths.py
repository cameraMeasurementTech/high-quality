"""Shared paths for standalone training workspace.

Standalone layout (copy only this `training/` tree to a GPU box):

  training/
    vendor/shiny-guide/     # cloned by ./run/00_bootstrap_assets.sh
    data/prompts.txt        # validator image pool
    data/models/...         # local AstroWolf weights
    third_party/miner-reference/  # bundled validate.js
    pipeline/               # native vLLM launcher
    run/ scripts/ configs/

Monorepo layouts (404-gen-subnet with sibling shiny-guide/) still work via auto-detect.

Environment overrides (see .env.template):
  TRAINING_ROOT, WORKSPACE_ROOT, SHINY_GUIDE_ROOT, PROMPTS_POOL,
  MODEL_PATH, VALIDATE_JS
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


def vendor_shiny_guide_root() -> Path:
    return training_root() / "vendor" / "shiny-guide"


def workspace_root() -> Path:
    env = os.environ.get("WORKSPACE_ROOT", "").strip()
    if env:
        return Path(env).resolve()

    tr = training_root()
    vend = vendor_shiny_guide_root()
    if (vend / "pipeline_service").is_dir():
        return tr.resolve()

    for candidate in (tr.parent, tr.parent.parent):
        if (candidate / "shiny-guide" / "pipeline_service").is_dir():
            return candidate.resolve()
    return tr.resolve()


def repo_root() -> Path:
    """Legacy alias — same as workspace_root for path resolution."""
    return workspace_root()


def shiny_guide_root() -> Path:
    env = os.environ.get("SHINY_GUIDE_ROOT", "").strip()
    if env:
        p = Path(env).resolve()
        if (p / "pipeline_service").is_dir():
            return p

    vend = vendor_shiny_guide_root()
    if (vend / "pipeline_service").is_dir():
        return vend.resolve()

    sibling = workspace_root() / "shiny-guide"
    if (sibling / "pipeline_service").is_dir():
        return sibling.resolve()

    raise FileNotFoundError(
        "shiny-guide not found. Run ./run/00_bootstrap_assets.sh\n"
        f"  expected: {vend}/pipeline_service\n"
        f"  or set SHINY_GUIDE_ROOT=/path/to/shiny-guide"
    )


def my_agent_root() -> Path:
    """Optional challenger pipeline (monorepo only). Not used in standalone mode."""
    env = os.environ.get("MY_AGENT_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    candidate = workspace_root() / "my-agent"
    if candidate.is_dir():
        return candidate.resolve()
    return training_root().parent


def pipeline_root() -> Path:
    return training_root() / "pipeline"


def vendor_prompts_scene_coder_dir() -> Path:
    return training_root() / "vendor" / "pipeline_prompts" / "scene_coder"


def pipeline_root_for_prompts() -> Path:
    """Which tree supplies coder prompts for dataset packing (train=serve).

    Prefer vendored prompt snapshot (works without live shiny-guide checkout path),
    then shiny-guide root from bootstrap.
    """
    choice = os.environ.get("PROMPTS_ROOT", "shiny-guide").strip().lower()
    vend = vendor_prompts_scene_coder_dir()
    if choice in {"vendor", "bundled", "snapshot"} and (vend / "prompts.py").is_file():
        return training_root() / "vendor" / "pipeline_prompts"

    if choice in {"shiny-guide", "shiny", "sg", "vendor", "bundled", "snapshot"}:
        if (vend / "prompts.py").is_file():
            return training_root() / "vendor" / "pipeline_prompts"
        return shiny_guide_root()

    if choice in {"my-agent", "myagent", "ma"}:
        root = my_agent_root()
        if not (root / "pipeline_service").is_dir():
            raise FileNotFoundError(
                f"PROMPTS_ROOT=my-agent but {root}/pipeline_service not found "
                "(standalone workspace uses shiny-guide / vendor prompts)"
            )
        return root

    p = Path(choice)
    if p.is_dir():
        return p.resolve()
    raise ValueError(f"unknown PROMPTS_ROOT={choice!r}")


def default_data_root() -> Path:
    return training_root() / "data"


def default_model_dir() -> Path:
    return default_data_root() / "models" / "Qwen-3.6-27B-AstroWolf"


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
        "validate.js not found. Run ./run/00_install_all.sh or set VALIDATE_JS=\n"
        f"  expected bundled path: {bundled}"
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
    """Resolve the ~99k image URL pool for standalone training/.

    Priority:
      1. PROMPTS_POOL env (absolute or relative to training/)
      2. training/data/prompts.txt  (bootstrap default)
      3. training/prompts.txt
      4. workspace_root/prompts.txt (legacy monorepo)
    """
    env = os.environ.get("PROMPTS_POOL", "").strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = training_root() / p
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"PROMPTS_POOL not found: {p}\n"
                "Run ./run/00_bootstrap_assets.sh"
            )
        return p

    for candidate in (
        default_data_root() / "prompts.txt",
        training_root() / "prompts.txt",
        workspace_root() / "prompts.txt",
    ):
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "prompts.txt not found. Run ./run/00_bootstrap_assets.sh or set PROMPTS_POOL=\n"
        f"  expected: {default_data_root() / 'prompts.txt'} (~99k https URLs)"
    )


def ensure_repo_on_path() -> Path:
    """Allow importing scene_coder prompts from the active pipeline tree."""
    vend = vendor_prompts_scene_coder_dir()
    if (vend / "prompts.py").is_file():
        return vend

    root = shiny_guide_root()
    ps = root / "pipeline_service"
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
