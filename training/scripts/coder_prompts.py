"""Load production coder prompts without importing the full pipeline_service package."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


from paths import pipeline_root_for_prompts


def load_coder_prompts() -> tuple[str, str]:
    pipeline_root = pipeline_root_for_prompts()
    scene_coder_dir = pipeline_root / "pipeline_service" / "modules" / "scene_coder"
    prompts_path = scene_coder_dir / "prompts.py"
    if not prompts_path.is_file():
        raise FileNotFoundError(prompts_path)

    ps = pipeline_root / "pipeline_service"
    if str(ps) not in sys.path:
        sys.path.insert(0, str(ps))

    # Stub packages so `from modules.scene_coder.X` does NOT execute __init__.py
    # (which would pull SceneCoderAgent → pydantic → full stack).
    if "modules" not in sys.modules:
        sys.modules["modules"] = types.ModuleType("modules")
    if "modules.scene_coder" not in sys.modules:
        pkg = types.ModuleType("modules.scene_coder")
        pkg.__path__ = [str(scene_coder_dir)]  # type: ignore[attr-defined]
        sys.modules["modules.scene_coder"] = pkg

    for name in ("few_shot_examples", "threejs_reference"):
        mod_name = f"modules.scene_coder.{name}"
        if mod_name in sys.modules:
            continue
        leaf = scene_coder_dir / f"{name}.py"
        leaf_spec = importlib.util.spec_from_file_location(mod_name, leaf)
        assert leaf_spec and leaf_spec.loader
        leaf_mod = importlib.util.module_from_spec(leaf_spec)
        sys.modules[mod_name] = leaf_mod
        leaf_spec.loader.exec_module(leaf_mod)

    spec = importlib.util.spec_from_file_location(
        "modules.scene_coder.prompts", prompts_path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modules.scene_coder.prompts"] = mod
    spec.loader.exec_module(mod)
    return mod.CODER_SYSTEM_PROMPT, mod.CODER_USER_TEMPLATE_IMAGE_ONLY
