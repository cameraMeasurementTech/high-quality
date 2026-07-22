"""Load production coder prompts without importing the full pipeline_service package."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from paths import pipeline_root_for_prompts, shiny_guide_root, vendor_prompts_scene_coder_dir


def _scene_coder_dir() -> Path:
    vend = vendor_prompts_scene_coder_dir()
    if (vend / "prompts.py").is_file():
        return vend
    sg = shiny_guide_root()
    scene = sg / "pipeline_service" / "modules" / "scene_coder"
    if (scene / "prompts.py").is_file():
        return scene
    raise FileNotFoundError(
        f"coder prompts missing. Run ./run/00_bootstrap_assets.sh\n"
        f"  tried: {vend}\n"
        f"  tried: {scene}"
    )


def load_coder_prompts() -> tuple[str, str]:
    _ = pipeline_root_for_prompts()  # validate PROMPTS_ROOT
    scene_coder_dir = _scene_coder_dir()
    prompts_path = scene_coder_dir / "prompts.py"

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
        if not leaf.is_file():
            raise FileNotFoundError(f"missing {leaf} — re-run ./run/00_bootstrap_assets.sh")
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
