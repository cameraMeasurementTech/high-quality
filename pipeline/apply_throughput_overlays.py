#!/usr/bin/env python3
"""Apply throughput overlays onto shiny-guide pipeline_service (idempotent).

Patches:
  - pipeline.skip_render support (JS-only Phase A collect)
  - prepare/download concurrency when use_planner=false (was hard-capped at 1)

Usage:
  python pipeline/apply_throughput_overlays.py
  python pipeline/apply_throughput_overlays.py /path/to/shiny-guide
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def training_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_shiny(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).resolve()
    else:
        sys.path.insert(0, str(training_root() / "scripts"))
        from paths import shiny_guide_root

        root = shiny_guide_root()
    ps = root / "pipeline_service"
    if not ps.is_dir():
        raise SystemExit(f"pipeline_service not found under {root}")
    return ps


def patch_file(path: Path, old: str, new: str, label: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new.strip() in text and old not in text:
        print(f"  [ok] {label} already applied")
        return False
    if old not in text:
        if "skip_render" in text and label.startswith("settings"):
            print(f"  [ok] {label} already present (variant)")
            return False
        print(f"  [warn] {label}: pattern not found in {path}", file=sys.stderr)
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"  [patch] {label}")
    return True


def main() -> int:
    ps = resolve_shiny(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"==> Applying throughput overlays → {ps}")

    settings = ps / "config" / "settings.py"
    patch_file(
        settings,
        """    refinement_enabled: bool = True


class VllmServeConfig(BaseModel):""",
        """    refinement_enabled: bool = True
    # When true, skip Chromium render after code+check. Use for JS-only data
    # collection (e.g. DPO candidate collect); render later during duel scoring.
    # Ignored when coder.ensemble_size > 1 (multigen judge needs renders).
    skip_render: bool = False


class VllmServeConfig(BaseModel):""",
        "settings.skip_render",
    )

    factory = ps / "pipeline" / "factory.py"
    text = factory.read_text(encoding="utf-8")
    if "prepare_limit" in text and "skip_render" in text:
        print("  [ok] factory already patched")
    else:
        old = """    return Pipeline(
        planner=planner,
        coder=coder,
        critic=critic,
        judge=judge,
        embedder=embedder,
        js_checker=js_checker,
        renderer=renderer,
        session_store=session_store,
        http_client=http_client,
        coder_multimodal=actors.coder.multimodal,
        use_planner=use_planner,
        coder_ensemble_size=ensemble_size,
        coder_ensemble_temperature=actors.coder.ensemble_temperature,
        render_from_object=settings.pipeline.render_from_object,
        refinement_enabled=settings.pipeline.refinement_enabled,
        max_iter=policy.max_iter,
        score_threshold=policy.score_threshold,
        task_deadline_s=policy.task_deadline_s,
        planner_limit=actors.planner.workers if use_planner else 1,
        coder_limit=actors.coder.workers,
        js_checker_limit=actors.checker.workers,
        renderer_limit=actors.renderer.workers,
        critic_limit=actors.critic.workers,
        judge_limit=actors.judge.workers,
    )"""
        new = """    # prepare_inputs always uses the planner semaphore for image downloads,
    # even when use_planner=false — do not collapse that to 1.
    if use_planner:
        prepare_limit = actors.planner.workers
    else:
        prepare_limit = max(actors.planner.workers, actors.coder.workers, 32)

    skip_render = bool(settings.pipeline.skip_render) and ensemble_size <= 1

    return Pipeline(
        planner=planner,
        coder=coder,
        critic=critic,
        judge=judge,
        embedder=embedder,
        js_checker=js_checker,
        renderer=renderer,
        session_store=session_store,
        http_client=http_client,
        coder_multimodal=actors.coder.multimodal,
        use_planner=use_planner,
        coder_ensemble_size=ensemble_size,
        coder_ensemble_temperature=actors.coder.ensemble_temperature,
        render_from_object=settings.pipeline.render_from_object,
        refinement_enabled=settings.pipeline.refinement_enabled,
        skip_render=skip_render,
        max_iter=policy.max_iter,
        score_threshold=policy.score_threshold,
        task_deadline_s=policy.task_deadline_s,
        planner_limit=prepare_limit,
        coder_limit=actors.coder.workers,
        js_checker_limit=actors.checker.workers,
        renderer_limit=actors.renderer.workers,
        critic_limit=actors.critic.workers,
        judge_limit=actors.judge.workers,
    )"""
        if old in text:
            factory.write_text(text.replace(old, new, 1), encoding="utf-8")
            print("  [patch] factory.prepare_limit+skip_render")
        else:
            print("  [warn] factory: pattern not found", file=sys.stderr)

    orch = ps / "pipeline" / "orchestrator.py"
    otext = orch.read_text(encoding="utf-8")
    if "self.skip_render = skip_render" in otext:
        print("  [ok] orchestrator already patched")
    else:
        otext2 = otext.replace(
            "        refinement_enabled: bool = True,\n        planner_limit: int = 2,",
            "        refinement_enabled: bool = True,\n"
            "        skip_render: bool = False,\n"
            "        planner_limit: int = 2,",
            1,
        )
        otext2 = otext2.replace(
            "        self.refinement_enabled = refinement_enabled\n\n        self.max_iter",
            "        self.refinement_enabled = refinement_enabled\n"
            "        self.skip_render = skip_render\n\n        self.max_iter",
            1,
        )
        otext2 = otext2.replace(
            """                # Renderer
                await renderer_stage(
                    task,
                    renderer=self.renderer,
                    sem=self._sem["renderer"],
                    status=self.task_status,
                )

            if not self.refinement_enabled:""",
            """                if not self.skip_render:
                    await renderer_stage(
                        task,
                        renderer=self.renderer,
                        sem=self._sem["renderer"],
                        status=self.task_status,
                    )

            if not self.refinement_enabled:""",
            1,
        )
        if otext2 != otext:
            orch.write_text(otext2, encoding="utf-8")
            print("  [patch] orchestrator.skip_render")
        else:
            print("  [warn] orchestrator: could not apply", file=sys.stderr)

    gen = ps / "pipeline" / "generation_pipeline.py"
    gtext = gen.read_text(encoding="utf-8")
    if "skip_render=true — Chromium renderer not started" in gtext:
        print("  [ok] generation_pipeline already patched")
    else:
        gtext2 = gtext.replace(
            "        await self.js_checker.startup()\n        await self.renderer.startup()\n",
            """        await self.js_checker.startup()
        skip_render = bool(self.settings.pipeline.skip_render) and (
            self.settings.actors.coder.ensemble_size <= 1
        )
        if skip_render:
            logger.info(
                "[Pipeline] skip_render=true — Chromium renderer not started "
                "(JS-only collect; render during duel scoring)"
            )
        else:
            await self.renderer.startup()
""",
            1,
        )
        gtext2 = gtext2.replace(
            "            logger.info(\"Judge: DISABLED (ensemble_size=1)\")\n"
            "        logger.info(f\"Iter cap: {eb.max_iter}",
            "            logger.info(\"Judge: DISABLED (ensemble_size=1)\")\n"
            "        logger.info(\n"
            "            f\"skip_render={skip_render} | refinement={self.settings.pipeline.refinement_enabled}\"\n"
            "        )\n"
            "        logger.info(f\"Iter cap: {eb.max_iter}",
            1,
        )
        gtext2 = gtext2.replace(
            "        if self._pipeline is not None:\n"
            "            await self._pipeline.stop()\n"
            "        await self.renderer.shutdown()\n"
            "        await self.js_checker.shutdown()\n",
            "        if self._pipeline is not None:\n"
            "            await self._pipeline.stop()\n"
            "        skip_render = bool(self.settings.pipeline.skip_render) and (\n"
            "            self.settings.actors.coder.ensemble_size <= 1\n"
            "        )\n"
            "        if not skip_render:\n"
            "            await self.renderer.shutdown()\n"
            "        await self.js_checker.shutdown()\n",
            1,
        )
        if gtext2 != gtext:
            gen.write_text(gtext2, encoding="utf-8")
            print("  [patch] generation_pipeline.skip_render_startup")
        else:
            print("  [warn] generation_pipeline: could not apply", file=sys.stderr)

    checker = ps / "modules" / "js_checker" / "module.py"
    ctext = checker.read_text(encoding="utf-8")
    if "Clear any prior failed flag from earlier repair attempts" in ctext:
        print("  [ok] js_checker failed-clear already patched")
    else:
        old_c = """        else:
            m = task.js_metrics or {}
            bbox = m.get("bbox") or {}"""
        new_c = """        else:
            # Clear any prior failed flag from earlier repair attempts so
            # skip_render / no-refine paths still record successful JS.
            task.failed = False
            task.failure_reason = None
            m = task.js_metrics or {}
            bbox = m.get("bbox") or {}"""
        if old_c in ctext:
            checker.write_text(ctext.replace(old_c, new_c, 1), encoding="utf-8")
            print("  [patch] js_checker.clear_failed_on_pass")
        else:
            print("  [warn] js_checker: pattern not found", file=sys.stderr)

    orch2 = ps / "pipeline" / "orchestrator.py"
    o2 = orch2.read_text(encoding="utf-8")
    if "JS-only path: checker already passed" in o2:
        print("  [ok] orchestrator skip_render success flags already patched")
    else:
        old_o = """                if not self.skip_render:
                    await renderer_stage(
                        task,
                        renderer=self.renderer,
                        sem=self._sem["renderer"],
                        status=self.task_status,
                    )

            if not self.refinement_enabled:"""
        new_o = """                if not self.skip_render:
                    await renderer_stage(
                        task,
                        renderer=self.renderer,
                        sem=self._sem["renderer"],
                        status=self.task_status,
                    )
                elif task.js_valid and task.js_code:
                    # JS-only path: checker already passed; ensure success flags.
                    task.failed = False
                    task.failure_reason = None

            if not self.refinement_enabled:"""
        if old_o in o2:
            orch2.write_text(o2.replace(old_o, new_o, 1), encoding="utf-8")
            print("  [patch] orchestrator.skip_render_success_flags")
        else:
            print("  [warn] orchestrator success-flags: pattern not found", file=sys.stderr)

    print("==> Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
