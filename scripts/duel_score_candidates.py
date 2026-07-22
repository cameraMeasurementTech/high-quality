#!/usr/bin/env python3
"""Score two JS candidates per stem with production multiview duel logic (S1–S4).

Uses shiny-guide pipeline modules:
  validate.js gate → multiview render → DINO embeddings → OpenRouter judge (AB+BA)

Writes JSON for DPO packing (--source duel-scored in pack_dpo_dataset.py).

Prereqs:
  - Node >= 20, Chromium sidecars (INSTALL_SYSTEM=1)
  - OPENROUTER_API_KEY
  - CONFIG_FILE=pipeline/configuration.duel-judge.yaml
  - PYTHONPATH includes shiny-guide/pipeline_service (set by run script)

Usage:
  python duel_score_candidates.py \\
    --candidates-dir data/candidates/shiny_k2 \\
    --images data/images \\
    --list data/splits/train.txt \\
    --out data/duel_scores/candidate_duels.json \\
    --limit 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from paths import shiny_guide_root, validate_js_cli
from reward import RewardConfig, validate_js


def _setup_pipeline_imports() -> None:
    ps = shiny_guide_root() / "pipeline_service"
    if not ps.is_dir():
        raise FileNotFoundError(f"shiny-guide pipeline_service not found: {ps}")
    ps_str = str(ps.resolve())
    if ps_str not in sys.path:
        sys.path.insert(0, ps_str)


def load_stem_list(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    stems: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stems.add(line.split(None, 1)[0])
    return stems


def find_candidate_pair(stem_dir: Path) -> tuple[Path, Path] | None:
    """Return (sample_a, sample_b) — prefers sample_0 + sample_1."""
    a = stem_dir / "sample_0.js"
    b = stem_dir / "sample_1.js"
    if a.is_file() and b.is_file():
        return a, b
    samples = sorted(stem_dir.glob("sample_*.js"))
    if len(samples) >= 2:
        return samples[0], samples[1]
    return None


def mime_for(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/png")


def combine_verdicts(ab: str, ba: str) -> str:
    ba_norm = "B" if ba == "A" else "A"
    if ab == ba_norm:
        return ab
    return "draw"


def verdict_to_dict(v) -> dict:
    if hasattr(v, "model_dump"):
        return v.model_dump()
    return {
        "winner": getattr(v, "winner", None),
        "reason": getattr(v, "reason", ""),
        "confidence": getattr(v, "confidence", None),
        "decided_by": getattr(v, "decided_by", ""),
        "detail": getattr(v, "detail", {}) or {},
    }


async def duel_one(judge, stem, ref_bytes, ref_mime, side_a, side_b, max_stage):
    if not side_a.render_ok and not side_b.render_ok:
        return {"stem": stem, "winner": "draw", "reason": "both failed to render"}
    if not side_a.render_ok:
        return {"stem": stem, "winner": "B", "reason": "A failed to render"}
    if not side_b.render_ok:
        return {"stem": stem, "winner": "A", "reason": "B failed to render"}

    v_ab = await judge.compare(
        task_id=stem,
        match_label="AB",
        reference_bytes=ref_bytes,
        reference_mime=ref_mime,
        render_a=side_a.grid,
        render_b=side_b.grid,
        white_views_a=side_a.white,
        white_views_b=side_b.white,
        gray_views_a=side_a.gray,
        gray_views_b=side_b.gray,
        embeddings_a=side_a.embeddings,
        embeddings_b=side_b.embeddings,
    )
    v_ba = await judge.compare(
        task_id=stem,
        match_label="BA",
        reference_bytes=ref_bytes,
        reference_mime=ref_mime,
        render_a=side_b.grid,
        render_b=side_a.grid,
        white_views_a=side_b.white,
        white_views_b=side_a.white,
        gray_views_a=side_b.gray,
        gray_views_b=side_a.gray,
        embeddings_a=side_b.embeddings,
        embeddings_b=side_a.embeddings,
    )
    winner = combine_verdicts(v_ab.winner, v_ba.winner)
    return {
        "stem": stem,
        "winner": winner,
        "ab_winner": v_ab.winner,
        "ba_winner": v_ba.winner,
        "ab_decided_by": v_ab.decided_by,
        "ba_decided_by": v_ba.decided_by,
        "ab_reason": v_ab.reason,
        "ba_reason": v_ba.reason,
        "ab_confidence": v_ab.confidence,
        "ba_confidence": v_ba.confidence,
        "position_bias": winner == "draw",
        "ab": verdict_to_dict(v_ab),
        "ba": verdict_to_dict(v_ba),
    }


async def run(args) -> int:
    _setup_pipeline_imports()
    from openai import AsyncOpenAI

    from config.settings import settings
    from modules.judge.agent import JudgeAgent
    from modules.judge.dino import DinoEmbedder
    from modules.renderer.module import RendererModule
    from modules.renderer.settings import RendererConfig
    from pipeline.task import PipelineTask

    candidates_dir = Path(args.candidates_dir)
    images_dir = Path(args.images)
    allow = load_stem_list(Path(args.list) if args.list else None)

    val_cfg = RewardConfig(validate_cli=validate_js_cli())

    stem_dirs = sorted(d for d in candidates_dir.iterdir() if d.is_dir())
    if allow is not None:
        stem_dirs = [d for d in stem_dirs if d.name in allow]
    if args.limit:
        stem_dirs = stem_dirs[: args.limit]

    if not stem_dirs:
        print("No candidate stems found.", file=sys.stderr)
        return 1

    client_cfg = settings.llm_clients.get("openrouter")
    if client_cfg is None:
        print("ERROR: openrouter client missing in CONFIG_FILE", file=sys.stderr)
        return 2
    api_key = os.environ.get(client_cfg.api_key_env or "OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    client = AsyncOpenAI(base_url=client_cfg.base_url, api_key=api_key)
    judge = JudgeAgent(client, settings=settings.actors.judge, max_stage=args.max_stage)

    embedder = None
    if not args.no_embedder and settings.embedder.enabled:
        embedder = DinoEmbedder(settings.embedder)

    base_renderer = settings.renderer
    rcfg = RendererConfig(**{**base_renderer.model_dump()})
    rcfg.sidecar_port = args.sidecar_port
    rcfg.static_port_base = args.static_port_base
    rcfg.sidecar_count = args.sidecar_count
    rcfg.judge_multiview = True
    renderer = RendererModule(rcfg)

    async def render_grid(stem: str, js_code: str) -> bytes | None:
        task = PipelineTask(stem=stem, image_url="")
        task.js_code = js_code
        await renderer.process(task)
        return task.rendered_png

    class SideAssets:
        __slots__ = ("grid", "white", "gray", "embeddings", "render_ok")

        def __init__(self) -> None:
            self.grid = None
            self.white: dict[str, bytes] = {}
            self.gray: dict[str, bytes] = {}
            self.embeddings = None
            self.render_ok = False

    async def build_side(ref_vec, stem: str, js_code: str) -> SideAssets:
        s = SideAssets()
        s.grid = await render_grid(stem, js_code)
        s.render_ok = s.grid is not None
        if not s.render_ok:
            return s
        s.white, s.gray = await renderer.render_judge_views(stem, js_code)
        if embedder is not None and ref_vec is not None and s.white:
            try:
                s.embeddings = await embedder.build_candidate_npz(ref_vec, s.white)
            except Exception as exc:  # noqa: BLE001
                print(f"warn embed {stem[:12]}: {exc}", file=sys.stderr)
        return s

    print(f"Scoring {len(stem_dirs)} stems | judge={settings.actors.judge.model} | max_stage={args.max_stage}")
    await renderer.startup()

    records: list[dict] = []
    skipped = {"no_pair": 0, "no_img": 0, "both_invalid": 0, "draw": 0}

    try:
        for stem_dir in tqdm(stem_dirs, desc="duel-score"):
            stem = stem_dir.name
            pair = find_candidate_pair(stem_dir)
            if pair is None:
                skipped["no_pair"] += 1
                continue
            js_a_path, js_b_path = pair
            img_path = images_dir / f"{stem}.png"
            if not img_path.is_file():
                skipped["no_img"] += 1
                continue

            js_a = js_a_path.read_text(encoding="utf-8")
            js_b = js_b_path.read_text(encoding="utf-8")

            if args.require_validate:
                ok_a = validate_js(js_a, val_cfg)
                ok_b = validate_js(js_b, val_cfg)
                if not ok_a and not ok_b:
                    skipped["both_invalid"] += 1
                    continue

            ref_bytes = img_path.read_bytes()
            ref_mime = mime_for(img_path)

            ref_vec = None
            if embedder is not None:
                try:
                    ref_vec = await embedder.embed_reference(ref_bytes)
                except Exception as exc:  # noqa: BLE001
                    print(f"warn ref embed {stem[:12]}: {exc}", file=sys.stderr)

            side_a = await build_side(ref_vec, stem, js_a)
            side_b = await build_side(ref_vec, stem, js_b)

            rec = await duel_one(
                judge, stem, ref_bytes, ref_mime, side_a, side_b, args.max_stage
            )
            rec["candidate_a"] = str(js_a_path.resolve())
            rec["candidate_b"] = str(js_b_path.resolve())
            rec["reference_image"] = str(img_path.resolve())

            if rec["winner"] == "A":
                rec["chosen_file"] = rec["candidate_a"]
                rec["rejected_file"] = rec["candidate_b"]
            elif rec["winner"] == "B":
                rec["chosen_file"] = rec["candidate_b"]
                rec["rejected_file"] = rec["candidate_a"]
            else:
                skipped["draw"] += 1

            records.append(rec)
    finally:
        await renderer.shutdown()
        await client.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidates_dir": str(candidates_dir.resolve()),
        "images_dir": str(images_dir.resolve()),
        "judge_model": settings.actors.judge.model,
        "max_stage": args.max_stage,
        "skipped": skipped,
        "n_scored": len(records),
        "n_pairs_for_dpo": sum(1 for r in records if r.get("chosen_file")),
        "records": records,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} ({payload['n_pairs_for_dpo']} decisive pairs for DPO)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Multiview duel score for DPO candidate pairs")
    ap.add_argument("--candidates-dir", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-stage", type=int, default=4)
    ap.add_argument("--sidecar-port", type=int, default=8013)
    ap.add_argument("--static-port-base", type=int, default=13100)
    ap.add_argument("--sidecar-count", type=int, default=2)
    ap.add_argument("--no-embedder", action="store_true")
    ap.add_argument("--require-validate", action="store_true", default=True)
    ap.add_argument("--no-require-validate", action="store_false", dest="require_validate")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
