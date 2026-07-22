#!/usr/bin/env python3
"""Validator-shaped scalar rewards for GRPO.

Phases (controlled by env / RewardConfig):
  cheap   — validate.js + format heuristics (always on)
  render  — optional HTTP call to render-service-js / pipeline renderer
  s1      — optional OpenAI-compatible VLM front-match penalty → s1_score

Hard fails always dominate successful renders so GRPO learns reliability first.

Usage as library:
  from reward import make_reward_fn
  reward_fn = make_reward_fn(RewardConfig(...))

CLI smoke test:
  python reward.py --js path/to/stem.js --image path/to/stem.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from paths import validate_js_cli

R_FAIL_VALID = -2.0
R_FAIL_RENDER = -1.5

EXPORT_RE = re.compile(r"export\s+default\s+function\s+generate\s*\(\s*THREE\s*\)")


@dataclass
class RewardConfig:
    mode: str = "cheap"  # cheap | render | s1
    validate_cli: Path = field(default_factory=validate_js_cli)
    node_bin: str = "node"
    w_s1: float = 0.45
    w_dino: float = 0.15
    w_s4: float = 0.15
    w_fmt: float = 0.05
    w_group: float = 0.0  # set >0 when wiring within-group duels
    cache_dir: Path | None = None
    render_url: str | None = None  # e.g. http://127.0.0.1:3000/render
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key_env: str = "OPENROUTER_API_KEY"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_bonus(js: str) -> float:
    score = 0.0
    if EXPORT_RE.search(js):
        score += 0.5
    if "fitToUnitCube" in js or "0.95" in js:
        score += 0.25
    if "Math.random" in js or "Date.now" in js:
        score -= 0.5
    if "import " in js or "require(" in js:
        score -= 0.5
    return max(0.0, min(1.0, score))


def validate_js(js: str, cfg: RewardConfig) -> bool:
    with tempfile.TemporaryDirectory(prefix="grpo-val-") as td:
        path = Path(td) / "cand.js"
        path.write_text(js if js.endswith("\n") else js + "\n", encoding="utf-8")
        try:
            proc = subprocess.run(
                [cfg.node_bin, str(cfg.validate_cli), "--json", str(path)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            return False
        return bool(payload.get("passed"))


def cache_get(cfg: RewardConfig, key: str) -> float | None:
    if cfg.cache_dir is None:
        return None
    p = cfg.cache_dir / f"{key}.json"
    if not p.is_file():
        return None
    try:
        return float(json.loads(p.read_text(encoding="utf-8"))["reward"])
    except Exception:  # noqa: BLE001
        return None


def cache_put(cfg: RewardConfig, key: str, reward: float, extra: dict | None = None) -> None:
    if cfg.cache_dir is None:
        return
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"reward": reward, **(extra or {})}
    (cfg.cache_dir / f"{key}.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def try_render_ok(js: str, cfg: RewardConfig) -> bool | None:
    """Return True/False if render_url configured, else None (skip)."""
    if not cfg.render_url:
        return None
    try:
        import httpx

        with httpx.Client(timeout=120.0) as client:
            r = client.post(cfg.render_url, json={"code": js})
            if r.status_code >= 400:
                return False
            data = r.json()
            return bool(data.get("ok") or data.get("success") or data.get("views"))
    except Exception:  # noqa: BLE001
        return False


def try_s1_score(js: str, image_path: str | None, cfg: RewardConfig) -> float | None:
    """Cheap proxy: ask a VLM for a 0-10 front-match penalty, convert to s1_score.

    Production uses AB/BA on 4 angles; this is a single-image proxy for Phase-2 GRPO.
    Self-host GLM-4.6V-Flash for final runs.
    """
    if cfg.mode not in {"s1", "full"} or not cfg.judge_base_url or not cfg.judge_model:
        return None
    if not image_path or not Path(image_path).is_file():
        return None
    try:
        import base64

        import httpx

        api_key = os.environ.get(cfg.judge_api_key_env) or os.environ.get("OPENAI_API_KEY") or "local"
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        # Truncate JS for judge context — we only need structural summary signal.
        js_snip = js[:6000]
        prompt = (
            "You judge how well a Three.js generate(THREE) module would match a "
            "reference object image in a FRONT view. Reply with ONLY a JSON object "
            '{"penalty": <float 0-10>} where 0=perfect match, 10=completely wrong. '
            "Consider silhouette, landmarks, proportions, and color.\n\n"
            f"JS (truncated):\n{js_snip}"
        )
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": cfg.judge_model,
            "temperature": 0.0,
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        with httpx.Client(timeout=120.0, headers=headers) as client:
            r = client.post(f"{cfg.judge_base_url.rstrip('/')}/chat/completions", json=payload)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            if isinstance(text, list):
                text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
            m = re.search(r"\{[^}]*\}", str(text))
            if not m:
                return None
            penalty = float(json.loads(m.group(0))["penalty"])
            return max(0.0, min(1.0, 1.0 - penalty / 10.0))
    except Exception:  # noqa: BLE001
        return None


def score_one(js: str, cfg: RewardConfig, image_path: str | None = None) -> float:
    key = _sha(js + "|" + (image_path or "") + "|" + cfg.mode)
    cached = cache_get(cfg, key)
    if cached is not None:
        return cached

    if not validate_js(js, cfg):
        cache_put(cfg, key, R_FAIL_VALID, {"reason": "validate_fail"})
        return R_FAIL_VALID

    render = try_render_ok(js, cfg)
    if render is False:
        cache_put(cfg, key, R_FAIL_RENDER, {"reason": "render_fail"})
        return R_FAIL_RENDER

    fmt = format_bonus(js)
    s1 = try_s1_score(js, image_path, cfg)
    # Baseline success reward so any valid JS beats fails.
    base = 0.35
    r = base + cfg.w_fmt * fmt
    if s1 is not None:
        r += cfg.w_s1 * s1
    else:
        # Without S1, lean on format so groups still have variance.
        r += 0.2 * fmt

    if render is True:
        r += 0.1

    cache_put(cfg, key, r, {"s1": s1, "fmt": fmt, "render": render})
    return float(r)


def make_reward_fn(cfg: RewardConfig):
    """Return a TRL-compatible reward function.

    TRL GRPO calls: reward_fn(prompts, completions, **kwargs) -> list[float]
    For VLMs, kwargs may include `images` depending on version — we also accept
    a stem→image map via closure if completions carry stem metadata.
    """

    def reward_fn(completions, **kwargs):  # noqa: ANN003
        images = kwargs.get("images") or kwargs.get("image") or [None] * len(completions)
        # Flatten possible nested structures from chat completions
        flat: list[str] = []
        for c in completions:
            if isinstance(c, list):
                # [{role, content}] chat form
                text = ""
                for turn in c:
                    if isinstance(turn, dict) and turn.get("role") == "assistant":
                        content = turn.get("content", "")
                        if isinstance(content, list):
                            text = "".join(
                                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                            )
                        else:
                            text = str(content)
                flat.append(text)
            else:
                flat.append(str(c))

        rewards: list[float] = []
        for i, js in enumerate(flat):
            img = None
            if images is not None and i < len(images):
                item = images[i]
                if isinstance(item, (list, tuple)) and item:
                    item = item[0]
                if isinstance(item, str):
                    img = item
                elif hasattr(item, "filename"):
                    img = getattr(item, "filename", None)
            rewards.append(score_one(js, cfg, img))
        return rewards

    return reward_fn


def main() -> None:
    ap = argparse.ArgumentParser(description="Score one JS file with validator reward")
    ap.add_argument("--js", type=Path, required=True)
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--mode", type=str, default="cheap", choices=["cheap", "render", "s1"])
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = RewardConfig(
        mode=args.mode,
        cache_dir=args.cache_dir,
        render_url=os.environ.get("RENDER_URL"),
        judge_base_url=os.environ.get("JUDGE_BASE_URL"),
        judge_model=os.environ.get("JUDGE_MODEL"),
    )
    js = args.js.read_text(encoding="utf-8")
    r = score_one(js, cfg, str(args.image) if args.image else None)
    print(json.dumps({"file": str(args.js), "reward": r, "mode": cfg.mode}, indent=2))


if __name__ == "__main__":
    main()
