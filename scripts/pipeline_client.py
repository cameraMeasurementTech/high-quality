"""HTTP client for shiny-guide / my-agent batch /generate API."""
from __future__ import annotations

import time
from typing import Any

import httpx


def wait_ready(client: httpx.Client, max_wait: int = 3600) -> bool:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = client.get("/status")
            r.raise_for_status()
            status = r.json().get("status")
            if status not in {"warming_up", "starting"}:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(2)
    return False


def submit_and_wait(
    base_url: str,
    prompts: list[dict[str, str]],
    *,
    seed: int = 42,
    temperature: float | None = None,
    timeout: float = 900.0,
) -> dict[str, str]:
    """Submit batch, poll until complete, return stem -> js_code."""
    base = base_url.rstrip("/")
    with httpx.Client(base_url=base, timeout=timeout) as client:
        if not wait_ready(client):
            raise RuntimeError(f"pipeline not ready: {base}")

        payload: dict[str, Any] = {"prompts": prompts, "seed": seed}
        if temperature is not None:
            payload["temperature"] = float(temperature)
        r = client.post("/generate", json=payload)
        r.raise_for_status()
        accepted = r.json().get("accepted", len(prompts))

        deadline = time.time() + timeout
        while time.time() < deadline:
            sr = client.get("/status")
            sr.raise_for_status()
            data = sr.json()
            if data.get("status") == "complete" and data.get("progress", 0) >= accepted:
                break
            time.sleep(1)
        else:
            raise TimeoutError(f"batch timed out after {timeout}s")

        out: dict[str, str] = {}
        for p in prompts:
            stem = p["stem"]
            tr = client.get(f"/debug/tasks/{stem}")
            if tr.status_code != 200:
                continue
            task = tr.json()
            js = task.get("js_code") or task.get("code") or task.get("js")
            if js:
                out[stem] = str(js)
        return out
