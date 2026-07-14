from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Stage-relevant pipeline messages (skip renderer-sidecar ping noise by default).
LOG_FILTER_RE = re.compile(
    r"Batch (starting|done|retry|deadline|cleanup)"
    r"|MULTIGEN|\[JS_CHECK\]|\[RENDERER\].*(start|PASS|FAIL|judge views)"
    r"|\[2/7 Coder\]|\[5/7 Critic\]|\[Coder Repair\]|\[Judge "
    r"|\[BRACKET |\[TOKENS Actor"
    r"|\[TASK COMPLETED\]|\[FAIL\]|StageError"
    r"|Pipeline initialized|models up|GenerationPipeline starting"
    r"|\[DINO\]|Warmup",
    re.IGNORECASE,
)


def log_ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET}  {msg}")


def log_fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")


def log_info(msg: str) -> None:
    print(f"  {CYAN}--{RESET}  {msg}")


def log_step(msg: str) -> None:
    print(f"\n{BOLD}[step]{RESET} {msg}")


def log_pipeline(msg: str) -> None:
    """Print a live pipeline log line (clears the progress bar line first)."""
    sys.stdout.write("\r\033[K")
    print(f"  {DIM}[pipeline]{RESET} {msg}")


def format_pipeline_log_line(raw: str) -> str:
    """Collapse loguru file format to timestamp + message."""
    line = raw.rstrip()
    parts = line.split(" | ", 3)
    if len(parts) >= 4:
        return f"{parts[0]} | {parts[3]}"
    return line


def should_print_log_line(line: str, *, full_logs: bool) -> bool:
    if not line.strip():
        return False
    if full_logs:
        return True
    if "renderer-sidecar" in line and "DEBUG" in line:
        return False
    return LOG_FILTER_RE.search(line) is not None


class PipelineLogFollower:
    """Stream new lines from GET /debug/logs while the batch runs."""

    def __init__(self, client: httpx.Client, *, full_logs: bool = False) -> None:
        self.client = client
        self.full_logs = full_logs
        self._offset = 0
        self._available = True

    def poll(self) -> None:
        if not self._available:
            return
        try:
            r = self.client.get("/debug/logs", timeout=10.0)
        except httpx.HTTPError:
            return
        if r.status_code == 404:
            self._available = False
            log_info("pipeline log not available yet (/debug/logs)")
            return
        if r.status_code != 200:
            return

        text = r.text
        if self._offset > len(text):
            self._offset = 0
        chunk = text[self._offset :]
        self._offset = len(text)
        if not chunk:
            return

        for raw in chunk.splitlines():
            if not should_print_log_line(raw, full_logs=self.full_logs):
                continue
            log_pipeline(format_pipeline_log_line(raw))

    def fetch_all(self) -> str | None:
        try:
            r = self.client.get("/debug/logs", timeout=30.0)
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            return None
        return r.text


class TaskStatusFollower:
    """Print per-prompt stage transitions from GET /debug/tasks."""

    def __init__(self, client: httpx.Client, stems: list[str]) -> None:
        self.client = client
        self.stems = stems
        self._last: dict[str, str] = {}

    def poll(self) -> None:
        try:
            r = self.client.get("/debug/tasks", timeout=10.0)
        except httpx.HTTPError:
            return
        if r.status_code != 200:
            return
        for task in r.json().get("tasks", []):
            stem = task.get("stem")
            if stem not in self.stems:
                continue
            status = task.get("status", "unknown")
            prev = self._last.get(stem)
            if prev == status:
                continue
            self._last[stem] = status
            if prev is None:
                log_pipeline(f"{stem[:16]}… status={status}")
            else:
                log_pipeline(f"{stem[:16]}… {prev} -> {status}")
            if task.get("failed") and task.get("failure_reason"):
                log_pipeline(f"{stem[:16]}… FAIL: {task['failure_reason'][:160]}")
            elif task.get("refinement", {}).get("best_score", -1) >= 0:
                ref = task["refinement"]
                log_pipeline(
                    f"{stem[:16]}… score={ref.get('best_score', -1):.2f} "
                    f"iter={ref.get('best_iter', -1)}"
                )


def parse_prompts(path: str) -> list[dict]:
    prompts = []
    p = Path(path)
    if not p.exists():
        print(f"{RED}Prompts file not found: {path}{RESET}")
        sys.exit(1)
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 1:
            if line.startswith("http://") or line.startswith("https://"):
                stem = Path(urlparse(line).path).stem
                if stem:
                    prompts.append({"stem": stem, "image_url": line})
                else:
                    print(f"{YELLOW}Skipping URL without stem: {line}{RESET}")
                continue
            print(f"{YELLOW}Skipping malformed line: {line}{RESET}")
            continue
        if len(parts) < 2:
            print(f"{YELLOW}Skipping malformed line: {line}{RESET}")
            continue
        prompts.append({"stem": parts[0], "image_url": parts[1]})
    return prompts


def wait_health(client: httpx.Client, max_wait: int = 30) -> bool:
    log_step("Health check")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = client.get("/health")
            if r.status_code == 200:
                log_ok(f"GET /health -> 200")
                return True
        except httpx.ConnectError:
            pass
        time.sleep(1)
    log_fail(f"Server not reachable after {max_wait}s")
    return False


def wait_ready(client: httpx.Client, max_wait: int = 3600) -> bool:
    log_step("Waiting for ready status")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        r = client.get("/status")
        data = r.json()
        status = data["status"]
        if status == "warming_up":
            log_info(f"status={status}, waiting...")
            time.sleep(2)
            continue
        log_ok(f"GET /status -> {status}")
        return True
    log_fail(f"Still warming up after {max_wait}s")
    return False


def submit_batch(client: httpx.Client, prompts: list[dict], seed: int) -> bool:
    log_step(f"Submitting batch: {len(prompts)} prompts, seed={seed}")
    payload = {"prompts": prompts, "seed": seed}
    r = client.post("/generate", json=payload)
    if r.status_code == 200:
        data = r.json()
        log_ok(f"POST /generate -> accepted={data['accepted']}")
        return True
    log_fail(f"POST /generate -> {r.status_code}: {r.text}")
    return False


def poll_until_complete(
    client: httpx.Client,
    max_wait: int = 900,
    *,
    follow_logs: bool = False,
    full_logs: bool = False,
    stems: list[str] | None = None,
) -> dict | None:
    log_step("Polling status" + (" (live pipeline logs below)" if follow_logs else ""))
    deadline = time.time() + max_wait
    last_progress = -1
    data: dict = {}
    log_follower = PipelineLogFollower(client, full_logs=full_logs) if follow_logs else None
    task_follower = TaskStatusFollower(client, stems or []) if follow_logs and stems else None

    while time.time() < deadline:
        if log_follower is not None:
            log_follower.poll()
        if task_follower is not None:
            task_follower.poll()

        r = client.get("/status")
        data = r.json()
        status = data["status"]
        progress = data["progress"]
        total = data["total"]

        if progress != last_progress:
            bar_width = 30
            filled = int(bar_width * progress / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_width - filled)
            sys.stdout.write(f"\r  [{bar}] {progress}/{total} {status}    ")
            sys.stdout.flush()
            last_progress = progress

        if status == "complete":
            if log_follower is not None:
                log_follower.poll()
            print()
            log_ok(f"Complete: {progress}/{total}")
            return data

        time.sleep(1)

    print()
    log_fail(f"Timeout after {max_wait}s (status={data.get('status')}, {data.get('progress')}/{data.get('total')})")
    return None


def save_pipeline_log(client: httpx.Client, results_dir: Path) -> None:
    follower = PipelineLogFollower(client, full_logs=True)
    text = follower.fetch_all()
    if not text:
        log_info("pipeline.log not saved (server log unavailable)")
        return
    path = results_dir / "pipeline.log"
    path.write_text(text, encoding="utf-8")
    log_ok(f"pipeline.log saved ({path.stat().st_size:,} bytes)")


def save_results(client: httpx.Client, results_dir: Path, stems: list[str]) -> list[dict]:
    log_step(f"Saving results to {results_dir}/")
    results_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for stem in stems:
        r = client.get(f"/debug/tasks/{stem}")
        if r.status_code != 200:
            log_fail(f"{stem} -> HTTP {r.status_code}")
            records.append({"stem": stem, "error": f"HTTP {r.status_code}"})
            continue

        data = r.json()

        if data.get("js_code"):
            (results_dir / f"{stem}.js").write_text(data["js_code"], encoding="utf-8")

        if data.get("rendered_png_b64"):
            png_bytes = base64.b64decode(data["rendered_png_b64"])
            (results_dir / f"{stem}.png").write_bytes(png_bytes)

        if data.get("refinement_rendered_pngs_b64"):
            for i, png_b64 in enumerate(data["refinement_rendered_pngs_b64"]):
                png_bytes = base64.b64decode(png_b64)
                (results_dir / f"{stem}_refinement_{i+1}.png").write_bytes(png_bytes)
        if data.get("osd"):
            (results_dir / f"{stem}_osd.json").write_text(data["osd"], encoding="utf-8")

        input_saved = False
        image_url = data.get("image_url")
        if image_url:
            ext = Path(urlparse(image_url).path).suffix or ".png"
            try:
                img_resp = client.get(image_url, timeout=30.0, follow_redirects=True)
                if img_resp.status_code == 200:
                    (results_dir / f"{stem}_input{ext}").write_bytes(img_resp.content)
                    input_saved = True
                else:
                    log_fail(f"{stem} input image -> HTTP {img_resp.status_code}")
            except Exception as exc:
                log_fail(f"{stem} input image -> {type(exc).__name__}: {exc}")

        record = {k: v for k, v in data.items() if k not in ("js_code", "rendered_png_b64")}
        record["js_saved"] = bool(data.get("js_code"))
        record["png_saved"] = bool(data.get("rendered_png_b64"))
        record["osd_saved"] = bool(data.get("osd"))
        record["input_saved"] = input_saved
        records.append(record)

    (results_dir / "results.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_ok(f"results.json written ({len(records)} records)")
    return records


def show_results(records: list[dict], results_dir: Path, elapsed: float) -> None:
    print(f"\n{BOLD}{'=' * 50}")
    print(f" Results")
    print(f"{'=' * 50}{RESET}\n")

    ok_count = 0
    failed_count = 0
    png_count = 0
    input_count = 0

    for rec in sorted(records, key=lambda r: r.get("stem", "")):
        stem = rec.get("stem", "?")

        if rec.get("input_saved"):
            input_count += 1

        if "error" in rec:
            log_fail(f"{stem} -> {rec['error']}")
            failed_count += 1
            continue

        if rec.get("failed"):
            reason = rec.get("failure_reason") or "unknown"
            log_fail(f"{stem}.js -> {reason}")
            failed_count += 1
            continue

        ok_count += 1
        parts = [f"{stem}.js"]

        js_path = results_dir / f"{stem}.js"
        if js_path.exists():
            size = js_path.stat().st_size
            lines = len(js_path.read_text(encoding="utf-8").splitlines())
            parts.append(f"{size:,} bytes, {lines} lines")

        if rec.get("png_saved"):
            png_count += 1
            png_path = results_dir / f"{stem}.png"
            if png_path.exists():
                parts.append(f"PNG {png_path.stat().st_size:,} bytes")
            else:
                parts.append("PNG saved")

        if rec.get("osd_saved"):
            parts.append("OSD saved")

        if rec.get("input_saved"):
            parts.append("input saved")

        log_ok(" | ".join(parts))

    total = ok_count + failed_count
    print(f"\n{BOLD}{'=' * 50}")
    print(f" Summary")
    print(f"{'=' * 50}{RESET}\n")
    color = GREEN if ok_count == total else YELLOW if ok_count > 0 else RED
    print(f"  {color}{ok_count}/{total} passed{RESET}, {failed_count} failed, {png_count} with PNG, {input_count} with input")
    print(f"  Pipeline time: {elapsed:.1f}s")
    print(f"  Output dir: {results_dir.resolve()}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline smoke test")
    parser.add_argument("prompts_file", nargs="?", default="tests/prompts/test_prompts.txt")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=9000, help="max poll time in seconds")
    parser.add_argument("--limit", type=int, default=None, help="limit the number of prompts to test")
    parser.add_argument(
        "--name",
        default="test",
        help="output folder name (under --out-dir or the prompts file directory)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory to write run outputs (default: parent of prompts file)",
    )
    parser.add_argument(
        "--follow-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stream pipeline stage logs while waiting (default: on)",
    )
    parser.add_argument(
        "--full-logs",
        action="store_true",
        help="with --follow-logs, print all log lines (default: stage-relevant only)",
    )
    args = parser.parse_args()

    prompts = parse_prompts(args.prompts_file)
    if not prompts:
        print(f"{RED}No prompts found in {args.prompts_file}{RESET}")
        sys.exit(1)

    if args.limit:
        prompts = prompts[: args.limit]

    stems = [p["stem"] for p in prompts]
    if args.out_dir:
        results_dir = Path(args.out_dir) / args.name
    else:
        results_dir = Path(args.prompts_file).parent / args.name

    base_url = f"http://{args.host}:{args.port}"
    print(f"\n{BOLD}Pipeline Test{RESET}")
    print(f"  Target: {base_url}")
    print(f"  Prompts: {len(prompts)} from {args.prompts_file}")
    print(f"  Output: {results_dir}/ (--name={args.name})")
    print(f"  Seed: {args.seed}")
    if args.follow_logs:
        mode = "full" if args.full_logs else "stages only"
        print(f"  Logs: follow-logs={mode} -> also saved as pipeline.log")

    client = httpx.Client(base_url=base_url, timeout=30.0)

    try:
        if not wait_health(client):
            sys.exit(1)

        if not wait_ready(client):
            sys.exit(1)

        t_start = time.time()

        if not submit_batch(client, prompts, args.seed):
            sys.exit(1)

        if poll_until_complete(
            client,
            max_wait=args.timeout,
            follow_logs=args.follow_logs,
            full_logs=args.full_logs,
            stems=stems,
        ) is None:
            sys.exit(1)

        elapsed = time.time() - t_start

        results_dir.mkdir(parents=True, exist_ok=True)
        save_pipeline_log(client, results_dir)
        records = save_results(client, results_dir, stems)
        show_results(records, results_dir, elapsed)

    finally:
        client.close()


if __name__ == "__main__":
    main()
