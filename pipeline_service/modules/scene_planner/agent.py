from __future__ import annotations

import base64
import time
from typing import Any

from config.settings import ProviderRoutingConfig
from llm.session import SessionAgent
from llm.session_store import SessionStore
from logger_config import logger
from modules.base_agent import BaseAgent
from modules.scene_planner.prompts import PLANNER_OSD_PROMPT
from modules.scene_planner.schema import OSD
from config.settings import ActorConfig


_PLANNER_SYSTEM_PROMPT = (
    "You are Scene Planner in the Crucible3D procedural 3D pipeline.\n"
    "You look at one photograph and produce a structured Object Structural "
    "Description (OSD) so a Scene Coder can rebuild the object in Three.js "
    "part-by-part without guessing.\n\n"
    "Output: one markdown `scene_brief` (silhouette / proportions / materials / "
    "layout / landmark checklist / build order) plus a `parts` list. Each part "
    "MUST include: snake_case `name`, `narrative`, `count_hint`, `motif_role`, "
    "and the structured build fields `primitive`, `attach_to`, `size_frac`, "
    "`color_hex`, `material`.\n\n"
    "Inventory every visible landmark (handle, spout, brim, hole, fold, …) as "
    "its own part. Parent/child via `attach_to`. Do not invent hidden parts. "
    "Do not describe background, lighting, or camera.\n"
    "Return exactly one JSON OSD object — no prose, no markdown fences around "
    "the object (markdown INSIDE scene_brief is expected)."
)


class ScenePlannerAgent(BaseAgent):
    """Per-task Scene Planner."""

    actor = "planner"

    def __init__(
        self,
        clients: AsyncOpenAI,
        settings: ActorConfig,
        *,
        session_store: SessionStore,
        max_tool_iters: int = 4,
    ) -> None:
        super().__init__(clients, settings)
        self.session_store = session_store
        self.max_tool_iters = max_tool_iters
        self.reasoning_effort = settings.reasoning_effort

    def _build_session(self, task_id: str, actor: str) -> SessionAgent:
        return SessionAgent(
            task_id=task_id,
            actor=actor,
            system_prompt=_PLANNER_SYSTEM_PROMPT,
            model=self.model,
            max_tokens=self.max_tokens,
            tools=None,
            response_model=OSD,
            client=self.client,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
            max_tool_iters=self.max_tool_iters,
            reasoning_effort=self.reasoning_effort,
            enable_thinking=self.enable_thinking,
            backend=self.backend,
            providers=self.providers,
        )

    async def plan(
        self,
        task_id: str,
        *,
        image_bytes: bytes | None = None,
        image_url: str | None = None,
        mime: str = "image/jpeg",
    ) -> OSD:
        """Produce an OSD for the given image."""
        if not image_bytes and not image_url:
            raise ValueError("plan() requires image_bytes or image_url")

        session = self.session_store.get_or_create(
            task_id, self.actor, self._build_session,
        )
        if image_bytes is not None:
            b64 = base64.b64encode(image_bytes).decode()
            url_str = f"data:{mime};base64,{b64}"
        else:
            url_str = image_url  # type: ignore[assignment]

        content = [
            {"type": "text", "text": PLANNER_OSD_PROMPT},
            {"type": "image_url", "image_url": {"url": url_str}},
        ]
        img_kb = (len(image_bytes) / 1024.0) if image_bytes else 0.0
        logger.info(
            f"[1/7 Planner] Started Task {task_id} | Model: {self.model} | Image KB: {img_kb:.1f} | Source: {'bytes' if image_bytes else 'url'}"
        )
        t0 = time.time()
        osd = await session.run(content)
        dt = time.time() - t0
        logger.info(
            f"[1/7 Planner] Finished Task {task_id} | Elapsed: {dt:.1f}s | Object Type: {osd.object_type} | Parts: {len(osd.parts)}"
        )
        return osd
