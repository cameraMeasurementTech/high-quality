from __future__ import annotations

from pydantic import BaseModel, Field


class OSDPart(BaseModel):
    """One observed part of the object."""

    name: str = Field(
        ...,
        description=(
            "Short stable identifier — 'seat', 'front_left_leg', 'lampshade'. "
            "Used by the Coder as a JS `const` name and by the Critic/Patcher as "
            "`target_node_id` for iterative refinement. Prefer snake_case."
        ),
    )
    narrative: str = Field(
        ...,
        min_length=80,
        description=(
            "Two to five sentences covering shape, approximate size, position "
            "relative to the whole, color (hex if confident, else NL), "
            "material (NL phrasing that maps to PBR), and any modifier cue "
            "(bend/twist/taper). Written for a 3D artist to implement."
        ),
    )
    count_hint: str = Field(
        default="one",
        description=(
            "'one', 'four symmetric', 'two mirrored pairs', 'a single row of "
            "six'. Critical for the Coder — drives the instanced_group "
            "decision, so keep it as a structured field rather than burying "
            "it in the narrative."
        ),
    )
    motif_role: str | None = Field(
        default=None,
        description=(
            "Role hint — 'support', 'body', 'enclosure', 'landmark', "
            "'decoration'. Helps the Coder group related nodes and prioritize "
            "landmarks (handle/spout/brim/hole) over trim."
        ),
    )
    # Structured build fields (optional for backward compatibility; planner
    # should fill them so the Coder can map image → Three.js without guessing).
    primitive: str | None = Field(
        default=None,
        description=(
            "Preferred Three.js primitive family: box, cylinder, sphere, cone, "
            "torus, lathe, tube, extrude, capsule, instanced."
        ),
    )
    attach_to: str | None = Field(
        default=None,
        description=(
            "Parent part `name` this attaches to, or 'root' for the main group. "
            "Coder must parent.add(child) — never leave parts floating."
        ),
    )
    size_frac: str | None = Field(
        default=None,
        description=(
            "Relative size vs whole object, e.g. '~0.35H', '~0.6W', '~0.2D'."
        ),
    )
    color_hex: str | None = Field(
        default=None,
        description="Dominant color as #rrggbb when confident; else null.",
    )
    material: str | None = Field(
        default=None,
        description=(
            "PBR family phrase: polished metal, brushed metal, glossy plastic, "
            "matte plastic, wood, ceramic, leather, fabric, clear glass, "
            "frosted glass, rubber, generic."
        ),
    )


class OSD(BaseModel):
    """Object Structural Description."""

    object_type: str = Field(
        ...,
        description="Short lowercase noun — 'chair', 'goblet', 'bottle', 'car', 'lamp'.",
    )
    scene_brief: str = Field(
        ...,
        min_length=300,
        description=(
            "Markdown-formatted prose describing the whole object as if "
            "briefing a 3D artist. Expected sections (as ## headings inside "
            "the string):\n"
            "  ## Overall silhouette\n"
            "  ## Proportions\n"
            "  ## Materials and color palette\n"
            "  ## Layout and symmetry\n"
            "Typical length 600–2000 tokens. Coder reads this as the primary "
            "context — it is NOT a short summary."
        ),
    )
    parts: list[OSDPart] = Field(
        ...,
        description="Flat list of observed parts with per-part narratives.",
    )
    motif_hint: str | None = Field(
        default=None,
        description=(
            "If the object matches a known motif template (chair/table/"
            "bottle/mug/lamp/car/...), name it here. None if no clean match."
        ),
    )
    notes: str | None = Field(
        default=None,
        description="Overall caveats — occlusion, ambiguity, style cues. None if irrelevant.",
    )
