"""Scene Planner prompt — produces OSD (Scene Brief + structured parts).

Asks the model for a markdown Scene Brief plus a per-part list with both
holistic narrative AND structured build fields (primitive, attach_to,
size_frac, color_hex, material) so the Coder can implement detailed 3D
structure without guessing.
"""
from __future__ import annotations


PLANNER_OSD_PROMPT = """\
You are looking at a single 3D object in a photograph. Produce a **structured
Object Structural Description (OSD)** that a Three.js coder can implement
part-by-part with procedural primitives (box, cylinder, sphere, cone, torus,
lathe profile, tube-along-path, extruded shape, capsule, instanced groups).

## Analysis protocol (do this in order before writing JSON)

1. **Object class** — name the object in one short noun (`chair`, `coffee_pot`,
   `folded_tag`, `calendar_stand`, `hat`, …).
2. **Silhouette read** — overall aspect (taller/wider/cubic), front face, and
   whether it is rotationally symmetric, bilateral, or asymmetric.
3. **Landmark inventory** — list every distinctive feature you can see:
   handle, spout, lid/knob, brim, crown pinch, hole/eyelet, fold/crease,
   spiral binding, clip, wheels, legs, strap, toppings, windows, etc.
   If a landmark is visible, it MUST become its own part.
4. **Part hierarchy** — decide parent/child attachment (body → handle, body →
   lid, crown → brim). Nothing floats in empty space.
5. **Implied occluded structure** — parts continue behind what is visible.
   One visible front leg + one back leg ⇒ four legs. A lamp base supporting
   a neck ⇒ base + neck + shade as separate parts.
6. **Materials / colors** — sample dominant hues (#rrggbb when confident) and
   PBR families per part.

Return a SINGLE JSON object with this exact shape — fill every field:

{{
  "object_type": "<short lowercase noun, e.g. chair, coffee_pot, hat, folded_tag>",
  "scene_brief": "<markdown document, 600–2000 tokens, following the sections below>",
  "parts": [
    {{
      "name": "<snake_case id, e.g. 'seat', 'front_left_leg', 'spout', 'brim'>",
      "narrative": "<2–5 sentences — see per-part guidance below>",
      "count_hint": "<'one', 'four symmetric', 'two mirrored pairs', 'a single row of six'>",
      "motif_role": "<'body'|'support'|'enclosure'|'landmark'|'decoration'|null>",
      "primitive": "<'box'|'cylinder'|'sphere'|'cone'|'torus'|'lathe'|'tube'|'extrude'|'capsule'|'instanced'>",
      "attach_to": "<'root' or parent part name>",
      "size_frac": "<'~0.35H' | '~0.6W' | '~0.2D' — relative to whole object>",
      "color_hex": "<'#rrggbb' or null>",
      "material": "<'polished metal'|'brushed metal'|'glossy plastic'|'matte plastic'|'wood'|'ceramic'|'leather'|'fabric'|'clear glass'|'frosted glass'|'rubber'|'generic'>"
    }}
  ],
  "motif_hint": "<optional known motif; else null>",
  "notes": "<optional caveats, e.g. 'partially occluded on the left'; else null>"
}}

## Scene Brief structure

The `scene_brief` string IS markdown. Write these six `##` headings:

## Overall silhouette
Two to three paragraphs: what the object is, dominant shapes, style/era.
Brief a 3D artist who has not seen the photo.

## Proportions
Overall aspect ratio and relative sizes of major components with fractions
(e.g. "seat ~1/3 of total height; legs occupy bottom half").

## Materials and color palette
PBR cues + dominant colors with hex when confident. Note how materials
differ across parts.

## Layout and symmetry
How parts attach, which repeat, symmetry axes, and **explicit front → +Z**
orientation (which face of the object is the canonical front).

## Landmark checklist
Bullet list of every distinctive landmark from step 3, each mapped to a
`parts[].name`. If a landmark has no part, you failed — go back and add it.

## Build order
Numbered list of part names in the order the Coder should create them
(root body first, then attachments, then decoration). Example:
1. body  2. lid  3. handle  4. spout  5. base_ring

## Per-part fields

For each `parts[]` entry:

**Structured fields (required):**
- `primitive` — pick ONE that maps cleanly to Three.js:
    box/extrude → panels, boards, blades, folded sheets
    cylinder/cone → legs, necks, spouts, posts
    sphere/capsule → bulbs, knobs, soft cushions
    torus → rings, brims, tires
    lathe → bottles, pots, vases, goblets, rotational vessels
    tube → handles, pipes, cables, curved frames
    instanced → repeated identical parts (legs, spokes, petals)
- `attach_to` — `'root'` for the main body group, else the parent `name`
- `size_frac` — concrete relative size (`~0.3H`, `~0.5W`, …)
- `color_hex` — `#rrggbb` when confident, else null
- `material` — one of the PBR phrases listed in the schema above
- `motif_role` — `body` / `support` / `enclosure` / `landmark` / `decoration`
- `count_hint` — short structured phrase for repetition

**Narrative (2–5 sentences):** expand the structured fields into artist prose:
shape details, exact placement ("upper-front rim, protruding +Z"), modifiers
(taper, bend, pinch), and how it contacts the parent.

## Decomposition granularity

Think like an exploded-view technical illustrator:
  - Simple objects (ball, cup, stool): 3–8 parts.
  - Medium objects (chair, lamp, bottle-with-handle): 6–15 parts.
  - Complex objects (vehicle, appliance, instrument): 12–25 parts.

If your first pass is below the range, re-scan for missed landmarks.

## Commonly-missed part types (scan before finalizing)

  - Linear: cables, tubes, handles, rods, straps, antennas → tube/cylinder
  - Transparent: glass, windows, liquid → clear/frosted glass material
  - Repeated: legs, wheels, petals, slats → one part + count_hint
  - Trim: rings, bands, piping, rivets, seams → decoration role
  - Caps/endings: lids, knobs, foot pads, tips → landmark role
  - Thin sheets: tags, cards, paper with holes/folds → extrude + hole

## Rules

- Return ONLY a single JSON object — no prose outside JSON, no markdown fences.
- `scene_brief` IS markdown inside the JSON string.
- Collapse truly identical repeats into ONE part with count_hint.
- Do not describe background, lighting, camera, or mood.
- Do not invent hidden parts that are not structurally necessary.
- Every visible landmark MUST appear in both `parts[]` and the Landmark checklist.
- Every part MUST have `primitive`, `attach_to`, and `size_frac` filled.
"""
