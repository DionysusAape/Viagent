"""Select spatial CV sub-skills via vision LLM + src/skill/spatial/SKILL.md."""
from __future__ import annotations

from typing import Any, Dict, List

from graph.schema import VideoCase

from agents.routing.router_llm import vision_select_subskills

SPATIAL_SKILL_ORDER: List[str] = [
    "ela",
    "patch",
    "boundary",
    "edge",
    "multi_face",
    "blur",
]

_GENERAL_FALLBACK: List[str] = ["ela", "patch", "boundary", "edge"]

_LLM_INTRO = """You are a routing policy for a video authenticity analysis pipeline.

Choose which **spatial** (single-frame) CV sub-skills to run for THIS video. Only pick ids that are likely useful; omit ids that are inappropriate for what you see."""

_LLM_RULES = """Rules:
- Return only ids from the allowed list; each id at most once.
- Prefer a **minimal** useful set; do not select skills that clearly do not apply to the visible content.
- Omit face-specific skills (multi_face, blur) when there are no people / faces visible.
- For screencasts, slides, or text-only frames, omit face-specific skills and deprioritize patch/ELA if content is purely flat UI (still may keep edge/boundary if useful)."""


def select_spatial_skill_ids(
    case: VideoCase,
    artifacts: Dict[str, Any],
    config: Dict[str, Any],
    preview_images: List[str],
) -> List[str]:
    return vision_select_subskills(
        pack_name="spatial",
        skill_order=SPATIAL_SKILL_ORDER,
        llm_failure_fallback=_GENERAL_FALLBACK,
        empty_selection_fallback=_GENERAL_FALLBACK,
        case=case,
        artifacts=artifacts,
        config=config,
        intro=_LLM_INTRO,
        rules=_LLM_RULES,
        preview_images=preview_images,
    )
