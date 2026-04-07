"""Select physics CV sub-skills via vision LLM + src/skill/physics/SKILL.md."""
from __future__ import annotations

from typing import Any, Dict, List

from graph.schema import VideoCase

from agents.routing.router_llm import vision_select_subskills

PHYSICS_SKILL_ORDER: List[str] = ["optical_flow", "geometry_stability", "nsg_lite"]

_PHYSICS_FALLBACK: List[str] = ["optical_flow", "geometry_stability"]

_LLM_INTRO = """You are a routing policy for a video authenticity analysis pipeline.

Choose which **physics / motion** CV sub-skills to run based on the preview frames and context."""

_LLM_RULES = """Rules:
- Return only ids from the allowed list; each id at most once.
- **optical_flow**: skip mainly for static slide / still-image style content with no motion.
- **geometry_stability**: less meaningful for flat 2D animation; still useful for perspective scenes.
- **nsg_lite**: extra compute; omit if content is extremely static or you want a lighter pass.
- Prefer a minimal sufficient set."""


def select_physics_skill_ids(
    case: VideoCase,
    artifacts: Dict[str, Any],
    config: Dict[str, Any],
    preview_images: List[str],
) -> List[str]:
    return vision_select_subskills(
        pack_name="physics",
        skill_order=PHYSICS_SKILL_ORDER,
        llm_failure_fallback=list(PHYSICS_SKILL_ORDER),
        empty_selection_fallback=_PHYSICS_FALLBACK,
        case=case,
        artifacts=artifacts,
        config=config,
        intro=_LLM_INTRO,
        rules=_LLM_RULES,
        preview_images=preview_images,
    )
