"""Select temporal CV sub-skills via vision LLM + skills/temporal/SKILL.md."""
from __future__ import annotations

from typing import Any, Dict, List

from graph.schema import VideoCase

from agents.routing.router_llm import vision_select_subskills

TEMPORAL_SKILL_ORDER: List[str] = ["lpc", "feature_stability"]

_TEMPORAL_FALLBACK: List[str] = ["lpc"]

_LLM_INTRO = """You are a routing policy for a video authenticity analysis pipeline.

Choose which **temporal** CV sub-skills to run. These need frame sequences on disk; you only decide policy from preview images + context."""

_LLM_RULES = """Rules:
- Return only ids from the allowed list; each id at most once.
- **lpc** is for phase coherence around faces — omit if no faces or only non-face content visible.
- **feature_stability** tracks texture features — omit for static slides or nearly blank frames.
- Prefer minimal set; when unsure and faces or motion are visible, you may include both."""


def select_temporal_skill_ids(
    case: VideoCase,
    artifacts: Dict[str, Any],
    config: Dict[str, Any],
    preview_images: List[str],
) -> List[str]:
    return vision_select_subskills(
        pack_name="temporal",
        skill_order=TEMPORAL_SKILL_ORDER,
        llm_failure_fallback=list(TEMPORAL_SKILL_ORDER),
        empty_selection_fallback=_TEMPORAL_FALLBACK,
        case=case,
        artifacts=artifacts,
        config=config,
        intro=_LLM_INTRO,
        rules=_LLM_RULES,
        preview_images=preview_images,
    )
