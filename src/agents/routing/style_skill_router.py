"""Select style CV sub-skills via vision LLM + skills/style/SKILL.md."""
from __future__ import annotations

from typing import Any, Dict, List

from graph.schema import VideoCase

from agents.routing.router_llm import vision_select_subskills

STYLE_SKILL_ORDER: List[str] = ["fft"]

_FFT_FALLBACK: List[str] = ["fft"]

_LLM_INTRO = """You are a routing policy for a video authenticity analysis pipeline.

Choose whether to run **FFT / spectral** style analysis for these frames."""

_LLM_RULES = """Rules:
- Allowed ids: fft only.
- Include **fft** unless the preview is unusable (e.g. solid black) or policy forbids any extra CV.
- If fft is appropriate, return ["fft"]."""


def select_style_skill_ids(
    case: VideoCase,
    artifacts: Dict[str, Any],
    config: Dict[str, Any],
    preview_images: List[str],
) -> List[str]:
    return vision_select_subskills(
        pack_name="style",
        skill_order=STYLE_SKILL_ORDER,
        llm_failure_fallback=_FFT_FALLBACK,
        empty_selection_fallback=_FFT_FALLBACK,
        case=case,
        artifacts=artifacts,
        config=config,
        intro=_LLM_INTRO,
        rules=_LLM_RULES,
        preview_images=preview_images,
    )
