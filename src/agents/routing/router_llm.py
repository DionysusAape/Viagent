"""Sub-skill routing: multimodal LLM uses preview frames + src/skill/<agent>/SKILL.md."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Set

from graph.schema import SubskillRouterOutput, VideoCase
from llm.inference import call_llm
from util.agent_skill_pack import (
    extract_viagent_routing,
    load_agent_skill_pack,
    routing_subskills,
)
from util.logger import logger

from agents.routing.video_routing_context import build_video_routing_context

# How many sampled frames to show the router (chronologically spread).
ROUTER_PREVIEW_MAX_FRAMES = 4


def pick_router_preview_frames(
    frame_data_urls: List[Any],
    max_n: int = ROUTER_PREVIEW_MAX_FRAMES,
) -> List[str]:
    """Pick up to max_n non-None data URLs spread across the sequence."""
    valid: List[str] = [x for x in frame_data_urls if x]
    if not valid:
        return []
    if len(valid) == 1:
        return valid
    cap = min(max_n, len(valid))
    idxs = sorted(
        {
            min(round(i * (len(valid) - 1) / max(cap - 1, 1)), len(valid) - 1)
            for i in range(cap)
        }
    )
    return [valid[i] for i in idxs]


def vision_select_subskills(
    *,
    pack_name: str,
    skill_order: List[str],
    llm_failure_fallback: List[str],
    empty_selection_fallback: List[str],
    case: VideoCase,
    artifacts: Dict[str, Any],
    config: Dict[str, Any],
    intro: str,
    rules: str,
    preview_images: List[str],
) -> List[str]:
    """
    One multimodal LLM call: representative frames + JSON context + SKILL.md policy.
    Picks a subset of skill ids; never returns the full set unless the model chooses it.
    """
    valid: Set[str] = set(skill_order)
    _, body = load_agent_skill_pack(pack_name)
    routing = extract_viagent_routing(body)
    subskills = routing_subskills(routing)
    if not subskills:
        logger.warning(
            f"{pack_name}: no viagent_routing in src/skill/{pack_name}/SKILL.md; "
            f"using conservative fallback {empty_selection_fallback}"
        )
        return list(empty_selection_fallback)

    policy_text = "\n".join(f"- {s['id']}: {s['when']}" for s in subskills)
    ctx = build_video_routing_context(case, artifacts)
    if preview_images:
        vision_note = (
            f"You are given {len(preview_images)} representative frame image(s) from the video "
            "in chronological order (first = earliest). Base your decision on what you **see** "
            "in those images, together with the policy below and the JSON context."
        )
    else:
        vision_note = (
            "No frame images are attached (metadata-only). Use the JSON context and policy; "
            "prefer a smaller, conservative set of sub-skills."
        )

    prompt = f"""{intro}

{vision_note}

Allowed ids (exact strings): {", ".join(skill_order)}

Video context (JSON):
{json.dumps(ctx, ensure_ascii=False, indent=2)}

Sub-skill policy (from SKILL.md):
{policy_text}

{rules}

Respond with structured output: selected_skill_ids (list) and rationale (short)."""

    llm_conf = config.get("llm") or {}
    router_temp = llm_conf.get("skill_router_temperature")
    llm_block = {
        "provider": llm_conf.get("provider"),
        "model": llm_conf.get("model"),
        "temperature": float(
            router_temp if router_temp is not None else llm_conf.get("temperature", 0.2)
        ),
        "max_retries": llm_conf.get("max_retries", 3),
    }

    try:
        out = call_llm(
            prompt,
            llm_block,
            SubskillRouterOutput,
            images=preview_images if preview_images else None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"{pack_name} vision skill router failed: {exc}; using fallback {llm_failure_fallback}"
        )
        return list(llm_failure_fallback)

    picked = {x.strip() for x in out.selected_skill_ids if x.strip() in valid}
    ordered = [x for x in skill_order if x in picked]
    if not ordered:
        logger.warning(
            f"{pack_name} vision router returned no valid ids (rationale={out.rationale!r}); "
            f"using {empty_selection_fallback}"
        )
        return list(empty_selection_fallback)

    logger.info(f"{pack_name} vision router for {case.case_id}: {ordered} — {out.rationale[:200]}")
    return ordered
