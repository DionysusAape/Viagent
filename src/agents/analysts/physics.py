"""Physics/commonsense agent for detecting violations of objective physical laws across frames."""
from graph.constants import AgentKey
from graph.schema import AgentResult, GraphState, EvidenceItem, AnalystLLMOutput
from llm.inference import call_llm
from llm.prompt import PHYSICS_PROMPT
from util.logger import logger
from util.skills_config import cv_skills_enabled
from util.frame_sampling import sample_frames_for_llm
from skill.physics.optical_flow import analyze_frames_optical_flow_batch, format_optical_flow_for_prompt
from skill.physics.geometry_stability_check import (
    analyze_frames_geometry_stability_batch,
    format_geometry_stability_for_prompt,
)
from skill.physics.nsg_lite_video_analyzer import NSGLiteVideoAnalyzer, format_nsg_lite_for_prompt
from agents.routing.router_llm import pick_router_preview_frames, ROUTER_PREVIEW_MAX_FRAMES
from agents.routing.physics_skill_router import select_physics_skill_ids


def physics_agent(state: GraphState) -> GraphState:
    """Physics analysis agent checking objective physical plausibility."""
    agent_name = AgentKey.PHYSICS
    case = state["case"]
    artifacts = state["artifacts"]
    config = state["config"]

    logger.log_agent_status(agent_name, case.case_id, "Analyzing physical plausibility")

    frame_inputs = artifacts.get("frame_inputs")

    all_valid_frames = [frame for frame in frame_inputs if frame is not None]
    llm_conf = config.get("llm", {})
    max_images = int(llm_conf.get("max_images") or 50)
    sampled_frames = sample_frames_for_llm(all_valid_frames, max_images)

    frame_count = len(sampled_frames)

    frame_labels = [f"Frame {index + 1}" for index in range(frame_count)]
    frame_labels_str = ", ".join(frame_labels)

    if cv_skills_enabled(config):
        max_router_preview = int(
            llm_conf.get("skill_router_max_preview_frames", ROUTER_PREVIEW_MAX_FRAMES)
        )
        router_preview = pick_router_preview_frames(sampled_frames, max_router_preview)
        selected_skill_ids = select_physics_skill_ids(case, artifacts, config, router_preview)
        active = set(selected_skill_ids)
        logger.info(f"{case.case_id} physics sub-skills active: {selected_skill_ids}")
    else:
        active = set()
        logger.info(
            f"{case.case_id} physics: CV sub-skills disabled (enable_skills: false), vision-only"
        )

    physics_config = config.get("physics", {})

    optical_flow_description = (
        "**Optical Flow Analysis Evidence:**\n\n"
        "Optical flow analysis was not run for this case (skill routing).\n\n"
    )
    if "optical_flow" in active:
        try:
            optical_flow_results = analyze_frames_optical_flow_batch(
                sampled_frames, config=physics_config
            )
            optical_flow_description = format_optical_flow_for_prompt(
                optical_flow_results, frame_labels
            )
        except (ValueError, IOError) as error:
            logger.error(f"Optical flow analysis failed for {case.case_id}: {error}")
            raise RuntimeError(f"Optical flow analysis failed: {error}") from error

    geometry_stability_description = (
        "**Geometry Stability Analysis Evidence:**\n\n"
        "Geometry stability analysis was not run for this case (skill routing).\n\n"
    )
    if "geometry_stability" in active:
        try:
            geometry_results = analyze_frames_geometry_stability_batch(
                sampled_frames, config=physics_config
            )
            geometry_stability_description = format_geometry_stability_for_prompt(
                geometry_results, frame_labels
            )
        except (ValueError, IOError) as error:
            logger.warning(
                f"Geometry stability check failed for {case.case_id}: {error}, continuing without it"
            )
            geometry_stability_description = (
                "**Geometry Stability Analysis Evidence:**\n\n"
                "Geometry stability check not available.\n"
            )

    nsg_description = (
        "**NSG-lite Physical Consistency Evidence:**\n\n"
        "NSG-lite analysis was not run for this case (skill routing).\n\n"
    )
    if "nsg_lite" in active:
        try:
            nsg_analyzer = NSGLiteVideoAnalyzer()
            nsg_result = nsg_analyzer.analyze_frames_base64(sampled_frames)
            nsg_description = format_nsg_lite_for_prompt(nsg_result, frame_labels)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning(
                f"NSG-lite analysis failed for {case.case_id}: {error}, continuing without it"
            )
            nsg_description = (
                "**NSG-lite Physical Consistency Evidence:**\n\n"
                "NSG-lite analysis not available.\n"
            )

    prompt = PHYSICS_PROMPT.format(
        frame_count=frame_count,
        frame_labels=frame_labels_str,
        optical_flow_evidence=optical_flow_description,
        geometry_stability_evidence=geometry_stability_description,
        nsg_evidence=nsg_description,
    )

    llm_response = call_llm(
        prompt=prompt,
        config=config["llm"],
        pydantic_model=AnalystLLMOutput,
        images=sampled_frames,
    )

    evidence_items = []
    for item in llm_response.evidence or []:
        if not isinstance(item, dict):
            continue
        evidence_items.append(
            EvidenceItem(
                agent=agent_name,
                type=item.get("type", "unknown"),
                detail=item.get("detail", ""),
                score=item.get("score", 0.0),
            )
        )

    result = AgentResult(
        agent=agent_name,
        status="ok",
        score_fake=llm_response.score_fake,
        confidence=llm_response.confidence,
        evidence=evidence_items,
        error=None,
    )

    logger.info(
        f"Physics analysis for {case.case_id}: "
        f"score_fake={result.score_fake:.3f}, confidence={result.confidence:.3f}, "
        f"reasoning: {llm_response.reasoning}",
    )
    logger.log_agent_status(agent_name, case.case_id, "completed")

    return {"results": {agent_name: result}}
