"""Style/source analysis agent for detecting global CG/animation rendering style."""
from graph.constants import AgentKey
from graph.schema import AgentResult, GraphState, EvidenceItem, AnalystLLMOutput
from llm.inference import call_llm
from llm.prompt import STYLE_PROMPT
from util.logger import logger
from util.skills_config import cv_skills_enabled
from util.frame_sampling import sample_frames_for_llm
from skill.style.fft_analysis import analyze_frames_fft_batch, format_fft_features_for_prompt
from agents.routing.router_llm import pick_router_preview_frames, ROUTER_PREVIEW_MAX_FRAMES
from agents.routing.style_skill_router import select_style_skill_ids


def style_agent(state: GraphState) -> GraphState:
    """Style analysis agent analyzing global visual style and source type."""
    agent_name = AgentKey.STYLE
    case = state["case"]
    artifacts = state["artifacts"]
    config = state["config"]

    logger.log_agent_status(agent_name, case.case_id, "Analyzing global style and source type")

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
        selected_skill_ids = select_style_skill_ids(case, artifacts, config, router_preview)
        active = set(selected_skill_ids)
        logger.info(f"{case.case_id} style sub-skills active: {selected_skill_ids}")
    else:
        active = set()
        logger.info(
            f"{case.case_id} style: CV sub-skills disabled (enable_skills: false), vision-only"
        )

    fft_description = (
        "**FFT / frequency-domain style cues:**\n\n"
        "FFT analysis was not run for this case (skill routing).\n\n"
    )
    if "fft" in active:
        try:
            fft_features_list = analyze_frames_fft_batch(sampled_frames)
            fft_description = format_fft_features_for_prompt(fft_features_list, frame_labels)
        except (ValueError, IOError) as error:
            logger.error(f"FFT analysis failed for {case.case_id}: {error}")
            raise RuntimeError(f"FFT analysis failed: {error}") from error

    prompt = STYLE_PROMPT.format(
        frame_count=frame_count,
        frame_labels=frame_labels_str,
        fft_analysis=fft_description,
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
        f"Style analysis for {case.case_id}: "
        f"score_fake={result.score_fake:.3f}, confidence={result.confidence:.3f}, "
        f"reasoning: {llm_response.reasoning}",
    )
    logger.log_agent_status(agent_name, case.case_id, "completed")

    return {"results": {agent_name: result}}
