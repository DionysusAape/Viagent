"""Physics/commonsense agent for detecting violations of objective physical laws across frames."""
from graph.constants import AgentKey
from graph.schema import AgentResult, GraphState, EvidenceItem, AnalystLLMOutput
from llm.inference import call_llm
from llm.prompt import PHYSICS_PROMPT
from util.logger import logger
from util.frame_sampling import sample_frames_for_llm
from skill.physics.optical_flow import analyze_frames_optical_flow_batch, format_optical_flow_for_prompt
from skill.physics.geometry_stability_check import analyze_frames_geometry_stability_batch, format_geometry_stability_for_prompt
from skill.physics.nsg_lite_video_analyzer import NSGLiteVideoAnalyzer, format_nsg_lite_for_prompt


def physics_agent(state: GraphState) -> GraphState:
    """Physics analysis agent checking objective physical plausibility."""
    agent_name = AgentKey.PHYSICS
    case = state["case"]
    artifacts = state["artifacts"]
    config = state["config"]

    logger.log_agent_status(agent_name, case.case_id, "Analyzing physical plausibility")

    frame_inputs = artifacts.get("frame_inputs")

    # Filter out None frames, 再根据上限采样
    all_valid_frames = [frame for frame in frame_inputs if frame is not None]
    llm_conf = config.get("llm", {})
    max_images = int(llm_conf.get("max_images") or 50)
    sampled_frames = sample_frames_for_llm(all_valid_frames, max_images)

    # Prepare frame information
    frame_count = len(sampled_frames)

    # Format frame labels for prompt
    frame_labels = [f"Frame {index + 1}" for index in range(frame_count)]
    frame_labels_str = ", ".join(frame_labels)

    # Run optical flow analysis (required dependency: opencv-contrib-python)
    try:
        physics_config = config.get("physics", {})
        optical_flow_results = analyze_frames_optical_flow_batch(sampled_frames, config=physics_config)
        optical_flow_description = format_optical_flow_for_prompt(optical_flow_results, frame_labels)
    except (ValueError, IOError) as error:
        logger.error(f"Optical flow analysis failed for {case.case_id}: {error}")
        raise RuntimeError(f"Optical flow analysis failed: {error}") from error

    # Run geometry stability check (required dependency: opencv-contrib-python)
    geometry_stability_description = ""
    try:
        geometry_results = analyze_frames_geometry_stability_batch(
            sampled_frames, config=physics_config
        )
        geometry_stability_description = format_geometry_stability_for_prompt(
            geometry_results, frame_labels
        )
    except (ValueError, IOError) as error:
        logger.warning("Geometry stability check failed for {case.case_id}: {error}, continuing without it")
        geometry_stability_description = ("**Geometry Stability Analysis Evidence:**\n\nGeometry stability check not available.\n")

    # Run NSG-lite physical consistency analysis (CPU-friendly)
    nsg_description = ""
    try:
        nsg_analyzer = NSGLiteVideoAnalyzer()
        nsg_result = nsg_analyzer.analyze_frames_base64(sampled_frames)
        nsg_description = format_nsg_lite_for_prompt(nsg_result, frame_labels)
    except Exception as error:  # pragma: no cover - defensive
        logger.warning(f"NSG-lite analysis failed for {case.case_id}: {error}, continuing without it")
        nsg_description = ("**NSG-lite Physical Consistency Evidence:**\n\nNSG-lite analysis not available.\n")

    # Format prompt with frame information and algorithmic evidence
    prompt = PHYSICS_PROMPT.format(
        frame_count=frame_count,
        frame_labels=frame_labels_str,
        optical_flow_evidence=optical_flow_description,
        geometry_stability_evidence=geometry_stability_description,
        nsg_evidence=nsg_description,
    )

    # Call LLM with structured output (with frame images)
    llm_response = call_llm(
        prompt=prompt,
        config=config["llm"],
        pydantic_model=AnalystLLMOutput,
        images=sampled_frames,
    )

    # Convert LLM output directly to AgentResult
    result = AgentResult(
        agent=agent_name,
        status="ok",
        score_fake=llm_response.score_fake,
        confidence=llm_response.confidence,
        evidence=[
            EvidenceItem(
                agent=agent_name,
                type=item.get("type", "unknown"),
                detail=item.get("detail", ""),
                score=item.get("score", 0.0),
            )
            for item in llm_response.evidence
        ],
        error=None,
    )

    logger.info(
        f"Physics analysis for {case.case_id}: "
        f"score_fake={result.score_fake:.3f}, confidence={result.confidence:.3f}, "
        f"reasoning: {llm_response.reasoning}",
    )
    logger.log_agent_status(agent_name, case.case_id, "completed")

    # Return only the new result entry (LangGraph will merge it with existing results)
    return {"results": {agent_name: result}}
